"""Question classifier: DistilBERT fine-tuned on the generated questions, in plain PyTorch.

Fine-tuning in one paragraph: DistilBERT is a pretrained transformer (66M parameters)
that turns text into vectors capturing its meaning. We put a small new layer on top
that maps the vector of a question to 6 scores, one per category, and then train the
whole thing on our labeled questions: show a batch, measure how wrong the scores are
(the loss), compute how each weight should change to make them less wrong (gradients,
via backpropagation), nudge every weight a little in that direction (the optimizer
step), repeat. The new layer learns from scratch; the pretrained part only shifts
slightly, keeping what it already knows about English.

The pieces, in the order they are used:
1. pick_device       where the math runs (your Mac's GPU via MPS, else CPU)
2. split_train_val   hold out part of the training data to choose the best epoch
3. make_loader       text -> batches of token ids the model can read
4. train_epoch       one pass over the training data (the training loop itself)
5. predict           scores -> category names, for validation and testing
6. fine_tune         ties 1-5 together, keeps the best epoch
7. BertClassifier    the trained model as a `question -> category` function, saved/loaded
"""

import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup,
)

from sklearn_assistant.classifier.data import CATEGORIES, LabeledQuestion

BASE_MODEL = "distilbert-base-uncased"  # "uncased": lowercases text, so "Ridge" == "ridge"
MAX_LEN = 64          # tokens per question; the longest training question has 50
BATCH_SIZE = 32       # questions per training step
EPOCHS = 4            # passes over the training data
LEARNING_RATE = 5e-5  # step size; the usual range for fine-tuning BERT models is 2e-5 to 5e-5
WEIGHT_DECAY = 0.01   # gently pulls weights toward 0, a mild guard against overfitting
WARMUP_FRACTION = 0.1  # share of steps during which the learning rate ramps up from 0
VAL_FRACTION = 0.15   # share of the training data held out for choosing the best epoch

# Categories <-> integer ids. The model only works with numbers: category i is output
# number i. Stored in the saved model's config, so a loaded model maps ids back to names.
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


# ---- 1. Device -------------------------------------------------------------------

def pick_device() -> torch.device:
    """Apple GPU (MPS) when available, else CPU. CUDA isn't checked: this targets a Mac."""
    # Training is ~5-10x faster on the GPU. Inference on single short questions is fast
    # on either, which is why the server (AWS notes in the README) can use CPU only.
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---- 2. Train / validation split ------------------------------------------------

def split_train_val(questions: list[LabeledQuestion], seed: int = 0
                    ) -> tuple[list[LabeledQuestion], list[LabeledQuestion]]:
    """Hold out VAL_FRACTION of the generated questions, same category mix in both parts.

    The validation part decides which epoch's weights to keep. It must not be the test
    set: choosing by test score would make the test score optimistic.
    """
    # stratify= keeps each category's share equal in both parts.
    train, val = train_test_split(questions, test_size=VAL_FRACTION, random_state=seed,
                                  stratify=[q.category for q in questions])
    return train, val


# ---- 3. Batches ------------------------------------------------------------------

def make_loader(questions: list[LabeledQuestion], tokenizer, shuffle: bool) -> DataLoader:
    """Batches of (token ids, attention mask, label ids), tokenized on the fly."""

    def collate(batch: list[LabeledQuestion]) -> dict[str, torch.Tensor]:
        # The tokenizer splits text into sub-word pieces from DistilBERT's 30k vocabulary
        # and maps each to an id: "How do I scale?" -> [CLS] how do i scale ? [SEP]
        # -> [101, 2129, 2079, 1045, 4094, 1029, 102]. [CLS] is a special first token
        # whose final vector is used to represent the whole question for classification.
        enc = tokenizer([q.question for q in batch],
                        # A batch is one rectangular tensor, so shorter questions are
                        # padded to the longest one *in this batch* (not to MAX_LEN),
                        # which keeps batches small and fast.
                        padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            # 1 for real tokens, 0 for padding, so the model ignores the padding.
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([LABEL2ID[q.category] for q in batch]),
        }

    # DataLoader groups the list into batches and calls collate on each. Shuffling
    # the training data each epoch stops the model from learning the order of examples.
    return DataLoader(questions, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)


# ---- 4. Training loop ------------------------------------------------------------

def train_epoch(model, loader: DataLoader, optimizer, scheduler, device: torch.device) -> float:
    """One pass over the training data. Returns the mean loss."""
    # train() switches on training-only behavior, chiefly dropout: randomly zeroing some
    # activations, so the model can't rely on any single one (less overfitting).
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}  # tensors must sit on the device
        # Forward pass: token ids -> transformer -> new layer -> 6 raw scores ("logits")
        # per question, shape (batch size, 6).
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        # Cross-entropy loss: softmax turns the scores into probabilities, and the loss
        # is -log(probability given to the correct category). Near 0 when the model is
        # confidently right, large when it's confidently wrong; averaged over the batch.
        loss = F.cross_entropy(logits, batch["labels"])

        # Backward pass: PyTorch computes, for every one of the 66M weights, how the loss
        # would change if that weight changed (its gradient).
        loss.backward()
        # If one batch produces huge gradients, scale them down so one step can't wreck
        # the pretrained weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Move every weight a small step against its gradient (AdamW adapts the step
        # size per weight), then update the learning rate for the next step.
        optimizer.step()
        scheduler.step()
        # Gradients accumulate by default; clear them before the next batch.
        optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)


# ---- 5. Prediction ---------------------------------------------------------------

@torch.no_grad()  # no gradients needed when only predicting: faster, less memory
def predict(model, tokenizer, texts: list[str], device: torch.device) -> list[str]:
    """Category names for a list of questions."""
    # eval() turns dropout off, so predictions are deterministic.
    model.eval()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(device)
    logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    # The highest score wins (softmax isn't needed: it doesn't change which is highest).
    return [ID2LABEL[i] for i in logits.argmax(dim=-1).tolist()]


def predict_all(model, tokenizer, questions: list[LabeledQuestion], device: torch.device,
                batch_size: int = 128) -> list[str]:
    """predict() in chunks, so a large validation set doesn't need one giant batch."""
    texts = [q.question for q in questions]
    return [label for i in range(0, len(texts), batch_size)
            for label in predict(model, tokenizer, texts[i:i + batch_size], device)]


# ---- 6. Fine-tuning --------------------------------------------------------------

def fine_tune(train: list[LabeledQuestion], val: list[LabeledQuestion],
              epochs: int = EPOCHS, seed: int = 0, log=print):
    """Fine-tune DistilBERT on `train`, keeping the epoch with the best validation macro-F1.

    Returns (model, tokenizer, history), where history has one dict per epoch.
    """
    # Same seed -> same initial new-layer weights and same shuffling. (MPS still isn't
    # bit-for-bit deterministic, so repeated runs can differ slightly.)
    torch.manual_seed(seed)
    device = pick_device()

    # Downloads once to ~/.cache/huggingface/, then loads from disk. This is where the
    # "MISSING: classifier.weight ..." notice comes from: the new 6-way layer is
    # created with random weights, and the pretraining head ("UNEXPECTED: vocab_*")
    # is dropped.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    train_loader = make_loader(train, tokenizer, shuffle=True)
    # AdamW: the standard optimizer for transformers. Biases and LayerNorm weights are
    # commonly excluded from weight decay; with decay this small the difference is
    # negligible, so all weights get it here for simplicity.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    # Learning rate schedule: ramp up from 0 over the first 10% of steps (big early steps
    # on random new-layer weights could disturb the pretrained ones), then decrease
    # linearly to 0 by the end, for smaller, finer steps as training converges.
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_FRACTION * total_steps),
        num_training_steps=total_steps)

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        # Check on data the model didn't train on. Training loss keeps falling as the
        # model memorizes; validation F1 shows whether it still generalizes.
        predicted = predict_all(model, tokenizer, val, device)
        val_f1 = f1_score([q.category for q in val], predicted, labels=CATEGORIES,
                          average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1,
                        "epoch_seconds": time.perf_counter() - start})
        log(f"epoch {epoch}/{epochs}  train loss {train_loss:.4f}  "
            f"val macro-F1 {val_f1:.3f}  ({history[-1]['epoch_seconds']:.0f} s)")
        if val_f1 > best_f1:
            # Remember this epoch's weights (a CPU copy, so later epochs don't overwrite it).
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Roll back to the best epoch if a later one got worse (the start of overfitting).
    model.load_state_dict(best_state)
    return model, tokenizer, history


# ---- 7. The trained classifier ---------------------------------------------------

class BertClassifier:
    """A fine-tuned model as a `question -> category` function (the interface evaluate() uses)."""

    def __init__(self, model, tokenizer, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        # A copy, so moving it to another device doesn't affect the caller's model.
        self.model = copy.deepcopy(model).to(self.device)
        self.tokenizer = tokenizer

    def __call__(self, question: str) -> str:
        return predict(self.model, self.tokenizer, [question], self.device)[0]

    def save(self, path: Path) -> None:
        """Weights, config (incl. the category names) and tokenizer, in Hugging Face format."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "BertClassifier":
        # Same loading call as for the base model, but from our folder: every weight
        # is present this time, so there are no MISSING/UNEXPECTED notices.
        return cls(AutoModelForSequenceClassification.from_pretrained(path),
                   AutoTokenizer.from_pretrained(path), device)
