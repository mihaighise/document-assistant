"""Step 5: fine-tune DistilBERT as the question classifier and score it.

    python scripts/train_bert_classifier.py              # ~1 min on an M-series Mac
    python scripts/train_bert_classifier.py --epochs 3 --seed 1
    mlflow ui --backend-store-uri sqlite:///mlflow.db    # experiment "classifier"

Trains on datasets/classifier/train.jsonl (minus a validation part), tests on the same
105 hand-written questions as TF-IDF and the LLMs, logs to MLflow and saves the model
to data/models/distilbert/ (gitignored, ~260 MB).
"""

import argparse
import time

import torch

from sklearn_assistant.classifier import bert
from sklearn_assistant.classifier.data import load_test, load_train
from sklearn_assistant.config import CLASSIFIER_TRAIN_PATH, MODELS_DIR
from sklearn_assistant.evaluation.classifier import evaluate, print_report
from sklearn_assistant.tracking import file_sha, log_classifier_run

MODEL_DIR = MODELS_DIR / "distilbert"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=bert.EPOCHS)
    # Different seeds = different random new-layer weights, shuffling and split. Running
    # a few shows how much of a score difference is just luck.
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The generated questions are split into train and validation; the hand-written
    # test questions are only touched once, at the very end.
    train, val = bert.split_train_val(load_train(), seed=args.seed)
    test = load_test()
    print(f"{len(train)} train / {len(val)} validation questions, device: {bert.pick_device()}")

    start = time.perf_counter()
    model, tokenizer, history = bert.fine_tune(train, val, epochs=args.epochs, seed=args.seed)
    train_seconds = time.perf_counter() - start
    best = max(history, key=lambda h: h["val_macro_f1"])

    # Score on CPU, one question at a time: that's how a server without a GPU would
    # run it (see the AWS notes), so the latency is the deployable number.
    classifier = bert.BertClassifier(model, tokenizer, device=torch.device("cpu"))
    classifier("warm up")  # the first call is slow (memory allocation); don't time it
    summary, rows, matrix = evaluate(classifier, test)
    classifier.save(MODEL_DIR)

    log_classifier_run(
        "distilbert",
        params={"model": bert.BASE_MODEL, "epochs": args.epochs, "seed": args.seed,
                "best_epoch": best["epoch"], "batch_size": bert.BATCH_SIZE, "learning_rate": bert.LEARNING_RATE,
                "max_len": bert.MAX_LEN, "n_train": len(train), "n_val": len(val),
                "n_test": len(test), "train_set_sha": file_sha(CLASSIFIER_TRAIN_PATH),
                "train_device": str(bert.pick_device()), "test_device": "cpu"},
        metrics={**summary, "val_macro_f1": best["val_macro_f1"],
                 "train_seconds": train_seconds, "cost_per_1k_usd": 0.0},
        # The model folder is ~260 MB, so it stays in data/models/ instead of MLflow.
        rows=rows, matrix=matrix, history=history,
    )

    print(f"\nTrained in {train_seconds:.0f} s; kept epoch {best['epoch']} "
          f"(val macro-F1 {best['val_macro_f1']:.3f}). Saved to {MODEL_DIR}\n")
    print_report(summary, rows, matrix)


if __name__ == "__main__":
    main()
