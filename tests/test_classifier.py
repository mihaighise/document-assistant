from sklearn_assistant.classifier import tfidf
from sklearn_assistant.classifier.data import (
    CATEGORIES, LabeledQuestion, drop_leakage, load_test, load_train,
)
from sklearn_assistant.evaluation.classifier import evaluate


def test_evaluate_counts_missing_categories_as_zero_f1():
    questions = [LabeledQuestion("a", "howto"), LabeledQuestion("b", "howto"),
                 LabeledQuestion("c", "api")]
    summary, rows, matrix = evaluate(lambda q: "howto", questions)
    assert summary["accuracy"] == 2 / 3
    assert summary["f1/api"] == 0.0
    # howto F1 = 0.8; the other five categories score 0 and still count in the average.
    assert abs(summary["macro_f1"] - 0.8 / len(CATEGORIES)) < 1e-9
    assert [r["correct"] for r in rows] == [True, True, False]
    assert matrix[CATEGORIES.index("api")][CATEGORIES.index("howto")] == 1


def test_drop_leakage_removes_near_copies_only():
    train = [{"question": "How do I scale my features?", "category": "howto"},
             {"question": "Ridge vs Lasso?", "category": "comparison"}]
    kept = drop_leakage(train, ["how do I scale my features"])
    assert [t["category"] for t in kept] == ["comparison"]


def test_pipeline_learns_keyword_cues():
    texts = ["how do I scale data", "how do I split data", "error when fitting model",
             "warning when fitting data"] * 3
    labels = ["howto", "howto", "troubleshooting", "troubleshooting"] * 3
    model = tfidf.build_pipeline().fit(texts, labels)
    assert model.predict(["error when I scale"])[0] == "troubleshooting"


def test_datasets_are_well_formed_and_disjoint():
    train, test = load_train(), load_test()
    assert {q.category for q in train} == set(CATEGORIES)
    assert {q.category for q in test} == set(CATEGORIES)
    test_questions = {q.question.lower() for q in test}
    assert not any(q.question.lower() in test_questions for q in train)


def test_claude_llm_cost_from_token_counts():
    from types import SimpleNamespace

    from sklearn_assistant.agent.llm import ClaudeLLM

    llm = ClaudeLLM(model="claude-haiku-5-5", client=object(), fallback=False)
    assert llm.extra == {}  # Haiku requests must not carry `fallbacks`
    for _ in range(2):
        llm._count(SimpleNamespace(usage=SimpleNamespace(input_tokens=500_000,
                                                         output_tokens=100_000)))
    # 1M input at $0.10 + 0.2M output at $0.50 per million tokens.
    assert llm.calls == 2 and abs(llm.cost_usd() - 0.20) < 1e-9


def test_bert_split_keeps_category_mix():
    from sklearn_assistant.classifier import bert

    questions = [LabeledQuestion(f"q{i}", CATEGORIES[i % len(CATEGORIES)]) for i in range(120)]
    train, val = bert.split_train_val(questions)
    assert len(val) == 18 and len(train) == 102
    assert {q.category for q in val} == set(CATEGORIES)  # every category in both parts
    assert not {q.question for q in train} & {q.question for q in val}


def test_bert_loader_pads_per_batch_and_encodes_labels():
    import torch

    from sklearn_assistant.classifier import bert

    def fake_tokenizer(texts, **kwargs):
        # One token per word, padded to the longest text in the batch.
        width = max(len(t.split()) for t in texts)
        ids = [[1] * len(t.split()) + [0] * (width - len(t.split())) for t in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(ids)}

    questions = [LabeledQuestion("a b c", "api"), LabeledQuestion("a", "howto")]
    batch = next(iter(bert.make_loader(questions, fake_tokenizer, shuffle=False)))
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].tolist() == [bert.LABEL2ID["api"], bert.LABEL2ID["howto"]]
