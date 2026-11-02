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
