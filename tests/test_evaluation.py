from sklearn_assistant.config import DOCS_BASE_URL
from sklearn_assistant.evaluation.retrieval import (
    Question, load_questions, missing_labels, score,
)


def test_score_counts_each_gold_section_once():
    # The same section appearing twice (two parts of it) doesn't double-count.
    s = score(["x", "a", "a", "y", "z", "b"], ["a", "b"])
    assert s["recall_at_5"] == 0.5
    assert s["recall_at_10"] == 1.0
    assert s["hit_at_5"] == 1.0
    assert s["mrr"] == 0.5


def test_score_with_no_hits_is_zero():
    assert score(["x", "y"], ["a"]) == {
        "recall_at_5": 0.0, "recall_at_10": 0.0, "hit_at_5": 0.0, "mrr": 0.0,
    }


def test_missing_labels_compares_relative_urls():
    questions = [Question("q1", "?", "howto", ["modules/svm.html#complexity", "gone.html#x"])]
    chunks = [{"url": DOCS_BASE_URL + "modules/svm.html#complexity"}]
    assert missing_labels(questions, chunks) == ["q1: gone.html#x"]


def test_eval_set_is_well_formed():
    questions = load_questions()
    assert len({q.id for q in questions}) == len(questions)
    assert all(q.relevant and q.question.strip() for q in questions)
    assert {q.category for q in questions} <= {
        "howto", "concept", "comparison", "api", "troubleshooting",
    }
