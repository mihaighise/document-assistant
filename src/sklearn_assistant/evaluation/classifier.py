"""Scoring a question classifier on the test set.

Any classifier is passed in as a plain `predict(question) -> category` function, so the
TF-IDF model, a fine-tuned BERT and the LLM (later in step 5) are scored by the same code.
"""

import time
from collections.abc import Callable
from statistics import median

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn_assistant.classifier.data import CATEGORIES, LabeledQuestion


def evaluate(predict: Callable[[str], str], questions: list[LabeledQuestion]
             ) -> tuple[dict[str, float], list[dict], list[list[int]]]:
    """Classify each test question one at a time.

    Returns metrics, one row per question, and the confusion matrix (rows = true
    category, columns = predicted, both in CATEGORIES order).
    """
    rows, latencies = [], []
    for q in questions:
        # One question per call, as in the agent, so latency is per-question.
        start = time.perf_counter()
        predicted = predict(q.question)
        latencies.append((time.perf_counter() - start) * 1000)
        rows.append({"question": q.question, "category": q.category, "predicted": predicted,
                     "correct": predicted == q.category})

    y_true = [r["category"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    # Passing labels= fixes the category order and makes categories absent from the
    # predictions still count (with F1 0) instead of being silently skipped.
    per_class = f1_score(y_true, y_pred, labels=CATEGORIES, average=None, zero_division=0)
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        # Macro-F1: F1 per category, then a plain average. The test set is mostly howto
        # (49 of 105), so accuracy alone would reward always guessing howto (47%);
        # macro-F1 weights the 4 troubleshooting questions as much as the 49 howto ones.
        # (F1 = harmonic mean of precision, "how often a prediction of X is right", and
        # recall, "how many real X were found".)
        "macro_f1": f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0),
        # Median, not mean: one slow call (e.g. first-call warm-up) shouldn't dominate.
        "latency_ms": median(latencies),
        **{f"f1/{c}": float(f) for c, f in zip(CATEGORIES, per_class)},
    }
    matrix = confusion_matrix(y_true, y_pred, labels=CATEGORIES).tolist()
    return summary, rows, matrix


def format_confusion(matrix: list[list[int]]) -> str:
    """Confusion matrix as a text table: rows are the true category, columns the prediction."""
    short = [c[:8] for c in CATEGORIES]
    lines = ["true \\ pred".ljust(16) + "".join(f"{c:>9}" for c in short)]
    for c, row in zip(CATEGORIES, matrix):
        lines.append(f"{c:16}" + "".join(f"{n:9d}" for n in row))
    return "\n".join(lines)


def print_report(summary: dict[str, float], rows: list[dict], matrix: list[list[int]]) -> None:
    """Console version of what gets logged to MLflow."""
    print(f"Test: {len(rows)} hand-written questions")
    print(f"  accuracy  {summary['accuracy']:.3f}\n  macro-F1  {summary['macro_f1']:.3f}\n"
          f"  latency   {summary['latency_ms']:.2f} ms/question\n")
    print("F1 by category")
    for c in CATEGORIES:
        print(f"  {c:16} {summary[f'f1/{c}']:.3f}")
    print("\n" + format_confusion(matrix))
    print("\nMisclassified:")
    for r in rows:
        if not r["correct"]:
            print(f"  [{r['category']} -> {r['predicted']}] {r['question']}")
