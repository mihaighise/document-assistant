"""Step 5: train the TF-IDF + logistic regression question classifier and score it.

    python scripts/train_classifier.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiment "classifier"

Trains on datasets/classifier/train.jsonl (run generate_classifier_data.py first),
tests on the hand-written questions, logs the run to MLflow and saves the model to
data/models/tfidf_lr.joblib.
"""

import hashlib
import json
import tempfile
import time
from pathlib import Path

import mlflow

from sklearn_assistant.classifier import tfidf
from sklearn_assistant.classifier.data import load_test, load_train
from sklearn_assistant.config import (
    CLASSIFIER_TRAIN_PATH, EVAL_QUESTIONS_PATH, MLFLOW_TRACKING_URI, MODELS_DIR, OFF_TOPIC_TEST_PATH,
)
from sklearn_assistant.evaluation.classifier import evaluate, format_confusion
from sklearn_assistant.tracking import git_commit

EXPERIMENT = "classifier"
MODEL_PATH = MODELS_DIR / "tfidf_lr.joblib"


def file_sha(*paths: Path) -> str:
    """Fingerprint of the data files, so runs on different data are told apart in MLflow."""
    return hashlib.sha1(b"".join(p.read_bytes() for p in paths)).hexdigest()[:12]


def main() -> None:
    train, test = load_train(), load_test()

    start = time.perf_counter()
    search = tfidf.train([q.question for q in train], [q.category for q in train])
    train_seconds = time.perf_counter() - start
    model = search.best_estimator_

    # predict() takes a list and returns an array; wrap it as question -> category.
    summary, rows, matrix = evaluate(lambda q: model.predict([q])[0], test)
    tfidf.save(model, MODEL_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="tfidf_lr"):
        mlflow.log_params({
            "model": "tfidf_lr",
            # Which grid candidate won, e.g. char features on/off and C.
            "char_ngrams": search.best_params_["features__char"] != "drop",
            "C": search.best_params_["clf__C"],
            "n_train": len(train), "n_test": len(test),
            "train_set_sha": file_sha(CLASSIFIER_TRAIN_PATH),
            "test_set_sha": file_sha(EVAL_QUESTIONS_PATH, OFF_TOPIC_TEST_PATH),
            "git_commit": git_commit(),
        })
        mlflow.log_metrics({**summary, "cv_macro_f1": search.best_score_,
                            "train_seconds": train_seconds,
                            # LLM-based classifiers will log their real cost here.
                            "cost_per_1k_usd": 0.0})
        with tempfile.TemporaryDirectory() as tmp:
            errors = Path(tmp) / "misclassified.json"
            errors.write_text(json.dumps([r for r in rows if not r["correct"]], indent=1))
            confusion = Path(tmp) / "confusion_matrix.txt"
            confusion.write_text(format_confusion(matrix))
            for path in (errors, confusion, MODEL_PATH):
                mlflow.log_artifact(str(path))

    best = search.best_params_
    print(f"Trained on {len(train)} questions in {train_seconds:.1f} s "
          f"(char n-grams: {best['features__char'] != 'drop'}, C={best['clf__C']}, "
          f"CV macro-F1 {search.best_score_:.3f})\n")
    print(f"Test: {len(test)} hand-written questions")
    print(f"  accuracy  {summary['accuracy']:.3f}\n  macro-F1  {summary['macro_f1']:.3f}\n"
          f"  latency   {summary['latency_ms']:.2f} ms/question\n")
    print("F1 by category")
    for key, value in summary.items():
        if key.startswith("f1/"):
            print(f"  {key[3:]:16} {value:.3f}")
    print("\n" + format_confusion(matrix))
    print("\nMisclassified:")
    for r in rows:
        if not r["correct"]:
            print(f"  [{r['category']} -> {r['predicted']}] {r['question']}")


if __name__ == "__main__":
    main()
