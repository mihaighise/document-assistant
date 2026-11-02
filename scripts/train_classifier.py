"""Step 5: train the TF-IDF + logistic regression question classifier and score it.

    python scripts/train_classifier.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiment "classifier"

Trains on datasets/classifier/train.jsonl (run generate_classifier_data.py first),
tests on the hand-written questions, logs the run to MLflow and saves the model to
data/models/tfidf_lr.joblib.
"""

import time

from sklearn_assistant.classifier import tfidf
from sklearn_assistant.classifier.data import load_test, load_train
from sklearn_assistant.config import CLASSIFIER_TRAIN_PATH, MODELS_DIR
from sklearn_assistant.evaluation.classifier import evaluate, print_report
from sklearn_assistant.tracking import file_sha, log_classifier_run

MODEL_PATH = MODELS_DIR / "tfidf_lr.joblib"


def main() -> None:
    train, test = load_train(), load_test()

    start = time.perf_counter()
    search = tfidf.train([q.question for q in train], [q.category for q in train])
    train_seconds = time.perf_counter() - start
    model = search.best_estimator_
    best = search.best_params_
    char_ngrams = best["features__char"] != "drop"

    # predict() takes a list and returns an array; wrap it as question -> category.
    summary, rows, matrix = evaluate(lambda q: model.predict([q])[0], test)
    tfidf.save(model, MODEL_PATH)

    log_classifier_run(
        "tfidf_lr",
        # Which grid candidate won (char features on/off, C), and what it was trained on.
        params={"model": "tfidf_lr", "char_ngrams": char_ngrams, "C": best["clf__C"],
                "n_train": len(train), "n_test": len(test),
                "train_set_sha": file_sha(CLASSIFIER_TRAIN_PATH)},
        metrics={**summary, "cv_macro_f1": search.best_score_, "train_seconds": train_seconds,
                 "cost_per_1k_usd": 0.0},
        rows=rows, matrix=matrix, files=(MODEL_PATH,),
    )

    print(f"Trained on {len(train)} questions in {train_seconds:.1f} s "
          f"(char n-grams: {char_ngrams}, C={best['clf__C']}, "
          f"CV macro-F1 {search.best_score_:.3f})\n")
    print_report(summary, rows, matrix)


if __name__ == "__main__":
    main()
