"""Baseline question classifier: TF-IDF features + logistic regression.

How it works:
1. TF-IDF turns a question into a long, mostly-zero vector with one entry per word (or
   word pair, or character n-gram) seen in training. An entry is high when that term
   appears in this question but is rare across all questions (IDF), so "the" counts for
   little and "difference" or "error" count for a lot.
2. Logistic regression learns one weight per term per category. To classify, it sums
   the weights of the terms present and picks the category with the highest total
   (softmax turns the totals into probabilities).
So it learns cues like "vs", "difference between" -> comparison, "error", "warning" ->
troubleshooting, "how do I" -> howto. It sees words, not meaning: a question phrased with
words it never saw in training gets little signal.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline


def char_ngrams() -> TfidfVectorizer:
    """Character 2-5-grams within word boundaries ("scale" -> " s", "sc", "sca", ...)."""
    # They catch typos ("clasifier" still shares most n-grams with "classifier") and
    # pieces of identifiers ("fit_transform" shares "transform").
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


# Settings tried by cross-validation; the best combination is refit on all training data.
PARAM_GRID = {
    # With char n-grams vs. without ("drop" removes that part of the FeatureUnion).
    "features__char": [char_ngrams(), "drop"],
    # Inverse regularization strength: small C = smaller weights, a smoother model that
    # relies on many weak cues; large C trusts individual terms more (risks overfitting).
    "clf__C": [0.3, 1.0, 3.0, 10.0, 30.0],
}


def build_pipeline() -> Pipeline:
    """Untrained text -> category pipeline."""
    features = FeatureUnion([
        # Words and word pairs. Default English stopwords are NOT removed: words like
        # "how", "why", "what", "vs" and "not" are exactly what separates the categories.
        # sublinear_tf uses 1 + log(count), so a repeated word doesn't dominate.
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("char", char_ngrams()),
    ])
    # A Pipeline chains the steps so fit/predict run them in order, and so cross-
    # validation refits the vectorizers on each training fold (no peeking at the
    # held-out fold's vocabulary).
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def train(questions: list[str], labels: list[str], seed: int = 0) -> GridSearchCV:
    """Pick settings by 5-fold cross-validation on the training set, then refit on all of it.

    The returned object predicts with the best pipeline; `.best_params_` and
    `.best_score_` (mean CV macro-F1) say what was chosen and how well it did.
    """
    # Stratified: every fold keeps the same category mix as the whole training set.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Macro-F1 (F1 per category, averaged) so every category counts equally.
    # GridSearchCV tries every combination: 2 x 5 = 10 candidates, each fitted 5 times
    # (once per fold), in parallel (n_jobs=-1).
    search = GridSearchCV(build_pipeline(), PARAM_GRID, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(questions, labels)
    return search


def save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save only the refit best pipeline, not the whole search object.
    joblib.dump(getattr(model, "best_estimator_", model), path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
