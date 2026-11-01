"""Lexical retrieval: BM25 over chunks, with a tokenizer that understands code identifiers.

Questions about a library are full of exact names (`fit_transform`, `GridSearchCV`,
`n_jobs`). Each identifier is indexed whole and also split into its parts, so a query
for `fit_transform` matches it exactly while "grid search" still matches `GridSearchCV`.

The index is rebuilt in memory on load; for ~8k chunks that takes a couple of seconds,
so it isn't worth persisting.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

WORD_RE = re.compile(r"\w+")
# "GridSearchCV" -> Grid, Search, CV; "HistGradientBoosting2D" -> Hist, Gradient, Boosting, 2, D
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or that the "
    "this to was what when which with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, plus the snake_case/CamelCase parts of each identifier."""
    tokens = []
    for word in WORD_RE.findall(text):
        whole = word.lower()
        if whole not in STOPWORDS:
            tokens.append(whole)
        parts = [p.lower() for piece in word.split("_") for p in CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            # Single letters (the "n" in n_jobs) carry no signal on their own.
            tokens += [p for p in parts if len(p) > 1 and p not in STOPWORDS]
    return tokens


class BM25Index:
    """BM25 over a list of chunk dicts (as read from chunks.jsonl)."""

    def __init__(self, chunks: list[dict]):
        self.ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, BM25 score) pairs, best first. Zero-score chunks are skipped."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
