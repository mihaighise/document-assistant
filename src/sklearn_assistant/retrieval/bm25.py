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

# \w matches letters, digits and underscore, so "model_selection.GridSearchCV" yields the
# words "model_selection" and "GridSearchCV": dots, parentheses and spaces all act as
# separators, while snake_case names stay in one piece for the whole-identifier token.
WORD_RE = re.compile(r"\w+")
# "GridSearchCV" -> Grid, Search, CV; "HistGradientBoosting2D" -> Hist, Gradient, Boosting, 2, D
# Alternatives are tried left to right at each position:
#   [A-Z]+(?=[A-Z][a-z])  a run of capitals followed by a new Capitalized word ("CV" in "CVSplit")
#   [A-Z]?[a-z]+          a normal word, optionally capitalized ("Grid", "search")
#   [A-Z]+                a trailing acronym ("CV" at the end of "GridSearchCV")
#   \d+                   numbers ("2" in "2D")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
# Only words common in *questions* ("how do I ...") are dropped. BM25 already gives
# frequent words a low weight through IDF, but these appear in nearly every question
# and in few chunks, so they would otherwise add noise to every query.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or that the "
    "this to was what when which with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, plus the snake_case/CamelCase parts of each identifier."""
    tokens = []
    for word in WORD_RE.findall(text):
        # The whole word, lowercased: lets an exact query like "fit_transform" or
        # "gridsearchcv" match only chunks that contain that precise identifier.
        whole = word.lower()
        if whole not in STOPWORDS:
            tokens.append(whole)
        # Split on underscores first, then split each piece on case changes, so
        # "n_jobs" -> ["n", "jobs"] and "GridSearchCV" -> ["grid", "search", "cv"].
        # CAMEL_RE runs on the original casing, since lowercasing would erase the
        # boundaries it looks for.
        parts = [p.lower() for piece in word.split("_") for p in CAMEL_RE.findall(piece)]
        # A plain word ("forest") splits into just itself; only add parts for real
        # compound identifiers, otherwise every plain word would be counted twice.
        if len(parts) > 1:
            # Single letters (the "n" in n_jobs) carry no signal on their own.
            tokens += [p for p in parts if len(p) > 1 and p not in STOPWORDS]
    return tokens


class BM25Index:
    """BM25 over a list of chunk dicts (as read from chunks.jsonl)."""

    def __init__(self, chunks: list[dict]):
        # Positions line up: self.ids[i] is the chunk that BM25 calls document i.
        self.ids = [c["id"] for c in chunks]
        # BM25Okapi precomputes, for every token, how many chunks contain it (for IDF)
        # and each chunk's length (longer chunks get their term counts scaled down).
        # Default parameters: k1=1.5 (how quickly repeated terms stop adding score),
        # b=0.75 (how strongly to normalize by chunk length).
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, BM25 score) pairs, best first. Zero-score chunks are skipped."""
        # The query goes through the same tokenizer as the chunks; otherwise "GridSearchCV"
        # in a query wouldn't produce the "grid"/"search" tokens the index contains.
        # get_scores returns one score per chunk (a numpy array of ~8k floats).
        scores = self.bm25.get_scores(tokenize(query))
        # argsort is ascending, so reverse it for best-first, then keep the first k.
        # Sorting all ~8k scores takes well under a millisecond, so there's no need
        # for a partial sort.
        top = np.argsort(scores)[::-1][:k]
        # A zero score means the chunk shares no query token at all; returning it would
        # just give RRF arbitrary filler to rank.
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
