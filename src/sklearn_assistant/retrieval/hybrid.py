"""Hybrid retrieval: merge dense and BM25 rankings with Reciprocal Rank Fusion.

RRF scores each chunk by sum(1 / (RRF_K + rank)) over the rankings it appears in.
It only looks at ranks, so cosine similarities and BM25 scores never need to be
put on a common scale.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from sklearn_assistant.config import CHUNKS_PATH
from sklearn_assistant.retrieval.bm25 import BM25Index
from sklearn_assistant.retrieval.dense import DenseIndex

RRF_K = 60        # standard constant from the RRF paper; damps the weight of top ranks
CANDIDATES = 20   # results taken from each retriever before fusion

Mode = Literal["hybrid", "dense", "bm25"]


@dataclass
class Result:
    """One retrieved chunk, with where it ranked in each retriever (1-based, None if absent)."""

    chunk: dict          # the full chunk: id, text, title, url, page, doc_type
    score: float         # RRF score in hybrid mode, the retriever's own score otherwise
    dense_rank: int | None = None
    bm25_rank: int | None = None


def load_chunks() -> list[dict]:
    """Read the chunks written by scripts/ingest.py."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids into one list of (id, score), best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Dense + BM25 retrieval over the same chunks.

    `mode` lets the evaluation in step 3 measure each retriever on its own
    against the fused result.
    """

    def __init__(self, chunks: list[dict] | None = None):
        chunks = chunks if chunks is not None else load_chunks()
        self.chunks = {c["id"]: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex()
        # Catches re-running ingest without rebuilding the vector index.
        if self.dense.count() != len(chunks):
            raise RuntimeError(f"Chroma has {self.dense.count()} chunks but {CHUNKS_PATH} has "
                               f"{len(chunks)}; run scripts/build_index.py")

    def retrieve(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Result]:
        """Top-k chunks for `query`."""
        if mode == "dense":
            return [Result(self.chunks[cid], s, dense_rank=i)
                    for i, (cid, s) in enumerate(self.dense.search(query, k), start=1)]
        if mode == "bm25":
            return [Result(self.chunks[cid], s, bm25_rank=i)
                    for i, (cid, s) in enumerate(self.bm25.search(query, k), start=1)]

        dense_ids = [cid for cid, _ in self.dense.search(query, CANDIDATES)]
        bm25_ids = [cid for cid, _ in self.bm25.search(query, CANDIDATES)]
        dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
        bm25_rank = {cid: i for i, cid in enumerate(bm25_ids, start=1)}
        return [Result(self.chunks[cid], s, dense_rank.get(cid), bm25_rank.get(cid))
                for cid, s in rrf([dense_ids, bm25_ids])[:k]]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Shared retriever, built on first use (loads the model and BM25 index once)."""
    return HybridRetriever()


def retrieve(query: str, k: int = 5, mode: Mode = "hybrid") -> list[Result]:
    """Top-k chunks for `query` using the shared retriever."""
    return get_retriever().retrieve(query, k, mode)
