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

# Restricts `mode` to these three strings for type checkers and editors.
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
    # JSONL: one JSON object per line, so the file can be read line by line.
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids into one list of (id, score), best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            # Each list a chunk appears in adds 1/(k + rank). With k=60, rank 1 adds
            # 1/61 ≈ 0.0164 and rank 20 adds 1/80 = 0.0125: being in a list at all
            # matters more than the exact position. So a chunk ranked mid-list by
            # *both* retrievers (about 0.03) beats one that only a single retriever
            # placed first (0.0164).
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Dense + BM25 retrieval over the same chunks.

    `mode` lets the evaluation in step 3 measure each retriever on its own
    against the fused result.
    """

    def __init__(self, chunks: list[dict] | None = None):
        # Callers that already loaded the chunks (the scripts) can pass them in to avoid
        # reading the ~10 MB file twice.
        chunks = chunks if chunks is not None else load_chunks()
        # Both retrievers return only ids; this dict maps them back to full chunks.
        self.chunks = {c["id"]: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex()
        # Catches re-running ingest without rebuilding the vector index.
        # Comparing counts is a cheap proxy: changing the chunker almost always changes
        # how many chunks there are. Without this check, a stale index would return ids
        # missing from self.chunks and fail later with a confusing KeyError.
        if self.dense.count() != len(chunks):
            raise RuntimeError(f"Chroma has {self.dense.count()} chunks but {CHUNKS_PATH} has "
                               f"{len(chunks)}; run scripts/build_index.py")

    def retrieve(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Result]:
        """Top-k chunks for `query`."""
        # Single-retriever modes return that retriever's top k directly; there's nothing
        # to fuse.
        if mode == "dense":
            return [Result(self.chunks[cid], s, dense_rank=i)
                    for i, (cid, s) in enumerate(self.dense.search(query, k), start=1)]
        if mode == "bm25":
            return [Result(self.chunks[cid], s, bm25_rank=i)
                    for i, (cid, s) in enumerate(self.bm25.search(query, k), start=1)]

        # Hybrid: take a deeper candidate pool (20) from each retriever than the k we
        # return, so a chunk ranked, say, 12th by both can still rise to the top.
        dense_ids = [cid for cid, _ in self.dense.search(query, CANDIDATES)]
        bm25_ids = [cid for cid, _ in self.bm25.search(query, CANDIDATES)]
        # Remember each chunk's position in each list. This is only for inspection
        # (build_index.py and the eval artifacts show "d3 b-"); RRF computes its own ranks.
        dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
        bm25_rank = {cid: i for i, cid in enumerate(bm25_ids, start=1)}
        # .get() gives None for chunks that only one retriever found.
        return [Result(self.chunks[cid], s, dense_rank.get(cid), bm25_rank.get(cid))
                for cid, s in rrf([dense_ids, bm25_ids])[:k]]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Shared retriever, built on first use (loads the model and BM25 index once)."""
    # Building one costs a few seconds (reading chunks, tokenizing for BM25, opening
    # Chroma), so the module-level retrieve() below reuses a single instance.
    return HybridRetriever()


def retrieve(query: str, k: int = 5, mode: Mode = "hybrid") -> list[Result]:
    """Top-k chunks for `query` using the shared retriever."""
    return get_retriever().retrieve(query, k, mode)
