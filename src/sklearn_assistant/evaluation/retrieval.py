"""Retrieval evaluation against the hand-labeled question set.

Each question lists the doc sections a good answer would cite, as URLs relative to
the docs root (`modules/svm.html#complexity`). Labels are URLs rather than chunk ids
so they survive chunker changes: a section split into several parts still has one
URL, and any of its parts counts as a hit.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn_assistant.config import DOCS_BASE_URL, EVAL_QUESTIONS_PATH
from sklearn_assistant.retrieval.hybrid import HybridRetriever, Mode

K = 5            # headline cutoff: recall@5
DEPTH = 10       # results fetched per question; MRR and recall@10 look this deep


@dataclass
class Question:
    """One labeled evaluation question."""

    id: str
    question: str
    category: str         # howto | concept | comparison | api | troubleshooting
    relevant: list[str]   # section URLs relative to the docs root


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[Question]:
    """Read the evaluation set."""
    with open(path, encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f]


def relative_url(url: str) -> str:
    """Strip the version-pinned base so chunk URLs compare equal to labels."""
    return url.removeprefix(DOCS_BASE_URL)


def missing_labels(questions: list[Question], chunks: list[dict]) -> list[str]:
    """Labeled URLs that no chunk has, e.g. after the chunker dropped or renamed a section."""
    indexed = {relative_url(c["url"]) for c in chunks}
    return [f"{q.id}: {url}" for q in questions for url in q.relevant if url not in indexed]


def score(retrieved: list[str], relevant: list[str]) -> dict[str, float]:
    """Metrics for one question, given retrieved URLs in rank order."""
    gold = set(relevant)
    first_hit = next((rank for rank, url in enumerate(retrieved, start=1) if url in gold), None)
    return {
        "recall_at_5": len(gold & set(retrieved[:K])) / len(gold),
        "recall_at_10": len(gold & set(retrieved[:DEPTH])) / len(gold),
        "hit_at_5": float(first_hit is not None and first_hit <= K),
        "mrr": 1.0 / first_hit if first_hit and first_hit <= DEPTH else 0.0,
    }


def evaluate(retriever: HybridRetriever, questions: list[Question],
             mode: Mode) -> tuple[dict[str, float], list[dict]]:
    """Run every question through one retriever mode.

    Returns overall metrics (plus `<metric>/<category>` breakdowns and mean latency)
    and one row per question for inspecting failures.
    """
    rows = []
    for q in questions:
        start = time.perf_counter()
        results = retriever.retrieve(q.question, k=DEPTH, mode=mode)
        latency_ms = (time.perf_counter() - start) * 1000
        retrieved = [relative_url(r.chunk["url"]) for r in results]
        rows.append({"id": q.id, "category": q.category, "question": q.question,
                     **score(retrieved, q.relevant), "latency_ms": latency_ms,
                     "relevant": q.relevant, "retrieved": retrieved})

    metric_names = ["recall_at_5", "recall_at_10", "hit_at_5", "mrr"]
    summary = {m: mean(r[m] for r in rows) for m in metric_names}
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows)
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    for category, cat_rows in sorted(by_category.items()):
        for m in metric_names:
            summary[f"{m}/{category}"] = mean(r[m] for r in cat_rows)
    return summary, rows


def mean(values) -> float:
    """Arithmetic mean of an iterable."""
    values = list(values)
    return sum(values) / len(values)
