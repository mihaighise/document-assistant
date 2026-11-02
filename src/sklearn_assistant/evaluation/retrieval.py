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
    # Each JSONL line has exactly the Question fields, so it unpacks straight into the
    # dataclass. A typo in a field name fails here with a TypeError, not mid-run.
    with open(path, encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f]


def relative_url(url: str) -> str:
    """Strip the version-pinned base so chunk URLs compare equal to labels."""
    # Labels omit "https://scikit-learn.org/1.9/" so the eval set doesn't need editing
    # when the docs version changes.
    return url.removeprefix(DOCS_BASE_URL)


def missing_labels(questions: list[Question], chunks: list[dict]) -> list[str]:
    """Labeled URLs that no chunk has, e.g. after the chunker dropped or renamed a section."""
    # A gold URL that isn't in the index can never be retrieved, so that question would
    # score 0 forever and quietly drag down every run. Better to refuse to run.
    indexed = {relative_url(c["url"]) for c in chunks}
    return [f"{q.id}: {url}" for q in questions for url in q.relevant if url not in indexed]


def score(retrieved: list[str], relevant: list[str]) -> dict[str, float]:
    """Metrics for one question, given retrieved URLs in rank order."""
    gold = set(relevant)
    # Rank (1-based) of the first retrieved URL that is gold, or None if there is none.
    # next() with a default stops at the first match instead of scanning the whole list.
    first_hit = next((rank for rank, url in enumerate(retrieved, start=1) if url in gold), None)
    return {
        # Set intersection: two parts of the same gold section in the top 5 count once,
        # so recall can't exceed 1.0 and duplicates don't inflate the score.
        "recall_at_5": len(gold & set(retrieved[:K])) / len(gold),
        "recall_at_10": len(gold & set(retrieved[:DEPTH])) / len(gold),
        # 1.0 if anything useful made the top 5, else 0.0; averaged over questions it
        # becomes "share of questions where the LLM would see at least one right section".
        "hit_at_5": float(first_hit is not None and first_hit <= K),
        # Reciprocal rank: 1st -> 1.0, 2nd -> 0.5, 3rd -> 0.33 ..., 0 if nothing in the
        # top DEPTH. "MRR" is the mean of this over questions.
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
        # Time only the retrieve() call, so the latency covers embedding the query plus
        # both searches, and not the scoring below.
        start = time.perf_counter()
        results = retriever.retrieve(q.question, k=DEPTH, mode=mode)
        latency_ms = (time.perf_counter() - start) * 1000
        # Turn results into labels' format (relative URLs) so they can be compared to gold.
        retrieved = [relative_url(r.chunk["url"]) for r in results]
        # Each row keeps the retrieved list next to the gold one, so a failure can be
        # diagnosed from the per_question.json artifact alone without re-running anything.
        rows.append({"id": q.id, "category": q.category, "question": q.question,
                     **score(retrieved, q.relevant), "latency_ms": latency_ms,
                     "relevant": q.relevant, "retrieved": retrieved})

    metric_names = ["recall_at_5", "recall_at_10", "hit_at_5", "mrr"]
    # Macro-average: every question counts equally, however many gold sections it has.
    summary = {m: mean(r[m] for r in rows) for m in metric_names}
    summary["latency_ms"] = mean(r["latency_ms"] for r in rows)
    # Same metrics restricted to each category, logged as e.g. "recall_at_5/howto".
    # MLflow allows "/" in metric names, and the UI groups metrics that share a prefix.
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    for category, cat_rows in sorted(by_category.items()):
        for m in metric_names:
            summary[f"{m}/{category}"] = mean(r[m] for r in cat_rows)
    return summary, rows


def mean(values) -> float:
    """Arithmetic mean of an iterable."""
    # Materialize first: callers pass generators, which can't be measured with len().
    values = list(values)
    return sum(values) / len(values)
