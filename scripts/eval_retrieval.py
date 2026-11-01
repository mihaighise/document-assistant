"""Step 3: score each retriever mode on the eval set and log one MLflow run per mode.

    python scripts/eval_retrieval.py                 # dense, bm25 and hybrid
    python scripts/eval_retrieval.py --modes hybrid  # just one
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow

from sklearn_assistant.config import EMBED_MODEL, EVAL_QUESTIONS_PATH, MLFLOW_TRACKING_URI, ROOT
from sklearn_assistant.evaluation.retrieval import DEPTH, evaluate, load_questions, missing_labels
from sklearn_assistant.ingest.chunker import MAX_CHARS
from sklearn_assistant.retrieval import hybrid
from sklearn_assistant.retrieval.hybrid import HybridRetriever, load_chunks

EXPERIMENT = "retrieval"


def git_commit() -> str:
    """Current commit hash, marked dirty if there are uncommitted changes."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def main() -> None:
    """Evaluate the requested modes, log them to MLflow, and print a comparison table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"],
                        choices=["dense", "bm25", "hybrid"])
    args = parser.parse_args()

    questions = load_questions()
    chunks = load_chunks()
    if missing := missing_labels(questions, chunks):
        raise SystemExit("Labeled sections missing from the index:\n  " + "\n  ".join(missing))

    retriever = HybridRetriever(chunks)
    retriever.retrieve("warm up", k=1)  # load the model before timing anything
    eval_set_sha = hashlib.sha1(EVAL_QUESTIONS_PATH.read_bytes()).hexdigest()[:12]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    summaries = {}
    for mode in args.modes:
        summary, rows = evaluate(retriever, questions, mode)
        summaries[mode] = summary
        with mlflow.start_run(run_name=mode):
            mlflow.log_params({
                "mode": mode, "embed_model": EMBED_MODEL, "chunk_max_chars": MAX_CHARS,
                "n_chunks": len(chunks), "rrf_k": hybrid.RRF_K, "candidates": hybrid.CANDIDATES,
                "depth": DEPTH, "n_questions": len(questions), "eval_set_sha": eval_set_sha,
                "git_commit": git_commit(),
            })
            # MLflow metric names can't contain "@"; "recall_at_5/howto" is the per-category form.
            mlflow.log_metrics(summary)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "per_question.json"
                path.write_text(json.dumps(rows, indent=1))
                mlflow.log_artifact(str(path))

    print(f"\n{len(questions)} questions\n")
    print(f"{'mode':8} {'recall@5':>9} {'recall@10':>10} {'hit@5':>7} {'MRR':>7} {'ms/query':>9}")
    for mode, s in summaries.items():
        print(f"{mode:8} {s['recall_at_5']:9.3f} {s['recall_at_10']:10.3f} {s['hit_at_5']:7.3f} "
              f"{s['mrr']:7.3f} {s['latency_ms']:9.1f}")
    categories = sorted({q.category for q in questions})
    print("\nrecall@5 by category")
    print(f"{'mode':8} " + " ".join(f"{c:>15}" for c in categories))
    for mode, s in summaries.items():
        print(f"{mode:8} " + " ".join(f"{s[f'recall_at_5/{c}']:15.3f}" for c in categories))


if __name__ == "__main__":
    main()
