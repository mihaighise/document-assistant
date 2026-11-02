"""Helpers shared by the scripts that log runs to MLflow."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow

from sklearn_assistant.config import EVAL_QUESTIONS_PATH, MLFLOW_TRACKING_URI, OFF_TOPIC_TEST_PATH, ROOT
from sklearn_assistant.evaluation.classifier import format_confusion


def git_commit() -> str:
    """Current commit hash, marked dirty if there are uncommitted changes."""
    # Logged with every run so a result can be traced back to the exact code. "-dirty"
    # warns that the working tree had uncommitted edits, so checking out that commit
    # alone won't reproduce the run.
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    # `git status --porcelain` prints one line per changed or untracked file, and nothing
    # when the tree is clean.
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def file_sha(*paths: Path) -> str:
    """Fingerprint of data files, so runs on different data are told apart in MLflow."""
    return hashlib.sha1(b"".join(p.read_bytes() for p in paths)).hexdigest()[:12]


def log_classifier_run(run_name: str, params: dict, metrics: dict, rows: list[dict],
                       matrix: list[list[int]], files: tuple[Path, ...] = (),
                       history: list[dict] = ()) -> None:
    """Log one classifier evaluation to the MLflow experiment "classifier".

    Every classifier (TF-IDF, BERT, LLMs) logs the same metrics and artifacts, so they
    line up as columns in the MLflow UI's run comparison. `history` holds per-epoch
    numbers for trained models, logged as steps so MLflow can chart them.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("classifier")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, "test_set_sha": file_sha(EVAL_QUESTIONS_PATH,
                                                              OFF_TOPIC_TEST_PATH),
                           "git_commit": git_commit()})
        mlflow.log_metrics(metrics)
        for row in history:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
        with tempfile.TemporaryDirectory() as tmp:
            errors = Path(tmp) / "misclassified.json"
            errors.write_text(json.dumps([r for r in rows if not r["correct"]], indent=1))
            confusion = Path(tmp) / "confusion_matrix.txt"
            confusion.write_text(format_confusion(matrix))
            for path in (errors, confusion, *files):
                mlflow.log_artifact(str(path))
