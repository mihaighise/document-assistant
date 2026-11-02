"""Helpers shared by the scripts that log runs to MLflow."""

import subprocess

from sklearn_assistant.config import ROOT


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
