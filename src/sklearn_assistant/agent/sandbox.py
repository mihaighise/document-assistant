"""Run the Python code from an answer to check that it actually executes.

This is process isolation, not a security boundary: the code runs as your user, with
your filesystem and network access. It's limited to a throwaway working directory, a
stripped environment and a timeout, which is enough to catch broken examples. Step 8
moves it into a Docker container. Off by default; enable it per run.
"""

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

TIMEOUT_S = 60       # long enough for small model fits on toy datasets
MAX_OUTPUT = 2000    # characters of stderr/stdout kept for the fix prompt

# A fenced block opened with ```python (or ```py) up to the next closing fence.
CODE_BLOCK_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)


@dataclass
class RunResult:
    """Outcome of running the code."""

    ok: bool
    output: str  # stdout on success, the tail of stderr (the traceback) on failure


def extract_code(markdown: str) -> str:
    """All ```python blocks of an answer, joined in order.

    Joined rather than run one by one, because later blocks usually reuse variables
    from earlier ones (load data, then fit, then evaluate).
    """
    # Doctest-style lines (">>> x") can't run as a script; skip answers made of them.
    blocks = [b for b in CODE_BLOCK_RE.findall(markdown) if ">>>" not in b]
    return "\n\n".join(blocks)


def run_python(code: str, timeout: float = TIMEOUT_S) -> RunResult:
    """Execute `code` in a fresh interpreter and report whether it succeeded."""
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "snippet.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        # Minimal environment: no API keys or other secrets from the parent process.
        # HOME points at the temp dir so any caches or files the code writes vanish with it;
        # the Agg backend makes matplotlib render off-screen instead of opening windows.
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "MPLBACKEND": "Agg"}
        try:
            # sys.executable is the venv's Python, so scikit-learn is importable.
            # -I (isolated mode) ignores PYTHON* env vars and the user site-packages dir.
            proc = subprocess.run([sys.executable, "-I", script], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return RunResult(False, f"Timed out after {timeout:.0f}s")
    if proc.returncode == 0:
        return RunResult(True, proc.stdout[-MAX_OUTPUT:])
    # The end of stderr holds the exception type and message, the useful part for a fix.
    return RunResult(False, proc.stderr[-MAX_OUTPUT:])
