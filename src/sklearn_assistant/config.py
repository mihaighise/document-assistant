"""Project-wide paths and constants."""

from pathlib import Path

# Repository root (this file lives in src/sklearn_assistant/).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = RAW_DIR / "docs"            # unzipped HTML docs
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"  # output of scripts/ingest.py, one chunk per line

# Offline build of the docs published by scikit-learn (~100 MB zipped).
DOCS_ZIP_URL = "https://scikit-learn.org/stable/_downloads/scikit-learn-docs.zip"
# Citations point at the version-pinned site so links keep matching the indexed text.
DOCS_VERSION = "1.9"
DOCS_BASE_URL = f"https://scikit-learn.org/{DOCS_VERSION}/"

# Step 2: retrieval indexes.
CHROMA_DIR = DATA_DIR / "chroma"       # persistent vector store
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 512-token input limit; chunker.MAX_CHARS is sized for it

# Step 3: evaluation.
EVAL_QUESTIONS_PATH = ROOT / "eval" / "retrieval_questions.jsonl"  # hand-labeled, committed
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
