"""Download and unpack the official offline HTML build of the scikit-learn docs."""

import zipfile

import requests

from sklearn_assistant.config import DOCS_DIR, DOCS_ZIP_URL, RAW_DIR


def download_docs(force: bool = False) -> None:
    """Fetch the docs zip and extract it to DOCS_DIR.

    Skipped when the docs are already extracted, unless `force` is set.
    """
    if DOCS_DIR.exists() and not force:
        print(f"Docs already present at {DOCS_DIR}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "scikit-learn-docs.zip"
    print(f"Downloading {DOCS_ZIP_URL} ...")
    # Stream to disk in 1 MB blocks instead of holding the whole zip in memory.
    with requests.get(DOCS_ZIP_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for block in resp.iter_content(chunk_size=1 << 20):
                f.write(block)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DOCS_DIR)
    print(f"Extracted to {DOCS_DIR}")
