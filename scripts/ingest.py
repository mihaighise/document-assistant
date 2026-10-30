"""Step 1: download the docs and write section-level chunks to data/processed/chunks.jsonl."""

import json
from collections import Counter

from sklearn_assistant.config import CHUNKS_PATH, DOCS_DIR
from sklearn_assistant.ingest.chunker import chunk_docs
from sklearn_assistant.ingest.download import download_docs


def main() -> None:
    """Download (if needed), chunk, write JSONL, and print summary stats."""
    download_docs()
    chunks = chunk_docs(DOCS_DIR)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # Quick sanity check on chunk sizes; the max should stay near MAX_CHARS.
    lengths = sorted(len(c.text) for c in chunks)
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")
    print("By type:", dict(Counter(c.doc_type for c in chunks)))
    print(f"Chars: median={lengths[len(lengths) // 2]}, p95={lengths[int(len(lengths) * 0.95)]}, "
          f"max={lengths[-1]}")


if __name__ == "__main__":
    main()
