"""Step 2: embed the chunks into Chroma, then smoke-test hybrid retrieval on a few questions."""

import time

from sklearn_assistant.config import CHROMA_DIR
from sklearn_assistant.retrieval.dense import build_index
from sklearn_assistant.retrieval.hybrid import HybridRetriever, load_chunks

SAMPLE_QUERIES = [
    "how do I handle class imbalance in RandomForest?",
    "what's the difference between fit_transform and transform?",
    "n_jobs parameter",
]


def main() -> None:
    """Build the vector index and print the top results for each sample query."""
    chunks = load_chunks()
    start = time.perf_counter()
    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR} in {time.perf_counter() - start:.0f}s")

    retriever = HybridRetriever(chunks)
    for query in SAMPLE_QUERIES:
        print(f"\n{query}")
        for r in retriever.retrieve(query, k=3):
            # Ranks show which retriever(s) found it: "d3 b-" = dense rank 3, not in BM25 top 20.
            ranks = f"d{r.dense_rank or '-'} b{r.bm25_rank or '-'}"
            print(f"  {r.score:.4f}  [{ranks:>7}]  {r.chunk['title'][:70]}")
            print(f"           {r.chunk['url']}")


if __name__ == "__main__":
    main()
