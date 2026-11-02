"""Dense retrieval: chunk embeddings stored in a persistent Chroma collection."""

from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from sklearn_assistant.config import CHROMA_DIR, EMBED_MODEL

COLLECTION = "chunks"
# BGE v1.5 recommends this prefix on short queries (not on passages) for retrieval.
# The model was trained so that "prefix + question" lands near the passages that
# answer it, rather than near passages that merely *look like* the question.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# lru_cache with no arguments to vary means the body runs once and every later call
# returns the same object, so the ~130 MB model is loaded at most once per process.
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """The embedding model, loaded once per process (downloads on first use)."""
    # First use downloads the model to ~/.cache/huggingface; after that it loads from
    # disk. sentence-transformers picks the device itself (Apple MPS, CUDA, or CPU).
    return SentenceTransformer(EMBED_MODEL)


def get_client() -> chromadb.ClientAPI:
    """Client for the on-disk Chroma store."""
    # PersistentClient keeps the database in CHROMA_DIR (SQLite plus HNSW index files),
    # so the index survives between processes. Telemetry is off so Chroma doesn't send
    # usage pings.
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def build_index(chunks: list[dict], batch_size: int = 64) -> None:
    """Embed every chunk and (re)create the Chroma collection from scratch.

    Rebuilding rather than upserting means chunks dropped by a re-ingest don't
    linger in the index.
    """
    client = get_client()
    # Start from an empty collection. NotFoundError just means this is the first build.
    try:
        client.delete_collection(COLLECTION)
    except chromadb.errors.NotFoundError:
        pass
    # HNSW is the approximate nearest-neighbor graph Chroma searches. Its distance metric
    # is fixed when the collection is created; Chroma's default is squared L2, so cosine
    # has to be requested here explicitly.
    collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})

    # Normalized vectors make cosine distance equal to 1 - dot product.
    # Passages are embedded as-is: QUERY_PREFIX is only for the query side.
    # batch_size controls how many texts go through the model at once (memory vs. speed);
    # the result is one (n_chunks, 384) float array in the same order as `chunks`.
    embeddings = get_model().encode([c["text"] for c in chunks], batch_size=batch_size,
                                    normalize_embeddings=True, show_progress_bar=True)
    # Chroma caps how many records a single add() may contain (5461 here), so the
    # ~8k chunks go in two slices. Embeddings are sliced with the same offsets to stay aligned.
    step = client.get_max_batch_size()
    for i in range(0, len(chunks), step):
        batch = chunks[i:i + step]
        collection.add(
            # The chunk id links each vector back to the chunk in chunks.jsonl.
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i:i + step],
            # Text and metadata are stored alongside the vectors so the collection can be
            # browsed or filtered (e.g. where={"doc_type": "api"}) without the JSONL file.
            documents=[c["text"] for c in batch],
            metadatas=[{k: c[k] for k in ("title", "url", "page", "doc_type")} for c in batch],
        )


class DenseIndex:
    """Query side of the Chroma collection built by `build_index`."""

    def __init__(self):
        # get_collection (not get_or_create) fails loudly if build_index was never run,
        # instead of silently searching an empty collection.
        self.collection = get_client().get_collection(COLLECTION)

    def count(self) -> int:
        """Number of indexed chunks."""
        return self.collection.count()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, cosine similarity) pairs, best first."""
        # Embed the query the same way as the passages (same model, normalized), plus the
        # retrieval prefix BGE expects on queries.
        vec = get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
        # query() takes a batch of query vectors and returns one result list per query,
        # hence the [vec] in and the [0] out. Only distances are requested: the caller
        # looks up chunk text by id, so fetching documents here would be wasted work.
        res = self.collection.query(query_embeddings=[vec], n_results=k, include=["distances"])
        # Chroma returns cosine *distance* (0 = identical); convert it to similarity
        # (1 = identical) so a higher score means a better match, as with BM25.
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]
