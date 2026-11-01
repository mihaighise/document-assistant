"""Dense retrieval: chunk embeddings stored in a persistent Chroma collection."""

from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from sklearn_assistant.config import CHROMA_DIR, EMBED_MODEL

COLLECTION = "chunks"
# BGE v1.5 recommends this prefix on short queries (not on passages) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """The embedding model, loaded once per process (downloads on first use)."""
    return SentenceTransformer(EMBED_MODEL)


def get_client() -> chromadb.ClientAPI:
    """Client for the on-disk Chroma store."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def build_index(chunks: list[dict], batch_size: int = 64) -> None:
    """Embed every chunk and (re)create the Chroma collection from scratch.

    Rebuilding rather than upserting means chunks dropped by a re-ingest don't
    linger in the index.
    """
    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})

    # Normalized vectors make cosine distance equal to 1 - dot product.
    embeddings = get_model().encode([c["text"] for c in chunks], batch_size=batch_size,
                                    normalize_embeddings=True, show_progress_bar=True)
    # Chroma caps how many records a single add() may contain.
    step = client.get_max_batch_size()
    for i in range(0, len(chunks), step):
        batch = chunks[i:i + step]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i:i + step],
            documents=[c["text"] for c in batch],
            metadatas=[{k: c[k] for k in ("title", "url", "page", "doc_type")} for c in batch],
        )


class DenseIndex:
    """Query side of the Chroma collection built by `build_index`."""

    def __init__(self):
        self.collection = get_client().get_collection(COLLECTION)

    def count(self) -> int:
        """Number of indexed chunks."""
        return self.collection.count()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, cosine similarity) pairs, best first."""
        vec = get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
        res = self.collection.query(query_embeddings=[vec], n_results=k, include=["distances"])
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]
