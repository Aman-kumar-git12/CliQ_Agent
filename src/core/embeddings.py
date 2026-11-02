"""The local embedding model, loaded once and shared by the help chat and matching."""

from collections import OrderedDict
from functools import lru_cache
from threading import Lock

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings

from .config import PROJECT_ROOT

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge-small-en-v1.5 produces 384-dimensional vectors
EMBEDDING_DIMENSIONS = 384
# fastembed's default cache is in /tmp, which a Render instance loses every time it sleeps, so each cold
# start downloaded the model again. Inside the project it is kept with the build (see scripts/download_model.py).
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"
QUERY_CACHE_SIZE = 512


class QueryCachedEmbeddings(Embeddings):
    """Remembers the most recent query vectors.

    One help chat turn embeds the same question for the feedback lookup and for each vector search, and
    common questions repeat across users. Documents are not cached: they are embedded once at ingest.
    """

    def __init__(self, inner: Embeddings, max_entries: int = QUERY_CACHE_SIZE) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            vector = self._cache.get(text)
            if vector is not None:
                self._cache.move_to_end(text)
                return vector
        vector = self.inner.embed_query(text)
        with self._lock:
            self._cache[text] = vector
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return vector


def load_embedding_model() -> FastEmbedEmbeddings:
    """Loads (downloading on first use) the model; the build runs this once so startup only reads the disk."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, cache_dir=str(MODEL_CACHE_DIR))


@lru_cache
def get_embeddings() -> QueryCachedEmbeddings:
    return QueryCachedEmbeddings(load_embedding_model())
