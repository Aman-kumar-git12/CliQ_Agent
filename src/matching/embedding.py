"""Text embeddings for matching: the shared model when it loads, otherwise fallbacks, plus a per-text cache."""

import hashlib
import logging
import math
import re
from threading import Lock
from typing import Any

from ..core.embeddings import get_embeddings
from .tokens import tokenize

logger = logging.getLogger(__name__)

# BGE models retrieve better when a query (not a document) carries this instruction prefix
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
HASH_EMBEDDING_DIMENSIONS = 96


def load_embedding_backend() -> Any | None:
    """Loads the shared embedding model, falling back to raw fastembed, then to None (hash embedding).

    Model loading can fail in many ways (no network for the download, missing onnx runtime, bad cache),
    and matching must still work, so each attempt deliberately catches broadly and logs a warning.
    """
    try:
        embedder = get_embeddings()
        logger.info("Matching uses the shared FastEmbedEmbeddings model")
        return embedder
    # Deliberately broad: model download/ONNX failures vary; matching must still start with a weaker embedder
    except Exception:
        logger.warning("FastEmbedEmbeddings init failed", exc_info=True)

    try:
        # Imported here: it pulls in onnxruntime, which is only worth loading on this fallback path
        from fastembed import TextEmbedding

        embedder = TextEmbedding()
        logger.info("Matching uses fastembed TextEmbedding")
        return embedder
    # Deliberately broad: same reason; the hash embedding below always works
    except Exception:
        logger.warning("TextEmbedding init failed", exc_info=True)

    logger.warning("Matching uses the hash fallback embedding (no embedding model available)")
    return None


def hash_embedding(text: str, dimensions: int = HASH_EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic bag-of-tokens vector: each token adds +-1 to a hashed bucket, then L2-normalised."""
    vector = [0.0] * dimensions
    tokens = tokenize(re.split(r"\s+", text))
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class TextEmbedder:
    """Embeds profile text and queries with a cache keyed by the text's hash.

    `backend` is a LangChain embeddings object (embed_documents), a fastembed TextEmbedding (embed),
    or None to use the hash embedding.
    """

    def __init__(self, backend: Any | None) -> None:
        self._backend = backend
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = Lock()

    @property
    def uses_model(self) -> bool:
        return self._backend is not None

    def embed_text(self, text: str) -> list[float]:
        if not text:
            return []
        cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with self._cache_lock:
            cached_vector = self._cache.get(cache_key)
        if cached_vector is not None:
            return cached_vector

        if hasattr(self._backend, "embed_documents"):
            vector = self._backend.embed_documents([text])[0]
        elif self._backend is not None:
            vector = list(next(self._backend.embed([text])))
        else:
            vector = hash_embedding(text)

        with self._cache_lock:
            self._cache[cache_key] = vector
        return vector

    def embed_query(self, query: str) -> list[float]:
        """Embeds a short search query; only real BGE models get the instruction prefix (it is noise to the hash)."""
        if not query:
            return []
        if self.uses_model:
            return self.embed_text(f"{BGE_QUERY_INSTRUCTION}{query}")
        return self.embed_text(query)
