import hashlib
import re
from typing import Any, Dict, List, Optional
from threading import Lock

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:
    FastEmbedEmbeddings = None

from .utils import tokenize

class EmbeddingManager:
    def __init__(self) -> None:
        self._embedder = self._create_embedder()
        self._text_embedding_cache: Dict[str, List[float]] = {}
        self._cache_lock = Lock()

    def _create_embedder(self):
        print("[MatchingService] Initializing embedder...")
        if FastEmbedEmbeddings is not None:
            try:
                embedder = FastEmbedEmbeddings()
                print("[MatchingService] Using FastEmbedEmbeddings")
                return embedder
            except Exception as e:
                print(f"[MatchingService] FastEmbedEmbeddings init failed: {e}")

        if TextEmbedding is not None:
            try:
                embedder = TextEmbedding()
                print("[MatchingService] Using TextEmbedding")
                return embedder
            except Exception as e:
                print(f"[MatchingService] TextEmbedding init failed: {e}")

        print("[MatchingService] WARNING: Using fallback text embedding (no library available)")
        return None

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with self._cache_lock:
            cached_vector = self._text_embedding_cache.get(cache_key)
        if cached_vector is not None:
            return cached_vector

        if hasattr(self._embedder, "embed_documents"):
            vector = self._embedder.embed_documents([text])[0]
        elif self._embedder is not None:
            vector = list(next(self._embedder.embed([text])))
        else:
            vector = self._fallback_text_embedding(text)

        with self._cache_lock:
            self._text_embedding_cache[cache_key] = vector
        return vector

    def _fallback_text_embedding(self, text: str, dimensions: int = 96) -> List[float]:
        vector = [0.0] * dimensions
        tokens = tokenize(re.split(r"\s+", text))
        
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = sum(value * value for value in vector)**0.5
        if norm == 0: return vector
        return [value / norm for value in vector]
