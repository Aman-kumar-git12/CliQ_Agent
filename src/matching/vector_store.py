"""Profile embeddings saved in MongoDB, so a restart (or a Render instance waking up) doesn't re-embed everyone.

One document per profile: {_id: profile id, signature, model, vector}. The vector is packed float32 bytes
(384 floats = 1.5 KB), about a third of the size of a plain number array. `signature` is the hash of the
profile content; a stored vector is used only while it still matches. `model` keeps vectors from the real
embedding model and the hash fallback apart, since they can't be compared with each other.
"""

import logging
import time
from array import array
from threading import Lock
from typing import Any

from bson.binary import Binary

logger = logging.getLogger(__name__)

COLLECTION_NAME = "profile_embeddings"
# After a failed load (database unreachable at startup), try again this often instead of never
RELOAD_RETRY_SECONDS = 60


def pack_vector(vector: list[float]) -> Binary:
    return Binary(array("f", vector).tobytes())


def unpack_vector(data: bytes) -> list[float]:
    return _to_array(data).tolist()


def _to_array(data: bytes) -> array:
    values = array("f")
    values.frombytes(bytes(data))
    return values


class ProfileVectorStore:
    """Read-through store: everything is loaded into memory once, then each new embedding is written through.

    Vectors are held as float32 arrays (1.5 KB each instead of ~12 KB as a list of Python floats), so every
    profile fits in memory on a small instance. Every database error is logged and swallowed: without the
    store, matching just embeds again.
    """

    def __init__(self, collection: Any, model: str) -> None:
        self._collection = collection
        self._model = model
        self._vectors: dict[str, tuple[str, array]] = {}
        self._loaded = False
        self._next_attempt = 0.0
        self._lock = Lock()

    def _load(self) -> dict[str, tuple[str, array]]:
        with self._lock:
            if not self._loaded and time.monotonic() >= self._next_attempt:
                try:
                    stored = {
                        str(doc["_id"]): (doc["signature"], _to_array(doc["vector"]))
                        for doc in self._collection.find({"model": self._model}, {"signature": 1, "vector": 1})
                    }
                    # Vectors saved while the database was unreachable are newer than the stored ones
                    self._vectors = {**stored, **self._vectors}
                    self._loaded = True
                    logger.info("Loaded %d stored profile embeddings", len(stored))
                # Deliberately broad: an unreachable or misconfigured database must not stop matching
                except Exception:
                    self._next_attempt = time.monotonic() + RELOAD_RETRY_SECONDS
                    logger.warning("Could not load stored profile embeddings", exc_info=True)
            return self._vectors

    def preload(self) -> None:
        """Loads every stored vector now (at startup) instead of on the first ranking request."""
        self._load()

    def get(self, profile_id: str, signature: str) -> list[float] | None:
        """The stored vector for this profile, if it was made from the same content."""
        stored = self._load().get(profile_id)
        return stored[1].tolist() if stored and stored[0] == signature else None

    def save(self, profile_id: str, signature: str, vector: list[float]) -> None:
        if not vector:
            return
        packed = array("f", vector)
        self._load()
        with self._lock:
            self._vectors[profile_id] = (signature, packed)
        try:
            self._collection.update_one(
                {"_id": profile_id},
                {"$set": {"signature": signature, "model": self._model, "vector": Binary(packed.tobytes())}},
                upsert=True,
            )
        # Deliberately broad: same reason as _load; the vector is still cached in memory
        except Exception:
            logger.warning("Could not store the embedding for profile %s", profile_id, exc_info=True)
