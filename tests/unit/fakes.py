"""Small in-memory stand-ins for MongoDB collections and embedding models."""

import hashlib

import numpy as np


class FakeCursor(list):
    def sort(self, key: str, direction: int) -> "FakeCursor":
        return FakeCursor(sorted(self, key=lambda d: d.get(key), reverse=direction < 0))

    def limit(self, n: int) -> "FakeCursor":
        return FakeCursor(self[:n])


class FakeCollection:
    """Supports the handful of pymongo calls the feedback memory makes."""

    def __init__(self) -> None:
        self.docs: list[dict] = []

    def create_index(self, *args, **kwargs) -> None:
        pass

    def find(self, query: dict | None = None, projection: dict | None = None) -> FakeCursor:
        query = query or {}
        return FakeCursor(dict(d) for d in self.docs if all(d.get(k) == v for k, v in query.items()))

    def update_one(self, query: dict, update: dict, upsert: bool = False) -> None:
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                doc.update(update["$set"])
                return
        if upsert:
            self.docs.append(dict(update["$set"]))

    def delete_one(self, query: dict) -> None:
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in query.items())]


class FakeEmbeddings:
    """Same text -> same vector; different texts -> nearly orthogonal vectors."""

    def embed_query(self, text: str) -> list[float]:
        seed = int(hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=64).tolist()
