"""Stores knowledge base chunks in MongoDB Atlas Vector Search and searches them."""

import logging
import time

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from ...core.embeddings import EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

COLLECTION_NAME = "cliq_vectors"
INDEX_NAME = "default"
# The in-memory fallback scans the whole collection, so it is capped
FALLBACK_SCAN_LIMIT = 1000
INDEX_POLL_SECONDS = 5

ATLAS_UI_STEPS = (
    "Create it in the Atlas UI: Database > Browse Collections > {collection} > Search Indexes > "
    "Create Search Index > Atlas Vector Search > JSON Editor, name it '{index}', and paste: "
    '{{"fields": [{{"type": "vector", "path": "embedding", "numDimensions": {dims}, "similarity": "cosine"}}, '
    '{{"type": "filter", "path": "section"}}]}}'
)


class KnowledgeStore:
    """The vector collection behind the help chat (one document per knowledge base chunk)."""

    def __init__(self, collection: Collection, embeddings: Embeddings) -> None:
        self.collection = collection
        self.embeddings = embeddings
        self.vector_search = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
        )

    def similarity_search(self, query: str, k: int = 4, pre_filter: dict | None = None) -> list[Document]:
        """Atlas Vector Search, falling back to an in-memory scan when the index is missing or not ready."""
        try:
            results = self.vector_search.similarity_search(query, k=k, pre_filter=pre_filter)
            if results:
                return results
        except PyMongoError as exc:
            logger.warning("Atlas Vector Search failed: %s", exc)

        logger.warning("Using local in-memory fallback search (is the '%s' index READY?)", INDEX_NAME)
        try:
            return self._local_search(query, k, pre_filter)
        except PyMongoError as exc:
            logger.error("Local fallback search failed: %s", exc)
            return []

    def _local_search(self, query: str, k: int, pre_filter: dict | None) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        docs = [d for d in self.collection.find(pre_filter or {}).limit(FALLBACK_SCAN_LIMIT) if "embedding" in d]
        # Dot product equals cosine similarity because the stored vectors are normalized
        docs.sort(key=lambda d: np.dot(query_vector, d["embedding"]), reverse=True)
        return [
            Document(
                page_content=d["text"],
                metadata={key: value for key, value in d.items() if key not in ("_id", "embedding", "text")},
            )
            for d in docs[:k]
        ]

    def replace_documents(self, documents: list[Document]) -> None:
        """Deletes every stored chunk, then embeds and stores the new ones."""
        self.collection.delete_many({})
        logger.info("Ingesting %d chunks into %s", len(documents), self.collection.name)
        self.vector_search.add_documents(documents)

    def _index_status(self) -> tuple[str | None, bool]:
        for index in self.collection.list_search_indexes(INDEX_NAME):
            return index.get("status"), bool(index.get("queryable"))
        return None, False

    def ensure_vector_index(self, wait_seconds: float = 300) -> bool:
        """Creates the vector index if needed and waits until it is READY. Returns True once it is queryable."""
        status, _ = self._index_status()
        if status is None:
            logger.info("Creating vector search index '%s'", INDEX_NAME)
            try:
                self.vector_search.create_vector_search_index(dimensions=EMBEDDING_DIMENSIONS, filters=["section"])
            except PyMongoError as exc:
                steps = ATLAS_UI_STEPS.format(
                    collection=self.collection.full_name, index=INDEX_NAME, dims=EMBEDDING_DIMENSIONS
                )
                logger.error("Could not create the vector search index automatically: %s\n%s", exc, steps)
                return False
        else:
            logger.info("Vector search index '%s' already exists (status: %s)", INDEX_NAME, status)
        return self._wait_until_ready(wait_seconds)

    def _wait_until_ready(self, wait_seconds: float) -> bool:
        deadline = time.time() + wait_seconds
        while True:
            status, queryable = self._index_status()
            if status == "READY" and queryable:
                logger.info("Vector search index '%s' is READY", INDEX_NAME)
                return True
            if status == "FAILED" or time.time() > deadline:
                logger.error("Vector search index '%s' is not ready (status: %s)", INDEX_NAME, status)
                return False
            logger.info("Waiting for index '%s' (status: %s)", INDEX_NAME, status)
            time.sleep(INDEX_POLL_SECONDS)
