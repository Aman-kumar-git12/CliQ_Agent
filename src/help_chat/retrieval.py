"""Step 2 of a help chat turn: finds the knowledge base chunks for each part of the question."""

import logging

from langchain_core.documents import Document

from .knowledge.vectorstore import KnowledgeStore
from .routing import DIRECT_INTENTS
from .schemas import RoutedPart

logger = logging.getLogger(__name__)

RETRIEVAL_K = 4


class Retriever:
    def __init__(self, store: KnowledgeStore, k: int = RETRIEVAL_K) -> None:
        self.store = store
        self.k = k

    def retrieve(self, parts: list[RoutedPart]) -> list[list[Document]]:
        """One list of chunks per part (empty for greeting/out-of-scope parts)."""
        return [[] if p.intent in DIRECT_INTENTS else self.search(p.question, p.section) for p in parts]

    def search(self, question: str, section: str) -> list[Document]:
        """k chunks from the routed section plus k from all sections, de-duplicated.

        The router can pick a section that does not hold the answer (e.g. sending a
        request is documented under "find", not "requests"), so results from the whole
        collection are always merged in.
        """
        section_docs = self.store.similarity_search(question, k=self.k, pre_filter={"section": {"$eq": section}})
        global_docs = self.store.similarity_search(question, k=self.k)

        docs, seen = [], set()
        for doc in section_docs + global_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
        logger.info(
            "Retrieved: %s", "; ".join(f"{d.metadata.get('section')} › {d.metadata.get('heading')}" for d in docs)
        )
        return docs
