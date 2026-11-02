"""Chat history, knowledge base search and retrieval against in-memory stand-ins for MongoDB."""

import threading

import pytest
from langchain_core.documents import Document
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from src.core.embeddings import QueryCachedEmbeddings
from src.help_chat.history import COLLECTION_NAME, MAX_HISTORY_MESSAGES, ChatHistoryStore
from src.help_chat.knowledge.vectorstore import KnowledgeStore
from src.help_chat.retrieval import Retriever
from src.help_chat.schemas import RoutedPart
from tests.unit.fakes import FakeCollection, FakeEmbeddings, UnreachableCollection


def _client(collection: object) -> dict:
    """client[database][collection], which is all ChatHistoryStore needs from a MongoClient."""
    return {"db": {COLLECTION_NAME: collection}}


# --- Chat history -------------------------------------------------------------------------------


def test_history_keeps_the_latest_messages_in_order() -> None:
    store = ChatHistoryStore(_client(FakeCollection()), "db")
    for i in range(MAX_HISTORY_MESSAGES):  # twice as many messages as are kept
        store.append_exchange("s1", f"q{i}", f"a{i}")
    store.append_exchange("other", "not mine", "no")

    messages = store.messages("s1")
    assert len(messages) == MAX_HISTORY_MESSAGES
    assert [m.content for m in messages[-2:]] == [f"q{MAX_HISTORY_MESSAGES - 1}", f"a{MAX_HISTORY_MESSAGES - 1}"]
    assert [m.type for m in messages[:2]] == ["human", "ai"]


def test_regenerate_drops_only_the_last_exchange() -> None:
    store = ChatHistoryStore(_client(FakeCollection()), "db")
    store.append_exchange("s1", "q1", "a1")
    store.append_exchange("s1", "q2", "a2")
    store.drop_last_exchange("s1")
    assert [m.content for m in store.messages("s1")] == ["q1", "a1"]


def test_history_store_starts_while_mongodb_is_down() -> None:
    collection = UnreachableCollection()
    store = ChatHistoryStore(_client(collection), "db")  # the index call fails but is not fatal
    with pytest.raises(ServerSelectionTimeoutError):
        store.messages("s1")  # the pipeline turns this into "answer without history"


# --- Knowledge base search ------------------------------------------------------------------------


class FakeVectorSearch:
    def __init__(self, error: Exception | None = None, results: list[Document] | None = None) -> None:
        self.error, self.results, self.vectors = error, results or [], []

    def similarity_search_by_vector(self, vector: list[float], k: int, pre_filter: dict | None = None) -> list:
        self.vectors.append(vector)
        if self.error:
            raise self.error
        return self.results


def _store(collection: object, vector_search: FakeVectorSearch) -> KnowledgeStore:
    store = KnowledgeStore.__new__(KnowledgeStore)  # skips building the real Atlas client
    store.collection, store.embeddings, store.vector_search = collection, FakeEmbeddings(), vector_search
    return store


def test_unreachable_database_raises_instead_of_answering_from_nothing() -> None:
    collection = UnreachableCollection()
    store = _store(collection, FakeVectorSearch(error=ServerSelectionTimeoutError("no servers")))
    with pytest.raises(ServerSelectionTimeoutError):
        store.similarity_search("how do I post")
    assert collection.calls == 0  # no second wait on the same dead database


def test_missing_index_falls_back_to_the_local_scan() -> None:
    collection = FakeCollection()
    vector = FakeEmbeddings().embed_query("how do I post")
    collection.docs = [{"text": "Click New Post", "embedding": vector, "section": "create", "heading": "Post"}]
    store = _store(collection, FakeVectorSearch(error=OperationFailure("index not found")))
    docs = store.similarity_search("how do I post", k=1)
    assert [d.page_content for d in docs] == ["Click New Post"]
    assert docs[0].metadata == {"section": "create", "heading": "Post"}


# --- Retrieval ----------------------------------------------------------------------------------


class RecordingStore:
    """Both searches must be in flight at once: each waits for the other at the barrier."""

    def __init__(self) -> None:
        self.embeddings = QueryCachedEmbeddings(CountingEmbeddings())
        self.barrier = threading.Barrier(2, timeout=5)

    def similarity_search(self, query: str, k: int = 4, pre_filter: dict | None = None) -> list[Document]:
        self.embeddings.embed_query(query)
        self.barrier.wait()
        section = "create" if pre_filter else "any"
        shared = Document(page_content="shared chunk", metadata={"section": "create", "heading": "Shared"})
        own = Document(page_content=f"{section} chunk", metadata={"section": section, "heading": section})
        return [own, shared]


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed_query(text)


def test_section_and_global_searches_run_together_and_embed_once() -> None:
    store = RecordingStore()
    docs = Retriever(store).retrieve(
        [RoutedPart(question="How do I post?", intent="steps", section="create"), RoutedPart(intent="greeting")]
    )
    assert [d.page_content for d in docs[0]] == ["create chunk", "shared chunk", "any chunk"]  # section first
    assert docs[1] == []
    assert store.embeddings.inner.calls == 1
