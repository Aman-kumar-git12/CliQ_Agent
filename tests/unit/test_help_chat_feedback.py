"""Feedback learning: approvals, rejections, chunk ranking and knowledge base versioning."""

import pytest
from langchain_core.documents import Document

from src.help_chat.feedback import FeedbackMemory
from tests.unit.fakes import FakeCollection, FakeEmbeddings

Q = "How do I create a post?"
GOOD = "Click **New Post**, write text, click **Share Post**."
BAD = "Go to Profile > Edit and type your post."
SOURCES = ["create::How to create a post"]


@pytest.fixture
def kb() -> dict:
    return {"version": "v1"}


@pytest.fixture
def memory(kb: dict) -> FeedbackMemory:
    return FeedbackMemory(FakeCollection(), FakeEmbeddings(), lambda: kb["version"], min_approvals=2)


def test_an_answer_needs_two_distinct_approvals(memory: FeedbackMemory) -> None:
    memory.record("m1", "userA", 1, Q, GOOD, "english", SOURCES, cacheable=True)
    assert memory.lookup(Q).approved is None
    memory.record("m2", "userA", 1, Q, GOOD, "english", SOURCES, cacheable=True)
    assert memory.lookup(Q).approved is None  # same user twice still counts once
    memory.record("m3", "userB", 1, Q, GOOD, "english", SOURCES, cacheable=True)
    assert memory.lookup(Q).approved["answer"] == GOOD


def test_one_down_vote_blocks_reuse_and_marks_the_answer_to_avoid(memory: FeedbackMemory) -> None:
    memory.record("m1", "userA", 1, Q, GOOD, cacheable=True)
    memory.record("m2", "userB", 1, Q, GOOD, cacheable=True)
    memory.record("m3", "userC", -1, Q, GOOD, cacheable=True)
    assert memory.lookup(Q).approved is None

    memory.record("m4", "userD", -1, Q, BAD)
    assert BAD in memory.lookup(Q).avoid


def test_personal_answers_are_never_reused(memory: FeedbackMemory) -> None:
    memory.record("m1", "userA", 1, Q, GOOD, cacheable=False)
    memory.record("m2", "userB", 1, Q, GOOD, cacheable=False)
    assert memory.lookup(Q).approved is None


def test_clearing_a_vote_removes_it(memory: FeedbackMemory) -> None:
    memory.record("m1", "userA", 1, Q, GOOD, cacheable=True)
    memory.record("m2", "userB", 1, Q, GOOD, cacheable=True)
    memory.record("m2", "userB", None, Q, GOOD)
    assert memory.lookup(Q).approved is None


def test_votes_from_an_older_knowledge_base_are_ignored(memory: FeedbackMemory, kb: dict) -> None:
    memory.record("m1", "userA", 1, Q, GOOD, cacheable=True)
    memory.record("m2", "userB", 1, Q, GOOD, cacheable=True)
    kb["version"] = "v2"
    memory._loaded_at = 0  # skip the 60-second cache
    assert memory.lookup(Q).approved is None
    assert memory.current_kb_version() == "v2"


def _doc(heading: str) -> Document:
    return Document(page_content=heading, metadata={"section": "find", "heading": heading})


def test_rank_docs_lifts_liked_and_drops_misleading_chunks(memory: FeedbackMemory) -> None:
    for i, user in enumerate(["x1", "x2", "x3"]):
        memory.record(f"bad{i}", user, -1, f"question {i}", f"bad answer {i}", sources=["find::Old search"])
    memory.record("g1", "y1", 1, "qq1", "good a", sources=["find::Search people"])
    memory.record("g2", "y2", 1, "qq2", "good b", sources=["find::Search people"])

    ranked = memory.rank_docs([_doc("Old search"), _doc("Neutral"), _doc("Search people")], memory.chunk_scores())
    assert [d.metadata["heading"] for d in ranked] == ["Search people", "Neutral"]
    assert memory.rank_docs([_doc("Old search")], memory.chunk_scores()) != []  # never drops everything
