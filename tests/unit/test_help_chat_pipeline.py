"""The streamed event sequence for normal answers, approved answers and failures."""

import groq
import httpx
import pytest
from langchain_core.documents import Document

from src.help_chat.feedback import FeedbackLookup
from src.help_chat.pipeline import ChatPipeline
from src.help_chat.schemas import RoutedPart
from src.help_chat.suggestions import FollowUpSuggester

DOC = Document(page_content="Click **New Post**.", metadata={"section": "create", "heading": "How to create a post"})


class FakeRouter:
    def route(self, message, chat_history):
        return "english", [RoutedPart(question=message, intent="steps", section="create")]


class FakeRetriever:
    def retrieve(self, parts):
        return [[DOC] for _ in parts]


class FakeWriter:
    def __init__(self, tokens=("Click ", "**New Post**."), error=None):
        self.tokens, self.error = tokens, error

    async def stream(self, message, language, parts, docs_by_part, extra_system=""):
        for token in self.tokens:
            yield token
        if self.error:
            raise self.error


class FakeHistory:
    def __init__(self):
        self.saved = []

    def messages(self, session_id):
        return []

    def append_exchange(self, session_id, question, answer):
        self.saved.append((question, answer))

    def drop_last_exchange(self, session_id):
        pass


class FakeFeedback:
    def __init__(self, approved=None):
        self.approved = approved

    def lookup(self, question):
        return FeedbackLookup(approved=self.approved)

    def chunk_scores(self):
        return {}

    def rank_docs(self, docs, scores):
        return docs


def make_pipeline(writer=None, feedback=None, history=None) -> ChatPipeline:
    return ChatPipeline(
        router=FakeRouter(),
        retriever=FakeRetriever(),
        writer=writer or FakeWriter(),
        history=history or FakeHistory(),
        feedback=feedback or FakeFeedback(),
        suggester=FollowUpSuggester([]),
        model_name="test-model",
    )


async def collect(pipeline: ChatPipeline, message: str = "How do I create a post?") -> list[tuple[str, dict]]:
    return [event async for event in pipeline.stream(message, "session-1")]


async def test_normal_answer_event_sequence() -> None:
    history = FakeHistory()
    events = await collect(make_pipeline(history=history))

    assert [e for e, _ in events] == ["status", "status", "status", "token", "token", "suggestions", "done"]
    done = events[-1][1]
    assert done["sources"] == ["create::How to create a post"]
    assert done["fromFeedback"] is False
    assert done["cacheable"] is True
    assert done["parts"] == [{"question": "How do I create a post?", "intent": "steps", "section": "create"}]
    assert history.saved == [("How do I create a post?", "Click **New Post**.")]


async def test_approved_answer_is_replayed_without_the_llm() -> None:
    approved = {"answer": "Saved answer", "language": "english", "sources": ["create::How to create a post"]}
    writer = FakeWriter(error=AssertionError("LLM must not be called"))
    events = await collect(make_pipeline(writer=writer, feedback=FakeFeedback(approved)))

    assert "".join(d["text"] for e, d in events if e == "token") == "Saved answer"
    assert events[-1][1]["fromFeedback"] is True


async def test_personal_question_skips_feedback_memory() -> None:
    approved = {"answer": "Saved answer", "sources": []}
    events = await collect(make_pipeline(feedback=FakeFeedback(approved)), "How many pending requests do I have?")
    assert events[-1][1]["fromFeedback"] is False
    assert events[-1][1]["cacheable"] is False


@pytest.fixture
def daily_limit() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com")
    text = "Rate limit reached for model `test-model` on tokens per day (TPD). Please try again in 10m0s."
    return groq.RateLimitError(text, response=httpx.Response(429, request=request), body=None)


async def test_rate_limit_becomes_a_friendly_error_and_opens_the_breaker(daily_limit) -> None:
    pipeline = make_pipeline(writer=FakeWriter(tokens=(), error=daily_limit))
    events = await collect(pipeline)

    error = next(d for e, d in events if e == "error")
    assert error["code"] == "rate_limited"
    assert error["kind"] == "daily_limit"
    assert error["retryAfter"] == 600
    assert "10 minutes" in error["message"]
    assert events[-1] == ("done", {"cacheable": False, "language": "english", "parts": events[-1][1]["parts"]})

    # The next question is answered from the breaker at once, without routing
    second = await collect(pipeline)
    assert [e for e, _ in second] == ["status", "token", "error", "done"]
    assert next(d for e, d in second if e == "error")["code"] == "rate_limited"


async def test_half_streamed_answer_keeps_its_text(daily_limit) -> None:
    events = await collect(make_pipeline(writer=FakeWriter(tokens=("Partial",), error=daily_limit)))
    text = "".join(d["text"] for e, d in events if e == "token")
    assert text.startswith("Partial\n\n_")


async def test_ask_collects_the_whole_turn() -> None:
    result = await make_pipeline().ask("How do I create a post?", "session-1")
    assert result.answer == "Click **New Post**."
    assert result.error is None
    assert len(result.statuses) == 3
