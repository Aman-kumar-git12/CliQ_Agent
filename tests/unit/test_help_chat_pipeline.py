"""The streamed event sequence for normal answers, approved answers and failures."""

import asyncio
from collections.abc import AsyncIterator

import groq
import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableGenerator
from pymongo.errors import ServerSelectionTimeoutError

from src.help_chat.answering import AnswerWriter
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


# --- The answer stream retries once, only before its first token ---------------------------------

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _streaming_llm(*attempts: tuple[tuple[str, ...], BaseException | None]) -> tuple[RunnableGenerator, list[int]]:
    """Attempt i streams attempts[i][0] and then raises attempts[i][1] (if set)."""
    calls: list[int] = []

    async def stream(_prompt: AsyncIterator[object]) -> AsyncIterator[str]:
        async for _ in _prompt:
            pass
        calls.append(1)
        tokens, error = attempts[len(calls) - 1]
        for token in tokens:
            yield token
        if error:
            raise error

    return RunnableGenerator(stream), calls


async def _answer(writer: AnswerWriter) -> str:
    parts = [RoutedPart(question="How do I post?", intent="steps", section="create")]
    return "".join([t async for t in writer.stream("How do I post?", "english", parts, [[DOC]])])


async def test_answer_stream_retries_a_5xx_before_the_first_token() -> None:
    error = groq.InternalServerError("down", response=httpx.Response(503, request=REQUEST), body=None)
    llm, calls = _streaming_llm(((), error), (("Click ", "New Post"), None))
    assert await _answer(AnswerWriter(llm)) == "Click New Post"
    assert len(calls) == 2


async def test_answer_stream_never_restarts_after_a_token() -> None:
    error = groq.APIConnectionError(request=REQUEST)
    llm, calls = _streaming_llm((("Click ",), error), (("again",), None))
    with pytest.raises(groq.APIConnectionError):
        await _answer(AnswerWriter(llm))
    assert len(calls) == 1


async def test_answer_stream_fails_fast_on_rate_limits(daily_limit) -> None:
    llm, calls = _streaming_llm(((), daily_limit), (("never",), None))
    with pytest.raises(groq.RateLimitError):
        await _answer(AnswerWriter(llm))
    assert len(calls) == 1


# --- MongoDB outages ----------------------------------------------------------------------------


class UnreachableHistory(FakeHistory):
    def messages(self, session_id):
        raise ServerSelectionTimeoutError("no servers")

    def append_exchange(self, session_id, question, answer):
        raise ServerSelectionTimeoutError("no servers")


async def test_history_outage_still_answers() -> None:
    events = await collect(make_pipeline(history=UnreachableHistory()))
    assert "".join(d["text"] for e, d in events if e == "token") == "Click **New Post**."
    assert [e for e, _ in events][-2:] == ["suggestions", "done"]
    assert not any(e == "error" for e, _ in events)


async def test_knowledge_base_outage_is_an_error_that_is_never_cached() -> None:
    class UnreachableRetriever:
        def retrieve(self, parts):
            raise ServerSelectionTimeoutError("no servers")

    pipeline = make_pipeline()
    pipeline.retriever = UnreachableRetriever()
    events = await collect(pipeline)
    error = next(d for e, d in events if e == "error")
    assert error["code"] == "db_unavailable"
    assert events[-1][0] == "done" and events[-1][1]["cacheable"] is False


async def test_unexpected_failures_end_the_stream_with_error_and_done() -> None:
    events = await collect(make_pipeline(writer=FakeWriter(tokens=("Half",), error=KeyError("bug"))))
    assert [e for e, _ in events][-3:] == ["token", "error", "done"]
    assert next(d for e, d in events if e == "error")["code"] == "internal_error"


async def test_client_disconnect_stops_the_answer_stream() -> None:
    closed = asyncio.Event()

    class EndlessWriter:
        async def stream(self, message, language, parts, docs_by_part, extra_system=""):
            try:
                yield "Click "
                await asyncio.Event().wait()  # an LLM that never finishes
            finally:
                closed.set()

    history = FakeHistory()
    first_token = asyncio.Event()

    async def consume() -> None:
        async for event, _ in make_pipeline(writer=EndlessWriter(), history=history).stream("How?", "s1"):
            if event == "token":
                first_token.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_token.wait(), timeout=5)
    task.cancel()  # what Starlette does when the backend closes the connection
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()  # the upstream stream was closed, so no more tokens are generated
    assert history.saved == []
