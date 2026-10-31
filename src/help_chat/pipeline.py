"""Runs one help chat turn step by step and yields Server-Sent Events.

Events: status {text}, token {text}, suggestions {items}, error {code, kind, retryAfter, message},
done {cacheable, language, parts, sources, fromFeedback}.
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .answering import AnswerWriter
from .errors import LimitBreaker, classify_error, resolve_language, user_message
from .feedback import FeedbackLookup, FeedbackMemory, source_docs, source_key
from .history import ChatHistoryStore
from .pages import build_extra_system, clean_name, describe_page, is_personal, refers_to_page
from .prompts import AVOID_ANSWERS_INSTRUCTION
from .retrieval import Retriever
from .routing import DIRECT_INTENTS, QuestionRouter
from .schemas import ChatResult, RoutedPart
from .suggestions import FollowUpSuggester
from .telemetry import error_code, log_chat

logger = logging.getLogger(__name__)

type Event = tuple[str, dict]

STATUS_UNDERSTANDING = "Understanding your question…"
STATUS_SEARCHING = "Searching CliQ docs…"
STATUS_WRITING = "Writing answer…"


@dataclass
class Turn:
    """Everything known about one question while it is being answered (logged when the turn ends)."""

    message: str
    session_id: str
    current_path: str | None
    user_context: dict | None
    regenerate: bool
    started_at: float = field(default_factory=time.perf_counter)
    language: str = "unknown"
    parts: list[RoutedPart] = field(default_factory=list)
    docs_by_part: list[list[Document]] = field(default_factory=list)
    answer: str = ""
    error: str | None = None
    cacheable: bool = False

    @property
    def parts_json(self) -> list[dict]:
        return [p.model_dump() for p in self.parts]


class ChatPipeline:
    def __init__(
        self,
        router: QuestionRouter,
        retriever: Retriever,
        writer: AnswerWriter,
        history: ChatHistoryStore,
        feedback: FeedbackMemory,
        suggester: FollowUpSuggester,
        model_name: str,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.writer = writer
        self.history = history
        self.feedback = feedback
        self.suggester = suggester
        self.model = model_name
        self.breaker = LimitBreaker()

    async def stream(
        self,
        message: str,
        session_id: str,
        current_path: str | None = None,
        user_context: dict | None = None,
        regenerate: bool = False,
    ) -> AsyncIterator[Event]:
        turn = Turn(message, session_id, current_path, user_context, regenerate)
        try:
            async for event in self._run(turn):
                yield event
        except asyncio.CancelledError:
            # The client pressed Stop or disconnected; the upstream LLM stream is cancelled with us
            turn.error = "client_cancelled"
            raise
        except Exception as exc:  # Stream boundary: every failure must become a message the user can read
            for event in self._failure_events(turn, exc):
                yield event
        finally:
            self._log(turn)

    async def ask(
        self, message: str, session_id: str, current_path: str | None = None, user_context: dict | None = None
    ) -> ChatResult:
        """Collects a full turn; used by POST /chat and the eval runner."""
        result = ChatResult()
        async for event, data in self.stream(message, session_id, current_path, user_context):
            if event == "token":
                result.answer += data["text"]
            elif event == "status":
                result.statuses.append(data["text"])
            elif event == "suggestions":
                result.suggestions = data["items"]
            elif event == "error":
                result.error = data["code"]
            elif event == "done":
                result.meta = data
        return result

    # ---------- steps ----------

    async def _run(self, turn: Turn) -> AsyncIterator[Event]:
        yield "status", {"text": STATUS_UNDERSTANDING}
        chat_history = await asyncio.to_thread(self._load_history, turn)

        page_relative = refers_to_page(turn.message)
        memory = await self._lookup_feedback(turn, bool(chat_history) or page_relative)

        if memory.approved and not turn.regenerate:
            async for event in self._replay_approved(turn, memory.approved):
                yield event
            return

        limit_events = self._limit_events(turn)
        if limit_events:
            for event in limit_events:
                yield event
            return

        # The router has no page parameter, so page-relative questions carry the page in the text
        page, _ = describe_page(turn.current_path)
        routed_message = f"{turn.message} (I am on the {page} page)" if page_relative and page else turn.message
        turn.language, turn.parts = await asyncio.to_thread(self.router.route, routed_message, chat_history)

        turn.docs_by_part = [[] for _ in turn.parts]
        if any(p.intent not in DIRECT_INTENTS for p in turn.parts):
            yield "status", {"text": STATUS_SEARCHING}
            turn.docs_by_part = await asyncio.to_thread(self._retrieve_ranked, turn.parts)

        yield "status", {"text": STATUS_WRITING}
        extra_system = self._extra_system(turn, memory.avoid)
        async for token in self.writer.stream(
            routed_message, turn.language, turn.parts, turn.docs_by_part, extra_system
        ):
            if token:
                turn.answer += token
                yield "token", {"text": token}

        async for event in self._finish(turn, page_relative):
            yield event

    def _load_history(self, turn: Turn) -> list[BaseMessage]:
        if turn.regenerate:
            # Regenerate must not see its own previous answer
            self.history.drop_last_exchange(turn.session_id)
        return self.history.messages(turn.session_id)

    async def _lookup_feedback(self, turn: Turn, depends_on_context: bool) -> FeedbackLookup:
        """Feedback only applies to standalone, non-personal questions (same rule as the shared cache)."""
        if depends_on_context or is_personal(turn.message):
            return FeedbackLookup()
        return await asyncio.to_thread(self.feedback.lookup, turn.message)

    def _limit_events(self, turn: Turn) -> list[Event]:
        """While the answer model is rate limited, reply at once instead of waiting on retries that will fail."""
        blocked = self.breaker.check(self.model)
        if not blocked:
            return []
        kind, seconds_left = blocked
        turn.error = "rate_limited"
        turn.answer = user_message(kind, resolve_language(turn.language, turn.message), seconds_left)
        return list(self._error_events(turn, kind, turn.answer, seconds_left))

    async def _replay_approved(self, turn: Turn, approved: dict) -> AsyncIterator[Event]:
        """Users already approved an answer to this question: reuse it without calling the LLM."""
        turn.answer, turn.language = approved["answer"], approved.get("language") or turn.language
        sources = approved.get("sources") or []
        for token in re.findall(r"\S+\s*|\s+", turn.answer):
            yield "token", {"text": token}
        await asyncio.to_thread(self.history.append_exchange, turn.session_id, turn.message, turn.answer)
        yield "suggestions", {"items": self.suggester.suggest(turn.message, [source_docs(sources)])}
        turn.cacheable = True
        yield (
            "done",
            {
                "cacheable": True,
                "language": turn.language,
                "parts": turn.parts_json,
                "sources": sources,
                "fromFeedback": True,
            },
        )

    def _retrieve_ranked(self, parts: list[RoutedPart]) -> list[list[Document]]:
        scores = self.feedback.chunk_scores()
        return [self.feedback.rank_docs(docs, scores) for docs in self.retriever.retrieve(parts)]

    @staticmethod
    def _extra_system(turn: Turn, avoid: list[str]) -> str:
        extra_system = build_extra_system(turn.user_context, turn.current_path)
        if avoid:
            extra_system += AVOID_ANSWERS_INSTRUCTION + "\n".join(f'- "{excerpt}"' for excerpt in avoid)
        return extra_system

    async def _finish(self, turn: Turn, page_relative: bool) -> AsyncIterator[Event]:
        await asyncio.to_thread(self.history.append_exchange, turn.session_id, turn.message, turn.answer)

        question = turn.parts[0].question if len(turn.parts) == 1 else turn.message
        yield "suggestions", {"items": self.suggester.suggest(question, turn.docs_by_part)}

        # Answers that depend on this user (their page, counts or name) must not be shared via the cache
        name = clean_name((turn.user_context or {}).get("firstname"))
        turn.cacheable = (
            bool(turn.answer)
            and not page_relative
            and not is_personal(turn.message)
            and not (name and name.lower() in turn.answer.lower())
        )
        sources = list(dict.fromkeys(source_key(d) for docs in turn.docs_by_part for d in docs))
        yield (
            "done",
            {
                "cacheable": turn.cacheable,
                "language": turn.language,
                "parts": turn.parts_json,
                "sources": sources,
                "fromFeedback": False,
            },
        )

    # ---------- failures and logging ----------

    def _failure_events(self, turn: Turn, exc: Exception) -> list[Event]:
        turn.error = error_code(exc)
        if turn.error.startswith("llm_") or turn.error == "rate_limited":
            # Provider errors are expected (quota, timeouts); one line is enough
            logger.warning("Help chat LLM error (%s): %s", turn.error, str(exc)[:300])
        else:
            logger.exception("Help chat turn failed")
        info = classify_error(exc)
        self.breaker.trip(info.model or self.model, info.kind, info.retry_after)
        text = user_message(info.kind, resolve_language(turn.language, turn.message), info.retry_after)
        # A half-streamed answer is kept, with the reason it stopped underneath
        shown = f"\n\n_{text}_" if turn.answer else text
        turn.answer += shown
        return list(self._error_events(turn, info.kind, shown, info.retry_after))

    @staticmethod
    def _error_events(turn: Turn, kind: str, text: str, retry_after: float | None) -> Iterator[Event]:
        """The message is sent as a token (so it is shown and saved) and repeated in the error event for the UI."""
        yield "token", {"text": text}
        yield (
            "error",
            {
                "code": turn.error,
                "kind": kind,
                "retryAfter": math.ceil(retry_after) if retry_after else None,
                "message": text.strip().strip("_"),
            },
        )
        yield "done", {"cacheable": False, "language": turn.language, "parts": turn.parts_json}

    def _log(self, turn: Turn) -> None:
        page, _ = describe_page(turn.current_path)
        log_chat(
            session_id=turn.session_id,
            language=turn.language,
            parts=turn.parts,
            chunks=sum(len(d) for d in turn.docs_by_part),
            model=self.model,
            started_at=turn.started_at,
            error=turn.error,
            current_page=page,
            cacheable=turn.cacheable,
        )
