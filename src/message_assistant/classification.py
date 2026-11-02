"""LLM classifiers for safety and routing, each with a rule-based fallback when the models fail."""

import contextvars
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from . import prompts
from .conversation import (
    Turn,
    build_context_sections,
    context_confidence,
    get_last_other_person_message,
)
from .model_chain import LLMUnavailableError, NamedModel, invoke_json_model
from .routing import is_contextual_chat_question, should_use_general_reply_mode
from .safety import contains_unsafe_messaging_request, join_request_parts

logger = logging.getLogger(__name__)

ROUTING_MODES = frozenset({"contextual", "general"})
HEURISTIC_SOURCE = "heuristic_fallback"
# Each request runs at most one classifier next to the safety check; this bounds concurrent Groq calls
CLASSIFIER_THREADS = 8

T = TypeVar("T")


def distinct_models(models: Sequence[NamedModel]) -> tuple[NamedModel, ...]:
    """Drops a model that is the same Groq model as an earlier one: it shares its outage and rate limit."""
    seen: set[str] = set()
    distinct = []
    for model in models:
        name = getattr(model.llm, "model_name", None)
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        distinct.append(model)
    return tuple(distinct)


class ModeClassifier:
    """Asks the models how to handle a request; returns (mode, source) where source names who decided."""

    def __init__(self, models: Sequence[NamedModel]) -> None:
        self._models = distinct_models(models)
        self._pool = ThreadPoolExecutor(max_workers=CLASSIFIER_THREADS, thread_name_prefix="message-ai")

    def classify_unless_blocked(self, request_text: str, classify: Callable[[], T]) -> T | None:
        """Runs the safety check and a routing classifier at the same time (one Groq round trip of waiting
        instead of two). Returns the classifier's result, or None when the request is blocked."""
        if not join_request_parts(request_text):
            return classify()  # nothing for the safety model to check
        # copy_context so the worker's log lines keep the request id
        future = self._pool.submit(contextvars.copy_context().run, classify)
        if self.should_block(request_text):
            return None
        return future.result()

    def should_block(self, *parts: str) -> bool:
        """True when the keyword screen or the safety model flags the request."""
        return contains_unsafe_messaging_request(*parts) or self._model_flags_unsafe(*parts)

    def _model_flags_unsafe(self, *parts: str) -> bool:
        text = join_request_parts(*parts)
        if not text:
            return False
        try:
            payload, _ = invoke_json_model(
                self._models, prompts.SAFETY_CLASSIFIER_SYSTEM_PROMPT, prompts.safety_classifier_user_prompt(text)
            )
        except LLMUnavailableError:
            # Fail open: the keyword screen has already run, and blocking all help on an outage is worse.
            logger.warning("Safety classifier unavailable; allowing request")
            return False
        allow = payload.get("allow")
        return not allow if isinstance(allow, bool) else False

    def classify_question_mode(self, question: str = "") -> tuple[str, str]:
        """'contextual' or 'general' for an Ask AI question."""
        normalized = (question or "").strip()
        if not normalized:
            return "contextual", "fallback_empty_question"
        result = self._classify(prompts.QUESTION_MODE_SYSTEM_PROMPT, prompts.question_mode_user_prompt(normalized))
        if result is not None:
            return result
        return ("contextual" if is_contextual_chat_question(question) else "general"), HEURISTIC_SOURCE

    def classify_reply_mode(
        self, conversation: list[Turn] | None = None, draft: str = "", older_context: str = ""
    ) -> tuple[str, str]:
        """'contextual' or 'general' (opener suggestions) for reply generation."""
        turns = conversation or []
        conversation_text, older_text = build_context_sections(turns, older_context)
        user_prompt = prompts.reply_mode_user_prompt(
            older_text,
            conversation_text,
            get_last_other_person_message(turns),
            draft,
            context_confidence(turns, older_context),
        )
        result = self._classify(prompts.REPLY_MODE_SYSTEM_PROMPT, user_prompt)
        if result is not None:
            return result
        return ("general" if should_use_general_reply_mode(turns, draft) else "contextual"), HEURISTIC_SOURCE

    def _classify(self, system_prompt: str, user_prompt: str) -> tuple[str, str] | None:
        """(mode, model source) when a model returns a valid mode, else None."""
        try:
            payload, source = invoke_json_model(self._models, system_prompt, user_prompt)
        except LLMUnavailableError:
            logger.warning("Routing classifier unavailable; using heuristic fallback")
            return None
        mode = str(payload.get("mode") or "").strip().lower()
        return (mode, source) if mode in ROUTING_MODES else None
