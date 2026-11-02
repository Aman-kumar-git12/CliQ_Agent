"""Step 1 of a help chat turn: one LLM call that splits the message into parts and classifies each."""

import logging
import re

import groq
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from .llm_errors import LLM_ATTEMPTS, is_quick_retry
from .prompts import LANGUAGE_INSTRUCTIONS, ROUTER_PROMPT, SECTION_HINTS
from .schemas import RoutedPart, RouterOutput

logger = logging.getLogger(__name__)

MAX_PARTS = 3
DEFAULT_SECTION = "cliq_details"
INTENTS = {"steps", "information", "navigation", "greeting", "general_assistance", "out_of_scope"}
# Intents answered without the knowledge base
DIRECT_INTENTS = {"greeting", "general_assistance", "out_of_scope"}

# Answered without the router call, which saves a second of latency on the most common opener
GREETINGS = {
    "hi", "hii", "hello", "hey", "hey there", "hello there", "namaste",
    "how are you", "how r u", "who are you",
}  # fmt: skip


def normalize(text: str) -> str:
    """Lowercases and strips punctuation/emoji so "Hello :)" matches "hello"."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def guess_language(text: str) -> str:
    return "hindi" if re.search(r"[ऀ-ॿ]", text) else "other"


class QuestionRouter:
    """Turns a user message into (language, parts) using the router LLM."""

    def __init__(self, llm: Runnable, sections: list[str]) -> None:
        self.sections = sections or [DEFAULT_SECTION]
        self.sections_text = "\n".join(
            f"- {s}: {SECTION_HINTS[s]}" if s in SECTION_HINTS else f"- {s}" for s in self.sections
        )
        self.chain = ROUTER_PROMPT | llm | StrOutputParser()

    def route(self, message: str, chat_history: list[BaseMessage] | None = None) -> tuple[str, list[RoutedPart]]:
        if normalize(message) in GREETINGS:
            language, parts = guess_language(message), [RoutedPart(question=message, intent="greeting")]
        else:
            language, parts = self._route_with_llm(message, chat_history or [])
        # The rewritten questions are the user's words, so they are only logged at DEBUG
        logger.info("Routed: language=%s parts=%s", language, [(p.intent, p.section) for p in parts])
        logger.debug("Routed questions: %s", [p.question for p in parts])
        return language, parts

    def _invoke(self, inputs: dict) -> str:
        attempt = 1
        while True:
            try:
                return self.chain.invoke(inputs)
            except groq.APIError as exc:
                if attempt >= LLM_ATTEMPTS or not is_quick_retry(exc):
                    raise
                attempt += 1
                logger.warning("Router call failed, retrying once: %s", str(exc)[:200])

    def _route_with_llm(self, message: str, chat_history: list[BaseMessage]) -> tuple[str, list[RoutedPart]]:
        try:
            raw = self._invoke({"input": message, "chat_history": chat_history, "sections": self.sections_text})
        except groq.BadRequestError as exc:
            if "json_validate_failed" not in str(exc):
                # e.g. a stray tool call that failed twice: not the user's fault, so answer from the raw message
                logger.warning("Router call rejected, answering from the raw message: %s", str(exc)[:200])
                return self.parse("", message)
            # JSON mode rejects output it can't validate; in practice that is an attempt
            # to make the router break format (prompt injection), so refuse politely.
            logger.warning("Router rejected the message, treating it as out of scope")
            return guess_language(message), [RoutedPart(question=message, intent="out_of_scope")]
        except groq.APIError as exc:
            # The router uses a different model than the answer, so its quota or outage
            # shouldn't block the answer: search with the raw message instead.
            logger.warning("Routing failed, answering from the raw message: %s", str(exc)[:200])
            return self.parse("", message)
        return self.parse(raw, message)

    def parse(self, raw: str, message: str) -> tuple[str, list[RoutedPart]]:
        """Validates router JSON, tolerating code fences; never falls back to out_of_scope."""
        start, end = raw.find("{"), raw.rfind("}")
        try:
            routed = RouterOutput.model_validate_json(raw[start : end + 1]) if start != -1 else None
        except (ValidationError, ValueError):
            routed = None

        if not routed or not routed.parts:
            # Unclear routing: still try the knowledge base with the raw message
            return guess_language(message), [RoutedPart(question=message)]

        parts = [
            RoutedPart(
                question=part.question.strip() or message,
                intent=part.intent if part.intent in INTENTS else "information",
                section=part.section if part.section in self.sections else DEFAULT_SECTION,
            )
            for part in routed.parts[:MAX_PARTS]
        ]
        language = routed.language if routed.language in LANGUAGE_INSTRUCTIONS else "other"
        return language, parts
