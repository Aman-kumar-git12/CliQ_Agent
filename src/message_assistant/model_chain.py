"""Calls the chat models in priority order, falling through to the next one when a call fails."""

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .json_output import extract_json_payload

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Every model in the chain failed or returned unusable output."""


@dataclass(frozen=True)
class NamedModel:
    """A chat model plus the label reported back as `generation_source`."""

    source: str
    llm: BaseChatModel


def _messages(system_prompt: str, user_prompt: str) -> list[BaseMessage]:
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def invoke_json_model(models: Sequence[NamedModel], system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
    """Returns (parsed JSON object, source label) from the first model that answers with valid JSON.

    Raises LLMUnavailableError, chained to the last failure, when no model succeeds.
    """
    last_error: Exception | None = None
    for model in models:
        try:
            response = model.llm.invoke(_messages(system_prompt, user_prompt))
            return extract_json_payload(getattr(response, "content", "") or ""), model.source
        # Deliberately broad: any provider, network or parsing failure should just move on to the next model.
        except Exception as exc:
            logger.warning("Message assistant model %s failed: %s", model.source, str(exc)[:200])
            last_error = exc
    raise LLMUnavailableError("No message assistant model produced a usable answer") from last_error


async def stream_with_fallback(
    models: Sequence[NamedModel], system_prompt: str, user_prompt: str, fallback_text: str
) -> AsyncIterator[str]:
    """Streams text from the first model that produces any; yields `fallback_text` if none produces text.

    A model that fails before its first chunk hands over to the next one. One that fails mid-stream ends the
    answer there: starting again with another model would repeat the text the user has already seen.
    """
    for model in models:
        streamed = False
        try:
            async for chunk in model.llm.astream(_messages(system_prompt, user_prompt)):
                content = getattr(chunk, "content", "") or ""
                if content:
                    streamed = True
                    yield content
            if streamed:
                return
        # Deliberately broad: any provider or network failure should just move on to the next model.
        except Exception as exc:
            logger.warning("Message assistant model %s failed while streaming: %s", model.source, str(exc)[:200])
            if streamed:
                return
    yield fallback_text
