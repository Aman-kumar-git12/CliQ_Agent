"""Groq chat models. Each is built on first use and then reused."""

from functools import lru_cache
from typing import Any

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from .config import get_settings

# Only these Groq models accept `reasoning_effort`; any other model rejects the request with a 400
REASONING_MODEL_PREFIX = "openai/gpt-oss"


def groq_options(model: str, *, fast: bool = False) -> dict[str, Any]:
    """Timeout, retries and reasoning effort for one model (fast=True for the router and classifiers)."""
    settings = get_settings()
    options: dict[str, Any] = {
        "timeout": settings.llm_classifier_timeout_seconds if fast else settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    effort = settings.llm_reasoning_effort.strip().lower()
    if effort and model.startswith(REASONING_MODEL_PREFIX):
        options["reasoning_effort"] = effort
    return options


def _groq(model: str, temperature: float, *, fast: bool = False, **kwargs: Any) -> ChatGroq:
    options = {**groq_options(model, fast=fast), **kwargs}
    return ChatGroq(model=model, temperature=temperature, api_key=get_settings().groq_api_key, **options)


@lru_cache
def get_classifier_llm() -> ChatGroq:
    """Stable model for classification and question rewriting."""
    return _groq(get_settings().classifier_model, 0, fast=True, streaming=True)


@lru_cache
def get_response_llm() -> ChatGroq:
    """Slightly creative model for user-facing text (message assistant, expertise)."""
    return _groq(get_settings().response_model, 0.7, streaming=True)


@lru_cache
def get_fallback_response_llm() -> ChatGroq:
    """Lower-variance model for structured output and degraded-mode recovery."""
    settings = get_settings()
    return _groq(settings.fallback_model or settings.classifier_model, 0.2, streaming=True)


@lru_cache
def get_help_router_llm() -> Runnable:
    """Help chat router: one JSON call. Groq JSON mode cannot stream, so streaming is disabled."""
    llm = _groq(get_settings().classifier_model, 0, fast=True, disable_streaming=True)
    return llm.bind(response_format={"type": "json_object"})


@lru_cache
def get_help_answer_llm() -> ChatGroq:
    """Help chat answers: low temperature so the same question gets the same facts."""
    return _groq(get_settings().response_model, 0.2, streaming=True)
