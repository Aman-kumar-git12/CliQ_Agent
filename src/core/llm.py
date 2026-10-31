"""Groq chat models. Each is built on first use and then reused."""

from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from .config import get_settings


def _groq(model: str, temperature: float, **kwargs) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature, api_key=get_settings().groq_api_key, **kwargs)


@lru_cache
def get_classifier_llm() -> ChatGroq:
    """Stable model for classification and question rewriting."""
    return _groq(get_settings().classifier_model, 0, streaming=True)


@lru_cache
def get_response_llm() -> ChatGroq:
    """Slightly creative model for user-facing text (message assistant, expertise)."""
    return _groq(get_settings().response_model, 0.7, streaming=True)


@lru_cache
def get_fallback_response_llm() -> ChatGroq:
    """Lower-variance model for structured output and degraded-mode recovery."""
    settings = get_settings()
    return _groq(settings.fallback_model or settings.classifier_model, 0.2, streaming=True)


# The help chat models fail fast (max_retries=1): with a daily limit, more retries only delay the error.


@lru_cache
def get_help_router_llm() -> Runnable:
    """Help chat router: one JSON call. Groq JSON mode cannot stream, so streaming is disabled."""
    llm = _groq(get_settings().classifier_model, 0, disable_streaming=True, max_retries=1)
    return llm.bind(response_format={"type": "json_object"})


@lru_cache
def get_help_answer_llm() -> ChatGroq:
    """Help chat answers: low temperature so the same question gets the same facts."""
    return _groq(get_settings().response_model, 0.2, streaming=True, max_retries=1)
