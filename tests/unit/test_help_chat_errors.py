"""Error classification, wait-time parsing and the rate-limit breaker."""

import time

import groq
import httpx
import pytest

from src.help_chat.errors import (
    LimitBreaker,
    classify_error,
    humanize_wait,
    parse_wait_seconds,
    resolve_language,
    user_message,
)
from src.help_chat.telemetry import error_code

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def rate_limit(text: str) -> groq.RateLimitError:
    return groq.RateLimitError(text, response=httpx.Response(429, request=REQUEST), body=None)


DAILY = rate_limit(
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org` on tokens per day (TPD): "
    "Limit 200000, Used 199000. Please try again in 1h20m16.368s."
)


def test_parse_wait_seconds() -> None:
    assert parse_wait_seconds("Please try again in 1h20m16.368s.") == pytest.approx(4816.368)
    assert parse_wait_seconds("try again in 7.5s") == 7.5
    assert parse_wait_seconds("no time given") is None


def test_humanize_wait_rounds_up_in_each_language() -> None:
    assert humanize_wait(4816) == "about 1 hour 21 minutes"
    assert humanize_wait(3) == "about 10 seconds"
    assert humanize_wait(None, "hinglish") == "kuch ghante"
    assert humanize_wait(120, "hindi") == "लगभग 2 मिनट"


def test_classify_daily_limit() -> None:
    info = classify_error(DAILY)
    assert info.kind == "daily_limit"
    assert info.model == "openai/gpt-oss-120b"
    assert info.retry_after == pytest.approx(4816.368)


def test_classify_other_errors() -> None:
    assert classify_error(groq.APITimeoutError(request=REQUEST)).kind == "timeout"
    assert classify_error(groq.APIConnectionError(request=REQUEST)).kind == "unreachable"
    assert classify_error(ValueError("bug")).kind == "internal"
    assert classify_error(rate_limit("too many requests")).kind == "busy"


def test_error_codes_are_stable() -> None:
    assert error_code(DAILY) == "rate_limited"
    assert error_code(groq.APITimeoutError(request=REQUEST)) == "llm_timeout"
    assert error_code(groq.APIConnectionError(request=REQUEST)) == "llm_unreachable"
    assert error_code(KeyError("x")) == "internal_error"


def test_resolve_language_guesses_without_the_router() -> None:
    assert resolve_language("unknown", "mera password kaise reset karu") == "hinglish"
    assert resolve_language("unknown", "पासवर्ड") == "hindi"
    assert resolve_language("unknown", "reset my password") == "english"
    assert resolve_language("hindi", "anything") == "hindi"


def test_user_message_fills_in_the_wait() -> None:
    assert "about 2 minutes" in user_message("busy", "english", 120)
    assert user_message("no-such-kind") == user_message("internal")


def test_breaker_blocks_until_reset() -> None:
    breaker = LimitBreaker()
    breaker.trip("model", "daily_limit", 0.05)
    kind, left = breaker.check("model")
    assert kind == "daily_limit" and 0 < left <= 0.05
    time.sleep(0.06)
    assert breaker.check("model") is None


def test_breaker_ignores_non_limit_errors() -> None:
    breaker = LimitBreaker()
    breaker.trip("model", "timeout", 30)
    assert breaker.check("model") is None
