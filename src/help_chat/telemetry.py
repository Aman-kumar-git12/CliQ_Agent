"""One structured log line per help chat turn, for dashboards (no message text, no raw session ids)."""

import asyncio
import hashlib
import json
import logging
import time

import groq

from .schemas import RoutedPart

logger = logging.getLogger("help_chat.telemetry")

# Checked in order: the groq exception types subclass each other, most specific first
ERROR_CODES: list[tuple[type[BaseException], str]] = [
    (asyncio.CancelledError, "client_cancelled"),
    (groq.RateLimitError, "rate_limited"),
    (groq.APITimeoutError, "llm_timeout"),
    (groq.APIConnectionError, "llm_unreachable"),
    (groq.BadRequestError, "llm_bad_request"),
    (groq.APIError, "llm_error"),
]


def hash_session(session_id: str) -> str:
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:12]


def error_code(exc: BaseException) -> str:
    """Short, stable error codes for dashboards instead of raw exception text."""
    for error_type, code in ERROR_CODES:
        if isinstance(exc, error_type):
            return code
    return "internal_error"


def log_chat(
    *,
    session_id: str,
    language: str,
    parts: list[RoutedPart],
    chunks: int,
    model: str,
    started_at: float,
    error: str | None = None,
    current_page: str | None = None,
    cacheable: bool | None = None,
) -> None:
    """Logs the turn as a JSON object so Render/Docker logs can be filtered by "event": "aichat"."""
    record = {
        "event": "aichat",
        "session": hash_session(session_id),
        "language": language,
        "parts": [{"intent": p.intent, "section": p.section} for p in parts],
        "chunks": chunks,
        "model": model,
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
        "error": error,
        "page": current_page,
        "cacheable": cacheable,
    }
    logger.info(json.dumps(record, ensure_ascii=False))
