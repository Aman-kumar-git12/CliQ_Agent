import asyncio
import hashlib
import json
import sys
import time

import groq


def hash_session(session_id):
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:12]


def error_code(exc):
    """Short, stable error codes for dashboards instead of raw exception text."""
    if isinstance(exc, asyncio.CancelledError):
        return "client_cancelled"
    if isinstance(exc, groq.RateLimitError):
        return "rate_limited"
    if isinstance(exc, groq.APITimeoutError):
        return "llm_timeout"
    if isinstance(exc, groq.APIConnectionError):
        return "llm_unreachable"
    if isinstance(exc, groq.BadRequestError):
        return "llm_bad_request"
    if isinstance(exc, groq.APIError):
        return "llm_error"
    return "internal_error"


def log_chat(*, session_id, language, parts, chunks, model, started_at, error=None, current_page=None, cacheable=None):
    """Writes one JSON line per chat request to stdout (Render/Docker collect it as a log line)."""
    record = {
        "event": "aichat",
        "session": hash_session(session_id),
        "language": language,
        "parts": [{"intent": p.get("intent"), "section": p.get("section")} for p in parts or []],
        "chunks": chunks,
        "model": model,
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
        "error": error,
        "page": current_page,
        "cacheable": cacheable,
    }
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()
