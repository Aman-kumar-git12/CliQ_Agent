"""Checks for endpoints only the CliQ backend may call."""

import hmac

from fastapi import Header, HTTPException

from .config import get_settings


def require_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: the request must carry X-Internal-Key equal to AGENT_INTERNAL_KEY."""
    expected = get_settings().agent_internal_key
    if not expected:
        raise HTTPException(status_code=503, detail="Feedback learning is disabled: AGENT_INTERNAL_KEY is not set")
    if not x_internal_key or not hmac.compare_digest(x_internal_key, expected):
        raise HTTPException(status_code=401, detail="Invalid internal key")
