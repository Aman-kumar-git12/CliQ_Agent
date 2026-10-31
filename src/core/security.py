"""Every feature endpoint may only be called by the CliQ backend, which sends the shared internal key."""

import hmac

from fastapi import Header, HTTPException

from .config import get_settings


def require_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: the request must carry X-Internal-Key equal to AGENT_INTERNAL_KEY."""
    expected = get_settings().agent_internal_key
    if not expected:
        raise HTTPException(status_code=503, detail="AGENT_INTERNAL_KEY is not set, so the agent accepts no calls")
    if not x_internal_key or not hmac.compare_digest(x_internal_key, expected):
        raise HTTPException(status_code=401, detail="Invalid internal key")
