"""Every feature endpoint may only be called by the CliQ backend, which sends the shared internal key."""

import hmac

from fastapi import Header, HTTPException

from .config import get_settings


def require_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: X-Internal-Key must equal AGENT_INTERNAL_KEY (or AGENT_INTERNAL_KEY_PREVIOUS)."""
    settings = get_settings()
    if not settings.agent_internal_key:
        raise HTTPException(status_code=503, detail="AGENT_INTERNAL_KEY is not set, so the agent accepts no calls")
    accepted = [key for key in (settings.agent_internal_key, settings.agent_internal_key_previous) if key]
    # Bytes, not str: compare_digest rejects non-ASCII strings, and a header may contain any latin-1 text.
    # Every key is compared (no short-circuit), so timing does not reveal which key matched.
    provided = (x_internal_key or "").encode()
    matches = [hmac.compare_digest(provided, key.encode()) for key in accepted]
    if not x_internal_key or not any(matches):
        raise HTTPException(status_code=401, detail="Invalid internal key")
