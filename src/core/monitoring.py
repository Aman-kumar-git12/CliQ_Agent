"""Optional Sentry error monitoring: does nothing unless SENTRY_DSN is set."""

import logging
from typing import Any

from .config import Settings
from .request_id import NO_REQUEST_ID, current_request_id

logger = logging.getLogger(__name__)

# Sentry's default scrubber does not know our shared-secret header
EXTRA_SENSITIVE_KEYS = ["x-internal-key", "x_internal_key"]


def _tag_request_id(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    """Lets a Sentry event be matched to the agent's (and the backend's) log lines."""
    request_id = current_request_id()
    if request_id != NO_REQUEST_ID:
        event.setdefault("tags", {})["request_id"] = request_id
    return event


def setup_sentry(settings: Settings) -> bool:
    """Initialises Sentry when a DSN is configured. Returns True when it did."""
    if not settings.sentry_dsn.strip():
        return False
    # Imported here so a run without Sentry configured never loads it
    import sentry_sdk
    from sentry_sdk.scrubber import DEFAULT_DENYLIST, EventScrubber

    sentry_sdk.init(
        dsn=settings.sentry_dsn.strip(),
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
        # Request bodies hold users' chat messages and profiles; they never leave the agent
        max_request_body_size="never",
        event_scrubber=EventScrubber(denylist=[*DEFAULT_DENYLIST, *EXTRA_SENSITIVE_KEYS]),
        before_send=_tag_request_id,
    )
    logger.info("Sentry error monitoring enabled (environment: %s)", settings.sentry_environment)
    return True
