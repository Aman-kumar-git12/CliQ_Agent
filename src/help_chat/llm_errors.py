"""Groq error groups shared by the help chat steps."""

import groq

# One retry at most: each attempt can take up to the LLM timeout, and the user is waiting
LLM_ATTEMPTS = 2


def is_quick_retry(exc: BaseException) -> bool:
    """Failures that come back fast and usually pass on a second try.

    - A dropped connection or a 5xx from Groq.
    - gpt-oss models occasionally emit a stray tool call (400 "tool_use_failed"); the same request then works.
    Not retried: timeouts (the user already waited the full timeout), rate limits (left to the limit breaker,
    so a daily limit fails at once) and other 400s such as JSON mode's json_validate_failed, which fail the
    same way every time.
    """
    if isinstance(exc, groq.APITimeoutError):
        return False
    if isinstance(exc, groq.BadRequestError):
        return "tool_use_failed" in str(exc)
    return isinstance(exc, (groq.APIConnectionError, groq.InternalServerError))
