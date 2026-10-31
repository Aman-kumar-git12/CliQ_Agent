"""Groq error groups shared by the help chat steps."""

import groq

# Only transient failures are worth retrying. A 400 (e.g. JSON mode's json_validate_failed on a
# prompt-injection message) fails the same way every time, so retrying only adds seconds.
# Rate limits are left to the Groq SDK's own retry and the pipeline's limit breaker, so a daily limit fails fast.
TRANSIENT_ERRORS = (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)

# gpt-oss models on Groq occasionally emit a stray tool call ("Tool choice is none, but model called a tool")
LLM_ATTEMPTS = 3
