"""Request bodies for the message assistant endpoints (field names match what the Node backend sends)."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

# Generous bounds above what the backend sends (40 messages, 1,200-character inputs, 12 history turns):
# they only stop a malformed request from building a huge prompt, which is slow and burns Groq tokens.
MAX_TEXT = 4000
MAX_TURNS = 100


def _clip(text: str | None) -> str | None:
    return text[:MAX_TEXT] if text else text


# Chat messages are the other person's words and are not length-checked by the backend: one long message
# is shortened rather than failing the whole request
ClippedText = Annotated[str | None, AfterValidator(_clip)]


class MessageAssistantTurn(BaseModel):
    """One message in the 1:1 chat; role is 'me' for the user."""

    role: str = Field(max_length=20)
    text: ClippedText = ""
    kind: str | None = Field(default="text", max_length=40)
    timestamp: str | None = Field(default=None, max_length=64)


class AssistantHistoryTurn(BaseModel):
    """One earlier Ask AI exchange, so follow-up questions keep their context."""

    role: str = Field(max_length=20)
    text: ClippedText = ""


class MessageAssistantRequest(BaseModel):
    """Body for /respond and /respond/stream; mode 'ask' answers `question`, anything else suggests replies."""

    conversation: list[MessageAssistantTurn] = Field(default=[], max_length=MAX_TURNS)
    assistant_history: list[AssistantHistoryTurn] = Field(default=[], max_length=MAX_TURNS)
    draft: str = Field(default="", max_length=MAX_TEXT)
    mode: str = Field(default="replies", max_length=20)
    question: str = Field(default="", max_length=MAX_TEXT)
    older_context: str = Field(default="", max_length=5 * MAX_TEXT)
    tone: str = Field(default="polite", max_length=40)
    other_name: str = Field(default="the other person", max_length=200)
    max_suggestions: int = Field(default=5, ge=1, le=10)
    # camelCase because that is the key the backend sends
    emojiPreference: str | None = Field(default="both", max_length=20)
