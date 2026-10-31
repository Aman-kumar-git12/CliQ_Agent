"""Request bodies for the message assistant endpoints (field names match what the Node backend sends)."""

from pydantic import BaseModel


class MessageAssistantTurn(BaseModel):
    """One message in the 1:1 chat; role is 'me' for the user."""

    role: str
    text: str | None = ""
    kind: str | None = "text"
    timestamp: str | None = None


class AssistantHistoryTurn(BaseModel):
    """One earlier Ask AI exchange, so follow-up questions keep their context."""

    role: str
    text: str | None = ""


class MessageAssistantRequest(BaseModel):
    """Body for /respond and /respond/stream; mode 'ask' answers `question`, anything else suggests replies."""

    conversation: list[MessageAssistantTurn] = []
    assistant_history: list[AssistantHistoryTurn] = []
    draft: str = ""
    mode: str = "replies"
    question: str = ""
    older_context: str = ""
    tone: str = "polite"
    other_name: str = "the other person"
    max_suggestions: int = 5
    # camelCase because that is the key the backend sends
    emojiPreference: str | None = "both"
