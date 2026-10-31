"""Pydantic models for the help chat API and for the router LLM's JSON output."""

from typing import Literal

from pydantic import BaseModel, Field

# ---------- API requests and responses ----------


class UserContext(BaseModel):
    """Only these fields reach the prompt; anything else the caller sends (emails, IDs) is dropped here."""

    firstname: str | None = Field(default=None, max_length=60)
    pendingRequests: int | None = Field(default=None, ge=0)
    unreadNotifications: int | None = Field(default=None, ge=0)
    expertiseFilled: bool | None = None


class ChatRequest(BaseModel):
    message: str = Field(max_length=1000)
    sessionId: str
    currentPath: str | None = Field(default=None, max_length=200)
    userContext: UserContext | None = None
    regenerate: bool = False

    def user_context_dict(self) -> dict | None:
        return self.userContext.model_dump(exclude_none=True) if self.userContext else None


class ChatResponse(BaseModel):
    question: str
    answer: str
    suggestions: list[str] = Field(default_factory=list)


class HistoryAppendRequest(BaseModel):
    sessionId: str
    question: str
    answer: str


class FeedbackRequest(BaseModel):
    messageId: str = Field(max_length=64)
    userHash: str = Field(max_length=64)
    rating: Literal[1, -1] | None = None  # None clears the vote
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=8000)
    language: str | None = Field(default=None, max_length=20)
    sources: list[str] = Field(default_factory=list, max_length=20)
    cacheable: bool = False


class OkResponse(BaseModel):
    ok: bool = True


class KbVersionResponse(BaseModel):
    kbVersion: str


# ---------- Router LLM output ----------


class RoutedPart(BaseModel):
    """One question inside the user's message, with its intent and knowledge base section."""

    question: str = ""
    intent: str = "information"
    section: str = "cliq_details"


class RouterOutput(BaseModel):
    language: str = "other"
    parts: list[RoutedPart] = Field(default_factory=list)


class ChatResult(BaseModel):
    """A complete (non-streamed) turn, used by POST /chat and the eval runner."""

    answer: str = ""
    suggestions: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    error: str | None = None
