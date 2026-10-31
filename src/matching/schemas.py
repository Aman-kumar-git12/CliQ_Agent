"""Request models for matching; camelCase field names are the Node backend's contract."""

from typing import Any

from pydantic import BaseModel


class MatchRequest(BaseModel):
    userProfile: dict[str, Any]
    historyProfile: dict[str, Any] | None = None
    candidates: list[dict[str, Any]]
    topK: int | None = None
    retrievalK: int | None = None


class SemanticSearchRequest(BaseModel):
    query: str
    candidates: list[dict[str, Any]]
    topK: int | None = None
    minScore: float | None = None


class PrecomputeRequest(BaseModel):
    profileId: str
    profile: dict[str, Any]
