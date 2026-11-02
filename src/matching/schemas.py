"""Request models for matching; camelCase field names are the Node backend's contract."""

from typing import Any

from pydantic import BaseModel, Field

# Generous upper bounds (the backend sends at most a few hundred candidates); they keep one oversized
# request from tying up the small instance's CPU and memory for everyone else.
MAX_CANDIDATES = 5000
MAX_RESULTS = 1000


class MatchRequest(BaseModel):
    userProfile: dict[str, Any]
    historyProfile: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(max_length=MAX_CANDIDATES)
    topK: int | None = Field(default=None, ge=0, le=MAX_RESULTS)
    retrievalK: int | None = Field(default=None, ge=0, le=MAX_CANDIDATES)


class SemanticSearchRequest(BaseModel):
    query: str = Field(max_length=500)
    candidates: list[dict[str, Any]] = Field(max_length=MAX_CANDIDATES)
    topK: int | None = Field(default=None, ge=0, le=MAX_RESULTS)
    minScore: float | None = Field(default=None, ge=-1.0, le=1.0)


class PrecomputeRequest(BaseModel):
    # Required and non-empty: an empty id is a bad request (422), not a server error
    profileId: str = Field(min_length=1, max_length=128)
    profile: dict[str, Any]
