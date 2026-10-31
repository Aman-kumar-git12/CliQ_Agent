"""Request and response models for the expertise generator; field names match what the Node backend sends."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExpertiseDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = ""
    address: str = ""


class ExpertisePayload(BaseModel):
    """A complete expertise profile as stored by the backend (camelCase keys are part of that contract)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    experience: str
    skills: list[str] = Field(default_factory=list)
    projects: str
    achievements: str
    interests: str
    aboutYou: str
    details: ExpertiseDetails
    format: int = 1


class ExpertiseAIAnswers(BaseModel):
    """The user's answers to the Magic Builder questions."""

    model_config = ConfigDict(extra="forbid")

    role: str = ""
    skills: str = ""
    interests: str = ""
    about: str = ""
    highlights: str = ""
    location: str = ""


class ExpertiseUserInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    firstname: str
    lastname: str = ""
    email: str
    imageUrl: str = ""


class ExpertiseGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_info: ExpertiseUserInfo
    answers: ExpertiseAIAnswers
    # A partial or empty profile stays a plain dict, so the union must not be narrowed
    existing_expertise: ExpertisePayload | dict = Field(default_factory=dict)
    mode: str = "fresh"


class ExpertiseGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expertise: ExpertisePayload
    generation_source: Literal["llm", "fallback"]
