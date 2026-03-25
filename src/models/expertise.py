from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExpertiseDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = ""
    address: str = ""


class ExpertisePayload(BaseModel):
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
    existing_expertise: ExpertisePayload | dict = Field(default_factory=dict)
    mode: str = "fresh"


class ExpertiseGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expertise: ExpertisePayload
    generation_source: Literal["llm", "fallback"]
