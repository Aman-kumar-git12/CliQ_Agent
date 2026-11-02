"""Request and response models for the expertise generator; field names match what the Node backend sends."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


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


# The answers go into the prompt. Free text from the user is shortened rather than rejected, so a long
# answer still builds a profile while an oversized request can't make the call slow and costly.
MAX_ANSWER = 4000
Answer = Annotated[str, AfterValidator(lambda text: text[:MAX_ANSWER])]


class ExpertiseAIAnswers(BaseModel):
    """The user's answers to the Magic Builder questions."""

    model_config = ConfigDict(extra="forbid")

    role: Answer = ""
    skills: Answer = ""
    interests: Answer = ""
    about: Answer = ""
    highlights: Answer = ""
    location: Answer = ""


class ExpertiseUserInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    firstname: str = Field(max_length=200)
    lastname: str = Field(default="", max_length=200)
    email: str = Field(max_length=320)
    imageUrl: str = Field(default="", max_length=2048)


class ExpertiseGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_info: ExpertiseUserInfo
    answers: ExpertiseAIAnswers
    # A partial or empty profile stays a plain dict, so the union must not be narrowed
    existing_expertise: ExpertisePayload | dict = Field(default_factory=dict)
    mode: str = Field(default="fresh", max_length=40)


class ExpertiseGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expertise: ExpertisePayload
    generation_source: Literal["llm", "fallback"]
