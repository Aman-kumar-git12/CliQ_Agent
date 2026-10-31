"""Application settings, loaded once from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every configurable value of the agent. Field names match the env variables (case-insensitive)."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_url: str = "*"
    log_level: str = "INFO"

    # MongoDB Atlas (chat history, vectors, feedback, matching profiles)
    database_url: str
    database_name: str = "website_db"

    # Groq models. The env names keep their historical "MESSAGE_ASSISTANT_" prefix.
    groq_api_key: str = ""
    classifier_model: str = Field("llama-3.1-8b-instant", alias="MESSAGE_ASSISTANT_CLASSIFIER_MODEL")
    response_model: str = Field("llama-3.1-8b-instant", alias="MESSAGE_ASSISTANT_RESPONSE_MODEL")
    fallback_model: str | None = Field(None, alias="MESSAGE_ASSISTANT_FALLBACK_MODEL")

    # Shared secret the backend sends on internal calls (feedback learning)
    agent_internal_key: str = ""

    # Help chat feedback learning
    feedback_similarity_threshold: float = 0.92
    feedback_min_approvals: int = Field(2, ge=1)

    # Smart connections ranking weights
    smart_discoverable_min_completeness: int = Field(1, ge=1)
    smart_profile_quality_weight: float = 0.12
    smart_retrieval_vector_weight: float = 0.72
    smart_retrieval_skill_weight: float = 0.08
    smart_retrieval_interest_weight: float = 0.05
    smart_retrieval_token_weight: float = 0.025
    smart_rank_vector_weight: float = 0.64
    smart_rank_skill_weight: float = 0.18
    smart_rank_interest_weight: float = 0.08
    smart_rank_token_weight: float = 0.06
    smart_explicit_preference_weight: float = 0.16
    smart_candidate_readiness_weight: float = 0.12

    # Semantic people search: the model and the hash fallback score very differently, so each has a floor
    semantic_search_min_score: float = 0.55
    semantic_search_fallback_min_score: float = 0.2

    @property
    def cors_origins(self) -> list[str]:
        return ["*"] if self.frontend_url == "*" else [self.frontend_url]


@lru_cache
def get_settings() -> Settings:
    """Reads the environment on first use, so importing this module has no side effects."""
    return Settings()
