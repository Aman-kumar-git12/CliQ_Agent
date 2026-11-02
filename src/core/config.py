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
    # "json" writes one JSON object per log line (for log aggregators); anything else is plain text
    log_format: str = "text"

    # MongoDB Atlas (chat history, vectors, feedback, matching profiles)
    database_url: str
    database_name: str = "website_db"

    # Groq models. The env names keep their historical "MESSAGE_ASSISTANT_" prefix.
    groq_api_key: str = ""
    classifier_model: str = Field("llama-3.1-8b-instant", alias="MESSAGE_ASSISTANT_CLASSIFIER_MODEL")
    response_model: str = Field("llama-3.1-8b-instant", alias="MESSAGE_ASSISTANT_RESPONSE_MODEL")
    fallback_model: str | None = Field(None, alias="MESSAGE_ASSISTANT_FALLBACK_MODEL")
    # Upper bounds on every Groq call, so a slow or overloaded provider turns into a fast, clear error.
    # For streams the timeout is the longest wait for the next chunk, not for the whole answer.
    llm_timeout_seconds: float = Field(20, gt=0)
    # The router and classifiers return a few JSON tokens and have a rule-based fallback, so they give up sooner
    llm_classifier_timeout_seconds: float = Field(8, gt=0)
    # The Groq SDK's own retries also wait out 429s (up to a minute each), which would turn a rate limit into a
    # hang; so they are off and each feature retries only the fast, transient failures itself
    llm_max_retries: int = Field(0, ge=0, le=3)
    # gpt-oss models reason before the first token; "low" makes short answers start much sooner ("" = model default)
    llm_reasoning_effort: str = "low"

    # Shared secret the backend sends on internal calls (feedback learning)
    agent_internal_key: str = ""
    # Also accepted while rotating the key, so the backend can switch over without downtime
    agent_internal_key_previous: str = ""

    # Sentry error monitoring: off unless a DSN is set
    sentry_dsn: str = ""
    sentry_environment: str = "production"
    sentry_traces_sample_rate: float = Field(0.1, ge=0.0, le=1.0)

    # Off by default: set it to this service's public URL to keep a Render instance awake (pings to localhost
    # never pass Render's proxy). An always-on free instance uses ~744 of the 750 free hours a month, so only
    # one free service can stay awake; the default leaves the agent free to sleep when idle.
    keepalive_url: str = ""

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
