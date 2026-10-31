"""Scores candidates against a user: retrieval score, final rank score, preference fit and readiness."""

import re
from dataclasses import dataclass
from typing import Any, Self

from ..core.config import Settings
from .preferences import (
    build_explicit_preference_profile,
    detect_experience_level,
    detect_location_bucket,
    detect_profile_type,
)
from .profile_features import Features
from .tokens import normalize_terms, normalize_text, shared_items

# Penalties per item the user has avoided before (skills, interests, tokens)
AVOIDED_SKILL_PENALTY = 0.06
AVOIDED_INTEREST_PENALTY = 0.04
AVOIDED_TOKEN_PENALTY = 0.02
MAX_COUNTED_TOKENS = 5
# Scores stay below 1.0 so no recommendation reads as a "perfect" match
MAX_SCORE = 0.99

# (preference key, weight per match, reason label); at most two matches count per preference
PREFERENCE_MATCH_WEIGHTS = [
    ("looking_for", 0.045, "Shared intent"),
    ("preferred_roles", 0.055, "Preferred role fit"),
    ("preferred_industries", 0.05, "Industry fit"),
    ("preferred_locations", 0.035, "Location fit"),
    ("preferred_languages", 0.03, "Language fit"),
    ("preferred_experience_levels", 0.04, "Experience fit"),
]
DEAL_BREAKER_PENALTY = 0.07
READINESS_FIELDS = ("activitySummary", "responsivenessSummary", "outcomeSummary")


@dataclass(frozen=True)
class RankingWeights:
    """Tunable weights for smart connections, read from settings (SMART_* env variables)."""

    min_discoverable_completeness: int = 1
    profile_quality_weight: float = 0.12
    retrieval_vector_weight: float = 0.72
    retrieval_skill_weight: float = 0.08
    retrieval_interest_weight: float = 0.05
    retrieval_token_weight: float = 0.025
    rank_vector_weight: float = 0.64
    rank_skill_weight: float = 0.18
    rank_interest_weight: float = 0.08
    rank_token_weight: float = 0.06
    explicit_preference_weight: float = 0.16
    candidate_readiness_weight: float = 0.12
    semantic_search_min_score: float = 0.55
    semantic_search_fallback_min_score: float = 0.2

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            min_discoverable_completeness=settings.smart_discoverable_min_completeness,
            profile_quality_weight=settings.smart_profile_quality_weight,
            retrieval_vector_weight=settings.smart_retrieval_vector_weight,
            retrieval_skill_weight=settings.smart_retrieval_skill_weight,
            retrieval_interest_weight=settings.smart_retrieval_interest_weight,
            retrieval_token_weight=settings.smart_retrieval_token_weight,
            rank_vector_weight=settings.smart_rank_vector_weight,
            rank_skill_weight=settings.smart_rank_skill_weight,
            rank_interest_weight=settings.smart_rank_interest_weight,
            rank_token_weight=settings.smart_rank_token_weight,
            explicit_preference_weight=settings.smart_explicit_preference_weight,
            candidate_readiness_weight=settings.smart_candidate_readiness_weight,
            semantic_search_min_score=settings.semantic_search_min_score,
            semantic_search_fallback_min_score=settings.semantic_search_fallback_min_score,
        )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def profile_quality_score(features: Features) -> float:
    """0..1 measure of how filled-in a profile is, so sparse profiles rank a little lower."""
    quality = (
        (0.32 if features["headline"] else 0.0)
        + min(len(features["skills"]), 5) * 0.09
        + min(len(features["interests"]), 4) * 0.06
        + min(len(features["text_fields"]), 4) * 0.08
    )
    return round(clamp(quality, 0.0, 1.0), 4)


def profile_quality_label(quality_score: float) -> str:
    if quality_score >= 0.82:
        return "High-signal profile"
    if quality_score >= 0.62:
        return "Strong profile"
    if quality_score >= 0.42:
        return "Moderate profile"
    return "Limited profile signal"


def _shared_count(user: Features | None, candidate: Features, key: str) -> int:
    return len(shared_items((user or {}).get(key, []), candidate[key]))


def retrieval_score(
    weights: RankingWeights,
    user_features: Features,
    candidate_features: Features,
    vector_score: float,
    negative_features: Features | None = None,
) -> float:
    """Cheap first-pass score used to shortlist candidates before the full ranking."""
    shared_skills = _shared_count(user_features, candidate_features, "skills")
    shared_interests = _shared_count(user_features, candidate_features, "interests")
    shared_tokens = _shared_count(user_features, candidate_features, "tokens")
    avoided_skills = _shared_count(negative_features, candidate_features, "skills")
    avoided_interests = _shared_count(negative_features, candidate_features, "interests")
    avoided_tokens = _shared_count(negative_features, candidate_features, "tokens")
    quality_bonus = profile_quality_score(candidate_features) * weights.profile_quality_weight

    return clamp(
        vector_score * weights.retrieval_vector_weight
        + shared_skills * weights.retrieval_skill_weight
        + shared_interests * weights.retrieval_interest_weight
        + min(shared_tokens, MAX_COUNTED_TOKENS) * weights.retrieval_token_weight
        - avoided_skills * AVOIDED_SKILL_PENALTY
        - avoided_interests * AVOIDED_INTEREST_PENALTY
        - min(avoided_tokens, MAX_COUNTED_TOKENS) * AVOIDED_TOKEN_PENALTY
        + quality_bonus,
        0.0,
        MAX_SCORE,
    )


def _candidate_search_space(candidate_profile: dict[str, Any], candidate_features: Features) -> str:
    """All candidate text the user's stated preferences are matched against, lower-cased."""
    return " ".join(
        [
            detect_profile_type(candidate_features),
            detect_location_bucket(candidate_profile or {}, candidate_features),
            detect_experience_level(candidate_profile or {}, candidate_features),
            *candidate_features.get("skills", []),
            *candidate_features.get("interests", []),
            *normalize_terms([candidate_profile.get("industry"), candidate_profile.get("industries")]),
            *normalize_terms(candidate_profile.get("languages")),
            *normalize_terms(
                [candidate_profile.get("lookingFor"), candidate_profile.get("openTo"), candidate_profile.get("goals")]
            ),
            candidate_features.get("headline", ""),
            *candidate_features.get("text_fields", []),
        ]
    ).lower()


def score_explicit_preferences(
    user_profile: dict[str, Any], candidate_profile: dict[str, Any], candidate_features: Features
) -> dict[str, Any]:
    """How well a candidate fits what the user said they want, in [-0.16, 0.18], with up to two reasons."""
    prefs = build_explicit_preference_profile(user_profile or {})
    if not prefs["has_signal"]:
        return {"score": 0.0, "reasons": []}

    search_space = _candidate_search_space(candidate_profile, candidate_features)
    score, reasons = 0.0, []
    for key, weight, label in PREFERENCE_MATCH_WEIGHTS:
        matches = [item for item in normalize_terms(prefs[key]) if item.lower() in search_space]
        if matches:
            score += min(len(matches), 2) * weight
            reasons.append(f"{label}: {', '.join(matches[:2])}")

    deal_breakers = [item for item in prefs["deal_breakers"] if item.lower() in search_space]
    if deal_breakers:
        score -= min(len(deal_breakers), 2) * DEAL_BREAKER_PENALTY
        reasons.append(f"Possible mismatch: {', '.join(deal_breakers[:2])}")
    return {"score": round(clamp(score, -0.16, 0.18), 4), "reasons": reasons[:2]}


def score_candidate_readiness(weights: RankingWeights, candidate_profile: dict[str, Any]) -> dict[str, Any]:
    """Bonus for candidates whose activity summaries report high response or acceptance percentages."""
    text = " ".join([normalize_text(candidate_profile.get(key)) for key in READINESS_FIELDS])
    rates = [float(m) / 100.0 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]
    if not rates:
        return {"score": 0.0, "reasons": []}
    score = round(clamp((sum(rates) / len(rates)) * weights.candidate_readiness_weight, 0.0, 0.18), 4)
    reasons, lowered = [], text.lower()
    if "response rate" in lowered or "responsiveness" in lowered:
        reasons.append("Responsive candidate")
    if "success score" in lowered or "acceptance rate" in lowered:
        reasons.append("Good connection follow-through")
    if "high activity" in lowered:
        reasons.append("Recently active")
    return {"score": score, "reasons": reasons[:2]}
