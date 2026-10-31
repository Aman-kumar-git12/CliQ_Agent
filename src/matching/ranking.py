"""Second-pass ranking of shortlisted candidates: final score and the reasons shown to the user."""

from dataclasses import dataclass
from typing import Any

from .profile_features import Features
from .scoring import (
    AVOIDED_INTEREST_PENALTY,
    AVOIDED_SKILL_PENALTY,
    AVOIDED_TOKEN_PENALTY,
    MAX_COUNTED_TOKENS,
    MAX_SCORE,
    RankingWeights,
    clamp,
    profile_quality_label,
    profile_quality_score,
    score_candidate_readiness,
    score_explicit_preferences,
)
from .tokens import shared_items

# Above this similarity to past connections, "people like the ones you connect with" is a fair reason
HISTORY_AFFINITY_THRESHOLD = 0.38
MAX_REASONS = 3


@dataclass
class RetrievedCandidate:
    """A candidate that passed the first pass, with the vector scores computed for it."""

    candidate: dict[str, Any]
    features: Features
    vector_score: float
    positive_vector_score: float
    retrieval_score: float


@dataclass
class UserContext:
    """The user's side of the comparison: own + history features, and what they avoided."""

    profile: dict[str, Any]
    features: Features
    negative_features: Features | None
    has_history: bool


def _overlap(shared: list[str], left: list[str], right: list[str]) -> float:
    return len(shared) / max(len(left), len(right), 1)


def _avoided(negative: Features | None, features: Features, key: str) -> list[str]:
    return shared_items(negative.get(key, []) if negative else [], features[key])


def _similarity_score(weights: RankingWeights, user: UserContext, item: RetrievedCandidate) -> float:
    features, user_f = item.features, user.features
    skill_overlap = _overlap(shared_items(user_f["skills"], features["skills"]), user_f["skills"], features["skills"])
    interest_overlap = _overlap(
        shared_items(user_f["interests"], features["interests"]), user_f["interests"], features["interests"]
    )
    token_overlap = _overlap(shared_items(user_f["tokens"], features["tokens"]), user_f["tokens"], features["tokens"])
    avoided_penalty = (
        len(_avoided(user.negative_features, features, "skills")) * AVOIDED_SKILL_PENALTY
        + len(_avoided(user.negative_features, features, "interests")) * AVOIDED_INTEREST_PENALTY
        + min(len(_avoided(user.negative_features, features, "tokens")), MAX_COUNTED_TOKENS) * AVOIDED_TOKEN_PENALTY
    )
    quality_bonus = profile_quality_score(features) * weights.profile_quality_weight
    return clamp(
        item.vector_score * weights.rank_vector_weight
        + skill_overlap * weights.rank_skill_weight
        + interest_overlap * weights.rank_interest_weight
        + token_overlap * weights.rank_token_weight
        - avoided_penalty
        + quality_bonus,
        0.0,
        MAX_SCORE,
    )


def _reasons(user: UserContext, item: RetrievedCandidate, extra_reasons: list[str]) -> list[str]:
    features, user_f = item.features, user.features
    shared_skills = shared_items(user_f["skills"], features["skills"])
    shared_interests = shared_items(user_f["interests"], features["interests"])
    shared_tokens = shared_items(user_f["tokens"], features["tokens"])
    reasons = []
    if shared_skills:
        reasons.append(f"Shared skills: {', '.join(shared_skills[:2])}")
    if shared_interests:
        reasons.append(f"Shared interests: {', '.join(shared_interests[:2])}")
    if shared_tokens:
        reasons.append(f"Similar profile themes around {', '.join(shared_tokens[:3])}")
    reasons.extend(extra_reasons)
    if not reasons and user.has_history and item.positive_vector_score > HISTORY_AFFINITY_THRESHOLD:
        reasons.append("Aligned with the kinds of people you usually connect with")
    if not reasons and features["headline"]:
        reasons.append(f"Profile fit: {features['headline'][:90]}")
    return reasons[:MAX_REASONS] or ["Recommended based on profile similarity"]


def rank_candidate(weights: RankingWeights, user: UserContext, item: RetrievedCandidate) -> dict[str, Any]:
    """Builds one ranked result ({id, score, reasons, profileQuality*}) for the API response."""
    candidate_profile = item.candidate.get("expertise") or {}
    preference_fit = score_explicit_preferences(user.profile or {}, candidate_profile, item.features)
    readiness = score_candidate_readiness(weights, candidate_profile)
    score = _similarity_score(weights, user, item)
    score = clamp(
        score + preference_fit["score"] * weights.explicit_preference_weight + readiness["score"], 0.0, MAX_SCORE
    )
    quality = profile_quality_score(item.features)
    return {
        "id": item.candidate.get("id"),
        "score": round(score, 4),
        "reasons": _reasons(user, item, [*preference_fit["reasons"], *readiness["reasons"]]),
        "profileQualityScore": quality,
        "profileQualityLabel": profile_quality_label(quality),
    }
