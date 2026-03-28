import os
import math

def _parse_float(value, fallback):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else fallback
    except (TypeError, ValueError):
        return fallback

DEFAULT_TOP_K = 60
DEFAULT_RETRIEVAL_K = 180
RANKING_CONFIG = {
    "min_discoverable_completeness": max(1, int(round(_parse_float(os.getenv("SMART_DISCOVERABLE_MIN_COMPLETENESS"), 1)))),
    "profile_quality_weight": _parse_float(os.getenv("SMART_PROFILE_QUALITY_WEIGHT"), 0.12),
    "low_profile_quality_score": _parse_float(os.getenv("SMART_LOW_PROFILE_QUALITY_SCORE"), 0.42),
    "retrieval_vector_weight": _parse_float(os.getenv("SMART_RETRIEVAL_VECTOR_WEIGHT"), 0.72),
    "retrieval_skill_weight": _parse_float(os.getenv("SMART_RETRIEVAL_SKILL_WEIGHT"), 0.08),
    "retrieval_interest_weight": _parse_float(os.getenv("SMART_RETRIEVAL_INTEREST_WEIGHT"), 0.05),
    "retrieval_token_weight": _parse_float(os.getenv("SMART_RETRIEVAL_TOKEN_WEIGHT"), 0.025),
    "rank_vector_weight": _parse_float(os.getenv("SMART_RANK_VECTOR_WEIGHT"), 0.64),
    "rank_skill_weight": _parse_float(os.getenv("SMART_RANK_SKILL_WEIGHT"), 0.18),
    "rank_interest_weight": _parse_float(os.getenv("SMART_RANK_INTEREST_WEIGHT"), 0.08),
    "rank_token_weight": _parse_float(os.getenv("SMART_RANK_TOKEN_WEIGHT"), 0.06),
    "explicit_preference_weight": _parse_float(os.getenv("SMART_EXPLICIT_PREFERENCE_WEIGHT"), 0.16),
    "candidate_readiness_weight": _parse_float(os.getenv("SMART_CANDIDATE_READINESS_WEIGHT"), 0.12),
}
