import hashlib
import json
import math
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover - environment dependent
    TextEmbedding = None

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:  # pragma: no cover - environment dependent
    FastEmbedEmbeddings = None

def _parse_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else fallback
    except (TypeError, ValueError):
        return fallback


PLACEHOLDER_VALUES = {
    "",
    "****",
    "professional seeker",
    "n/a",
    "na",
    "null",
    "undefined",
}
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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_meaningful_text(value: Any) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    return normalized.lower() not in PLACEHOLDER_VALUES


def _normalize_terms(value: Any) -> List[str]:
    raw_items = value if isinstance(value, list) else re.split(r"[,|\n;/]+", _normalize_text(value))

    deduped: Dict[str, str] = {}
    for item in raw_items:
        text = _normalize_text(item)
        if not _is_meaningful_text(text):
            continue
        key = text.lower()
        if key not in deduped:
            deduped[key] = text
    return list(deduped.values())


def _merge_unique_terms(*value_groups: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in value_groups:
        for item in _normalize_terms(group):
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(item)
    return merged


def _get_nested_object(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_preference_value(profile: Dict[str, Any], *keys: str) -> Any:
    containers = [
        _get_nested_object(profile),
        _get_nested_object(profile.get("preferences")),
        _get_nested_object(profile.get("matchPreferences")),
        _get_nested_object(profile.get("connectionPreferences")),
    ]

    for key in keys:
        for container in containers:
            if key in container and container.get(key) is not None:
                return container.get(key)
    return None


def _build_explicit_preference_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    looking_for = _normalize_terms(_get_preference_value(profile, "lookingFor", "openTo", "connectionGoals"))
    preferred_roles = _normalize_terms(_get_preference_value(profile, "preferredRoles", "roles", "desiredRoles"))
    preferred_industries = _normalize_terms(_get_preference_value(profile, "preferredIndustries", "industries", "desiredIndustries"))
    preferred_locations = _normalize_terms(_get_preference_value(profile, "preferredLocations", "locations", "targetLocations"))
    preferred_languages = _normalize_terms(_get_preference_value(profile, "preferredLanguages", "languages"))
    preferred_experience_levels = _normalize_terms(_get_preference_value(profile, "preferredExperienceLevels", "experienceLevels"))
    deal_breakers = _normalize_terms(_get_preference_value(profile, "dealBreakers", "avoid", "avoidTypes"))

    summary_parts = []
    if looking_for:
        summary_parts.append(f"Looking for: {', '.join(looking_for[:6])}")
    if preferred_roles:
        summary_parts.append(f"Preferred roles: {', '.join(preferred_roles[:6])}")
    if preferred_industries:
        summary_parts.append(f"Preferred industries: {', '.join(preferred_industries[:6])}")
    if preferred_locations:
        summary_parts.append(f"Preferred locations: {', '.join(preferred_locations[:6])}")
    if preferred_languages:
        summary_parts.append(f"Preferred languages: {', '.join(preferred_languages[:6])}")
    if preferred_experience_levels:
        summary_parts.append(f"Preferred experience levels: {', '.join(preferred_experience_levels[:6])}")
    if deal_breakers:
        summary_parts.append(f"Avoid: {', '.join(deal_breakers[:6])}")

    return {
        "looking_for": looking_for,
        "preferred_roles": preferred_roles,
        "preferred_industries": preferred_industries,
        "preferred_locations": preferred_locations,
        "preferred_languages": preferred_languages,
        "preferred_experience_levels": preferred_experience_levels,
        "deal_breakers": deal_breakers,
        "summary": ". ".join(summary_parts),
        "has_signal": len(summary_parts) > 0,
    }


def _tokenize(values: List[str]) -> List[str]:
    tokens = set()
    for value in values:
        text = _normalize_text(value).lower()
        for token in re.split(r"[^a-z0-9+#.]+", text):
            cleaned = token.strip()
            if len(cleaned) >= 3:
                tokens.add(cleaned)
    return sorted(tokens)


def _shared_items(left: List[str], right: List[str]) -> List[str]:
    right_lookup = {item.lower(): item for item in right}
    return [item for item in left if item.lower() in right_lookup]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class MatchingService:
    def __init__(self) -> None:
        self._embedder = self._create_embedder()
        self._text_embedding_cache: Dict[str, List[float]] = {}
        self._profile_embedding_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()

    def _create_embedder(self):
        print("[MatchingService] Initializing embedder...")
        if FastEmbedEmbeddings is not None:
            try:
                embedder = FastEmbedEmbeddings()
                print("[MatchingService] Using FastEmbedEmbeddings")
                return embedder
            except Exception as e:
                print(f"[MatchingService] FastEmbedEmbeddings init failed: {e}")

        if TextEmbedding is not None:
            try:
                embedder = TextEmbedding()
                print("[MatchingService] Using TextEmbedding")
                return embedder
            except Exception as e:
                print(f"[MatchingService] TextEmbedding init failed: {e}")

        print("[MatchingService] WARNING: Using fallback text embedding (no library available)")
        return None

    def _build_profile_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        explicit_preferences = _build_explicit_preference_profile(profile)
        skills = _merge_unique_terms(
            profile.get("skills"),
            profile.get("likedSkills"),
            profile.get("acceptedSkills"),
        )
        interests = _merge_unique_terms(
            profile.get("interests"),
            profile.get("likedInterests"),
            profile.get("acceptedInterests"),
        )
        details = profile.get("details") if isinstance(profile.get("details"), dict) else {}

        headline = ""
        for key in ("description", "aboutYou", "experience", "name"):
            if _is_meaningful_text(profile.get(key)):
                headline = _normalize_text(profile.get(key))
                break

        text_fields = [
            profile.get("description"),
            profile.get("aboutYou"),
            profile.get("experience"),
            profile.get("projects"),
            profile.get("achievements"),
            profile.get("historySummary"),
            profile.get("preferenceSummary"),
            profile.get("positiveSummary"),
            profile.get("negativeSummary"),
            profile.get("connectionHeadlines"),
            profile.get("likedThemes"),
            profile.get("avoidedThemes"),
            profile.get("explicitPreferenceSummary"),
            explicit_preferences.get("summary"),
            profile.get("activitySummary"),
            profile.get("responsivenessSummary"),
            profile.get("outcomeSummary"),
            profile.get("role"),
            profile.get("roles"),
            profile.get("industry"),
            profile.get("industries"),
            profile.get("languages"),
            profile.get("education"),
            profile.get("goals"),
            profile.get("lookingFor"),
            profile.get("openTo"),
            profile.get("availability"),
            details.get("address"),
        ]
        meaningful_text_fields = [_normalize_text(value) for value in text_fields if _is_meaningful_text(value)]

        sections = []
        if headline:
            sections.append(f"Headline: {headline}")
        if skills:
            sections.append(f"Skills: {', '.join(skills[:8])}")
        if interests:
            sections.append(f"Interests: {', '.join(interests[:8])}")
        if meaningful_text_fields:
            sections.append("Profile: " + " | ".join(meaningful_text_fields[:5]))

        tokens = _tokenize([headline, *skills, *interests, *meaningful_text_fields])
        completeness = len([headline, *skills, *interests, *meaningful_text_fields])

        return {
            "headline": headline,
            "skills": skills,
            "interests": interests,
            "text_fields": meaningful_text_fields,
            "tokens": tokens,
            "completeness": completeness,
            "text": "\n".join(sections).strip(),
            "explicit_preferences": explicit_preferences,
        }

    def _detect_profile_type(self, features: Dict[str, Any]) -> str:
        text = " ".join([features.get("headline", ""), *features.get("text_fields", [])]).lower()
        if "designer" in text or "ui/ux" in text or "ux " in text:
            return "design"
        if "manager" in text or "lead" in text or "strategy" in text:
            return "leadership"
        if "data" in text or "analyst" in text or "analytics" in text:
            return "data"
        if "devops" in text or "platform" in text or "cloud" in text:
            return "platform"
        if "developer" in text or "engineer" in text or "software" in text:
            return "engineering"
        if features.get("skills"):
            return str(features["skills"][0]).lower()
        if features.get("interests"):
            return str(features["interests"][0]).lower()
        return "generalist"

    def _detect_location_bucket(self, profile: Dict[str, Any], features: Dict[str, Any]) -> str:
        details = profile.get("details") if isinstance(profile.get("details"), dict) else {}
        raw_location = _normalize_text(details.get("address") or (features.get("text_fields") or [""])[0])
        if not raw_location:
            return "location:unknown"
        parts = [
            _normalize_text(part).lower()
            for part in re.split(r"[,/|-]+", raw_location)
            if _normalize_text(part)
        ]
        if not parts:
            return "location:unknown"
        return f"location:{parts[-1]}"

    def _detect_experience_level(self, profile: Dict[str, Any], features: Dict[str, Any]) -> str:
        source = " ".join([
            _normalize_text(profile.get("experienceLevel")),
            _normalize_text(profile.get("yearsOfExperience")),
            _normalize_text(profile.get("experience")),
            features.get("headline", ""),
            *features.get("text_fields", []),
        ]).lower()
        if not source:
            return "experience:unknown"
        if "student" in source or "intern" in source:
            return "experience:student"
        if "senior" in source or "lead" in source or "principal" in source or "10+" in source or "8+" in source:
            return "experience:senior"
        if "junior" in source or "entry" in source:
            return "experience:junior"
        if "mid" in source or "3+" in source or "4+" in source:
            return "experience:mid"
        return "experience:general"

    def _profile_quality_score(self, features: Dict[str, Any]) -> float:
        quality = (
            (0.32 if features["headline"] else 0.0)
            + min(len(features["skills"]), 5) * 0.09
            + min(len(features["interests"]), 4) * 0.06
            + min(len(features["text_fields"]), 4) * 0.08
        )
        return round(_clamp(quality, 0.0, 1.0), 4)

    def _profile_quality_label(self, quality_score: float) -> str:
        if quality_score >= 0.82:
            return "High-signal profile"
        if quality_score >= 0.62:
            return "Strong profile"
        if quality_score >= 0.42:
            return "Moderate profile"
        return "Limited profile signal"

    def _is_discoverable_profile(self, profile: Dict[str, Any]) -> bool:
        features = self._build_profile_features(profile)
        has_core_signal = (
            bool(features["headline"])
            or len(features["skills"]) > 0
            or len(features["interests"]) > 0
            or len(features["text_fields"]) > 0
        )
        return has_core_signal and features["completeness"] >= RANKING_CONFIG["min_discoverable_completeness"]

    def _profile_signature(self, profile: Dict[str, Any]) -> str:
        encoded = json.dumps(profile or {}, sort_keys=True, default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _embed_text(self, text: str) -> List[float]:
        if not text:
            return []

        cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with self._cache_lock:
            cached_vector = self._text_embedding_cache.get(cache_key)
        if cached_vector is not None:
            return cached_vector

        if hasattr(self._embedder, "embed_documents"):
            vector = self._embedder.embed_documents([text])[0]
        elif self._embedder is not None:
            vector = list(next(self._embedder.embed([text])))
        else:
            vector = self._fallback_text_embedding(text)

        with self._cache_lock:
            self._text_embedding_cache[cache_key] = vector
        return vector

    def _fallback_text_embedding(self, text: str, dimensions: int = 96) -> List[float]:
        vector = [0.0] * dimensions
        tokens = _tokenize(re.split(r"\s+", text))
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _get_profile_bundle(
        self,
        profile: Dict[str, Any],
        profile_id: Optional[str] = None,
        allow_profile_cache: bool = True,
    ) -> Dict[str, Any]:
        signature = self._profile_signature(profile)
        if allow_profile_cache and profile_id:
            with self._cache_lock:
                cached_bundle = self._profile_embedding_cache.get(profile_id)
            if cached_bundle and cached_bundle.get("signature") == signature:
                return cached_bundle

        features = self._build_profile_features(profile)
        vector = self._embed_text(features["text"]) if features["text"] else []
        bundle = {
            "signature": signature,
            "features": features,
            "vector": vector,
        }

        if allow_profile_cache and profile_id:
            with self._cache_lock:
                self._profile_embedding_cache[profile_id] = bundle

        return bundle

    def _build_history_context_profile(self, history_profile: Optional[Dict[str, Any]], mode: str = "positive") -> Dict[str, Any]:
        history_profile = history_profile or {}
        if mode == "negative":
            return {
                "skills": history_profile.get("avoidedSkills", []),
                "interests": history_profile.get("avoidedInterests", []),
                "aboutYou": history_profile.get("negativeSummary", ""),
                "projects": history_profile.get("avoidedThemes", []),
                "achievements": history_profile.get("negativeSummary", ""),
            }

        return {
            "skills": _merge_unique_terms(history_profile.get("acceptedSkills", []), history_profile.get("likedSkills", [])),
            "interests": _merge_unique_terms(history_profile.get("acceptedInterests", []), history_profile.get("likedInterests", [])),
            "aboutYou": history_profile.get("positiveSummary", ""),
            "projects": history_profile.get("acceptedHeadlines", []),
            "achievements": history_profile.get("likedThemes", []),
            "historySummary": history_profile.get("positiveSummary", ""),
        }

    def _merge_feature_sets(self, *feature_sets: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged_skills: List[str] = []
        merged_interests: List[str] = []
        merged_text_fields: List[str] = []
        merged_tokens = set()
        headline = ""

        for features in feature_sets:
            if not features:
                continue
            if not headline and features.get("headline"):
                headline = features["headline"]
            merged_skills = _merge_unique_terms(merged_skills, features.get("skills", []))
            merged_interests = _merge_unique_terms(merged_interests, features.get("interests", []))
            merged_text_fields = _merge_unique_terms(merged_text_fields, features.get("text_fields", []))
            merged_tokens.update(features.get("tokens", []))

        return {
            "headline": headline,
            "skills": merged_skills,
            "interests": merged_interests,
            "text_fields": merged_text_fields,
            "tokens": sorted(merged_tokens),
            "completeness": len([headline, *merged_skills, *merged_interests, *merged_text_fields]),
        }

    def _retrieval_score(
        self,
        user_features: Dict[str, Any],
        candidate_features: Dict[str, Any],
        vector_score: float,
        negative_features: Optional[Dict[str, Any]] = None,
    ) -> float:
        shared_skills = len(_shared_items(user_features["skills"], candidate_features["skills"]))
        shared_interests = len(_shared_items(user_features["interests"], candidate_features["interests"]))
        shared_tokens = len(_shared_items(user_features["tokens"], candidate_features["tokens"]))
        negative_skill_overlap = len(_shared_items((negative_features or {}).get("skills", []), candidate_features["skills"]))
        negative_interest_overlap = len(_shared_items((negative_features or {}).get("interests", []), candidate_features["interests"]))
        negative_token_overlap = len(_shared_items((negative_features or {}).get("tokens", []), candidate_features["tokens"]))
        profile_quality_score = self._profile_quality_score(candidate_features) * RANKING_CONFIG["profile_quality_weight"]

        return _clamp(
            vector_score * RANKING_CONFIG["retrieval_vector_weight"]
            + shared_skills * RANKING_CONFIG["retrieval_skill_weight"]
            + shared_interests * RANKING_CONFIG["retrieval_interest_weight"]
            + min(shared_tokens, 5) * RANKING_CONFIG["retrieval_token_weight"]
            - negative_skill_overlap * 0.06
            - negative_interest_overlap * 0.04
            - min(negative_token_overlap, 5) * 0.02
            + profile_quality_score,
            0.0,
            0.99,
        )

    def _score_explicit_preferences(
        self,
        user_profile: Dict[str, Any],
        candidate_profile: Dict[str, Any],
        candidate_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        preferences = _build_explicit_preference_profile(user_profile or {})
        if not preferences["has_signal"]:
            return {"score": 0.0, "reasons": []}

        candidate_role = self._detect_profile_type(candidate_features)
        candidate_location = self._detect_location_bucket(candidate_profile or {}, candidate_features)
        candidate_experience = self._detect_experience_level(candidate_profile or {}, candidate_features)
        candidate_industries = _normalize_terms([candidate_profile.get("industry"), candidate_profile.get("industries")])
        candidate_languages = _normalize_terms(candidate_profile.get("languages"))
        candidate_looking_for = _normalize_terms([
            candidate_profile.get("lookingFor"),
            candidate_profile.get("openTo"),
            candidate_profile.get("goals"),
        ])
        candidate_search_space = " ".join([
            candidate_role,
            candidate_location,
            candidate_experience,
            *candidate_features.get("skills", []),
            *candidate_features.get("interests", []),
            *candidate_industries,
            *candidate_languages,
            *candidate_looking_for,
            candidate_features.get("headline", ""),
            *candidate_features.get("text_fields", []),
        ]).lower()

        score = 0.0
        reasons: List[str] = []

        def _match(items: List[str], weight: float, label: str) -> None:
            nonlocal score
            matches = [item for item in _normalize_terms(items) if item.lower() in candidate_search_space]
            if matches:
                score += min(len(matches), 2) * weight
                reasons.append(f"{label}: {', '.join(matches[:2])}")

        _match(preferences["looking_for"], 0.045, "Shared intent")
        _match(preferences["preferred_roles"], 0.055, "Preferred role fit")
        _match(preferences["preferred_industries"], 0.05, "Industry fit")
        _match(preferences["preferred_locations"], 0.035, "Location fit")
        _match(preferences["preferred_languages"], 0.03, "Language fit")
        _match(preferences["preferred_experience_levels"], 0.04, "Experience fit")

        deal_breakers = [item for item in preferences["deal_breakers"] if item.lower() in candidate_search_space]
        if deal_breakers:
            score -= min(len(deal_breakers), 2) * 0.07
            reasons.append(f"Possible mismatch: {', '.join(deal_breakers[:2])}")

        return {
            "score": round(_clamp(score, -0.16, 0.18), 4),
            "reasons": reasons[:2],
        }

    def _score_candidate_readiness(self, candidate_profile: Dict[str, Any]) -> Dict[str, Any]:
        text = " ".join([
            _normalize_text(candidate_profile.get("activitySummary")),
            _normalize_text(candidate_profile.get("responsivenessSummary")),
            _normalize_text(candidate_profile.get("outcomeSummary")),
        ])
        numbers = [
            float(match) / 100.0
            for match in re.findall(r"(\d+(?:\.\d+)?)%", text)
        ]
        if not numbers:
            return {"score": 0.0, "reasons": []}

        avg_signal = sum(numbers) / len(numbers)
        score = round(_clamp(avg_signal * RANKING_CONFIG["candidate_readiness_weight"], 0.0, 0.18), 4)
        reasons: List[str] = []
        lowered = text.lower()
        if "response rate" in lowered or "responsiveness" in lowered:
            reasons.append("Responsive candidate")
        if "success score" in lowered or "acceptance rate" in lowered:
            reasons.append("Good connection follow-through")
        if "high activity" in lowered:
            reasons.append("Recently active")
        return {"score": score, "reasons": reasons[:2]}

    async def precompute_profile(self, profile_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        if not profile_id:
            raise ValueError("profile_id is required")

        bundle = self._get_profile_bundle(profile or {}, profile_id=profile_id, allow_profile_cache=True)
        return {
            "profileId": profile_id,
            "cached": True,
            "completeness": bundle["features"]["completeness"],
            "profileQualityScore": self._profile_quality_score(bundle["features"]),
            "profileQualityLabel": self._profile_quality_label(self._profile_quality_score(bundle["features"])),
            "hasMinimumSignal": self._is_discoverable_profile(profile or {}),
        }

    async def rank_candidates(
        self,
        user_profile: Dict[str, Any],
        history_profile: Optional[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        *,
        top_k: int = DEFAULT_TOP_K,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        user_bundle = self._get_profile_bundle(user_profile or {}, allow_profile_cache=False)
        positive_history_bundle = self._get_profile_bundle(
            self._build_history_context_profile(history_profile, "positive"),
            allow_profile_cache=False,
        ) if history_profile else None
        negative_history_bundle = self._get_profile_bundle(
            self._build_history_context_profile(history_profile, "negative"),
            allow_profile_cache=False,
        ) if history_profile else None
        user_features = self._merge_feature_sets(
            user_bundle["features"],
            positive_history_bundle["features"] if positive_history_bundle else None,
        )
        negative_features = negative_history_bundle["features"] if negative_history_bundle else None

        retrieved = []
        print(f"[MatchingService] Ranking {len(candidates)} candidates for {user_bundle.get('signature', 'unknown')}")
        
        for candidate in candidates:
            candidate_id = _normalize_text(candidate.get("id"))
            candidate_profile = candidate.get("expertise") or {}
            if not self._is_discoverable_profile(candidate_profile):
                continue

            candidate_bundle = self._get_profile_bundle(candidate_profile, profile_id=candidate_id, allow_profile_cache=True)
            base_vector_score = _cosine_similarity(user_bundle["vector"], candidate_bundle["vector"])
            positive_vector_score = (
                _cosine_similarity(positive_history_bundle["vector"], candidate_bundle["vector"])
                if positive_history_bundle
                else 0.0
            )
            negative_vector_score = (
                _cosine_similarity(negative_history_bundle["vector"], candidate_bundle["vector"])
                if negative_history_bundle
                else 0.0
            )
            vector_score = _clamp(base_vector_score * 0.64 + positive_vector_score * 0.28 - negative_vector_score * 0.22, 0.0, 0.99)
            retrieved.append(
                {
                    "candidate": candidate,
                    "bundle": candidate_bundle,
                    "vector_score": vector_score,
                    "positive_vector_score": positive_vector_score,
                    "negative_vector_score": negative_vector_score,
                    "retrieval_score": self._retrieval_score(
                        user_features,
                        candidate_bundle["features"],
                        vector_score,
                        negative_features,
                    ),
                }
            )
        
        print(f"[MatchingService] Retrieval step done: {len(retrieved)} eligible candidates")

        if not retrieved:
            return []

        retrieved.sort(key=lambda item: item["retrieval_score"], reverse=True)
        shortlisted = retrieved[: max(top_k, retrieval_k)]

        ranked_results = []
        for item in shortlisted:
            candidate = item["candidate"]
            features = item["bundle"]["features"]
            profile_quality_score = self._profile_quality_score(features)

            shared_skills = _shared_items(user_features["skills"], features["skills"])
            shared_interests = _shared_items(user_features["interests"], features["interests"])
            shared_tokens = _shared_items(user_features["tokens"], features["tokens"])
            avoided_skills = _shared_items((negative_features or {}).get("skills", []), features["skills"])
            avoided_interests = _shared_items((negative_features or {}).get("interests", []), features["interests"])
            avoided_tokens = _shared_items((negative_features or {}).get("tokens", []), features["tokens"])
            explicit_preference_match = self._score_explicit_preferences(user_profile or {}, candidate.get("expertise") or {}, features)
            candidate_readiness = self._score_candidate_readiness(candidate.get("expertise") or {})

            skill_overlap = len(shared_skills) / max(len(user_features["skills"]), len(features["skills"]), 1)
            interest_overlap = len(shared_interests) / max(len(user_features["interests"]), len(features["interests"]), 1)
            token_overlap = len(shared_tokens) / max(len(user_features["tokens"]), len(features["tokens"]), 1)
            avoided_overlap_penalty = (
                len(avoided_skills) * 0.06
                + len(avoided_interests) * 0.04
                + min(len(avoided_tokens), 5) * 0.02
            )
            profile_quality_bonus = profile_quality_score * RANKING_CONFIG["profile_quality_weight"]

            score = _clamp(
                item["vector_score"] * RANKING_CONFIG["rank_vector_weight"]
                + skill_overlap * RANKING_CONFIG["rank_skill_weight"]
                + interest_overlap * RANKING_CONFIG["rank_interest_weight"]
                + token_overlap * RANKING_CONFIG["rank_token_weight"]
                - avoided_overlap_penalty
                + profile_quality_bonus,
                0.0,
                0.99,
            )
            score = _clamp(
                score
                + explicit_preference_match["score"] * RANKING_CONFIG["explicit_preference_weight"]
                + candidate_readiness["score"],
                0.0,
                0.99,
            )

            reasons = []
            if shared_skills:
                reasons.append(f"Shared skills: {', '.join(shared_skills[:2])}")
            if shared_interests:
                reasons.append(f"Shared interests: {', '.join(shared_interests[:2])}")
            if shared_tokens:
                reasons.append(f"Similar profile themes around {', '.join(shared_tokens[:3])}")
            reasons.extend(explicit_preference_match["reasons"])
            reasons.extend(candidate_readiness["reasons"])
            if (
                not reasons
                and positive_history_bundle
                and positive_history_bundle["features"]["headline"]
                and item.get("positive_vector_score", 0) > 0.38
                and item.get("negative_vector_score", 0) < 0.2
            ):
                reasons.append("Aligned with the kinds of people you usually connect with")
            if avoided_skills or avoided_interests:
                reasons.append("Softer fit against some profiles you usually skip")
            if not reasons and features["headline"]:
                reasons.append(f"Profile fit: {features['headline'][:90]}")
            if profile_quality_score < RANKING_CONFIG["low_profile_quality_score"]:
                reasons.append("Partial profile, so this match is lower confidence")
            if len(reasons) < 3 and features["completeness"] >= 5:
                reasons.append("Well-filled profile with strong matching signal")

            ranked_results.append(
                {
                    "id": candidate.get("id"),
                    "score": round(score, 4),
                    "reasons": reasons[:3] or ["Recommended based on profile similarity"],
                    "profileQualityScore": profile_quality_score,
                    "profileQualityLabel": self._profile_quality_label(profile_quality_score),
                }
            )

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        return ranked_results[:top_k]


matching_service = MatchingService()
