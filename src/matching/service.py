"""Smart connections ranking and semantic people search over candidate profiles sent by the backend."""

from threading import Lock
from typing import Any

from .embedding import TextEmbedder, cosine_similarity
from .profile_features import (
    build_history_context_profile,
    build_profile_features,
    has_minimum_signal,
    merge_feature_sets,
    profile_signature,
)
from .ranking import RetrievedCandidate, UserContext, rank_candidate
from .scoring import MAX_SCORE, RankingWeights, clamp, profile_quality_label, profile_quality_score, retrieval_score
from .tokens import normalize_text, tokenize

DEFAULT_TOP_K = 60
DEFAULT_RETRIEVAL_K = 180
DEFAULT_SEARCH_TOP_K = 20
# How the base, history-positive and history-negative similarities blend into one vector score
BASE_VECTOR_WEIGHT = 0.64
POSITIVE_VECTOR_WEIGHT = 0.28
NEGATIVE_VECTOR_WEIGHT = 0.22

type Bundle = dict[str, Any]


class MatchingService:
    """Ranks candidates for a user and searches people by free text.

    Candidate embeddings are cached per profile id and reused until the profile content changes.
    """

    def __init__(self, embedder: TextEmbedder, weights: RankingWeights) -> None:
        self._embedder = embedder
        self._weights = weights
        self._profile_cache: dict[str, Bundle] = {}
        self._cache_lock = Lock()

    def _is_discoverable(self, profile: dict[str, Any]) -> bool:
        return has_minimum_signal(build_profile_features(profile), self._weights.min_discoverable_completeness)

    def _get_profile_bundle(self, profile: dict[str, Any], profile_id: str | None = None) -> Bundle:
        """Features + embedding for a profile; cached only when an id is given (candidates, not ad-hoc users)."""
        signature = profile_signature(profile)
        if profile_id:
            with self._cache_lock:
                cached_bundle = self._profile_cache.get(profile_id)
            if cached_bundle and cached_bundle.get("signature") == signature:
                return cached_bundle

        features = build_profile_features(profile)
        vector = self._embedder.embed_text(features["text"]) if features["text"] else []
        bundle = {"signature": signature, "features": features, "vector": vector}
        if profile_id:
            with self._cache_lock:
                self._profile_cache[profile_id] = bundle
        return bundle

    async def precompute_profile(self, profile_id: str, profile: dict[str, Any] | None) -> dict[str, Any]:
        """Warms the embedding cache for one profile and reports its quality."""
        if not profile_id:
            raise ValueError("profile_id is required")
        bundle = self._get_profile_bundle(profile or {}, profile_id=profile_id)
        quality = profile_quality_score(bundle["features"])
        return {
            "profileId": profile_id,
            "cached": True,
            "completeness": bundle["features"]["completeness"],
            "profileQualityScore": quality,
            "profileQualityLabel": profile_quality_label(quality),
            "hasMinimumSignal": self._is_discoverable(profile or {}),
        }

    async def semantic_search(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Ranks candidates against a free-text query ("people who know React and ML").

        Profile vectors come from the same cache as rank_candidates, so a warm cache only embeds the query.
        """
        query = normalize_text(query)
        if not query or not candidates:
            return []
        query_vector = self._embedder.embed_query(query)
        if not query_vector:
            return []
        if min_score is None:
            min_score = (
                self._weights.semantic_search_min_score
                if self._embedder.uses_model
                else self._weights.semantic_search_fallback_min_score
            )
        query_tokens = set(tokenize([query]))

        results = []
        for candidate in candidates:
            candidate_id = normalize_text(candidate.get("id"))
            profile = candidate.get("expertise") or {}
            if not candidate_id or not self._is_discoverable(profile):
                continue
            bundle = self._get_profile_bundle(profile, profile_id=candidate_id)
            score = cosine_similarity(query_vector, bundle["vector"])
            if score >= min_score:
                reasons = _search_reasons(bundle["features"], query_tokens)
                results.append({"id": candidate.get("id"), "score": round(score, 4), "reasons": reasons})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def rank_candidates(
        self,
        user_profile: dict[str, Any],
        history_profile: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = DEFAULT_TOP_K,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
    ) -> list[dict[str, Any]]:
        """Recommends connections: a cheap retrieval pass shortlists, then a fuller score orders the shortlist."""
        if not candidates:
            return []
        user_bundle = self._get_profile_bundle(user_profile or {})
        positive_bundle = negative_bundle = None
        if history_profile:
            positive_bundle = self._get_profile_bundle(build_history_context_profile(history_profile, "positive"))
            negative_bundle = self._get_profile_bundle(build_history_context_profile(history_profile, "negative"))
        user = UserContext(
            profile=user_profile,
            features=merge_feature_sets(
                user_bundle["features"], positive_bundle["features"] if positive_bundle else None
            ),
            negative_features=negative_bundle["features"] if negative_bundle else None,
            has_history=positive_bundle is not None,
        )

        retrieved = [
            self._retrieve(candidate, user, user_bundle, positive_bundle, negative_bundle)
            for candidate in candidates
            if self._is_discoverable(candidate.get("expertise") or {})
        ]
        if not retrieved:
            return []
        retrieved.sort(key=lambda x: x.retrieval_score, reverse=True)
        shortlisted = retrieved[: max(top_k, retrieval_k)]

        ranked = [rank_candidate(self._weights, user, item) for item in shortlisted]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _retrieve(
        self,
        candidate: dict[str, Any],
        user: UserContext,
        user_bundle: Bundle,
        positive_bundle: Bundle | None,
        negative_bundle: Bundle | None,
    ) -> RetrievedCandidate:
        bundle = self._get_profile_bundle(
            candidate.get("expertise") or {}, profile_id=normalize_text(candidate.get("id"))
        )
        base_score = cosine_similarity(user_bundle["vector"], bundle["vector"])
        positive_score = cosine_similarity(positive_bundle["vector"], bundle["vector"]) if positive_bundle else 0.0
        negative_score = cosine_similarity(negative_bundle["vector"], bundle["vector"]) if negative_bundle else 0.0
        vector_score = clamp(
            base_score * BASE_VECTOR_WEIGHT
            + positive_score * POSITIVE_VECTOR_WEIGHT
            - negative_score * NEGATIVE_VECTOR_WEIGHT,
            0.0,
            MAX_SCORE,
        )
        return RetrievedCandidate(
            candidate=candidate,
            features=bundle["features"],
            vector_score=vector_score,
            positive_vector_score=positive_score,
            retrieval_score=retrieval_score(
                self._weights, user.features, bundle["features"], vector_score, user.negative_features
            ),
        )


def _search_reasons(features: dict[str, Any], query_tokens: set[str]) -> list[str]:
    matched_skills = [s for s in features["skills"] if set(tokenize([s])) & query_tokens]
    matched_interests = [i for i in features["interests"] if set(tokenize([i])) & query_tokens]
    reasons = []
    if matched_skills:
        reasons.append(f"Skills: {', '.join(matched_skills[:3])}")
    if matched_interests:
        reasons.append(f"Interests: {', '.join(matched_interests[:2])}")
    if not reasons and features["headline"]:
        reasons.append(f"Profile fit: {features['headline'][:90]}")
    return reasons[:3] or ["Semantically similar profile"]
