import json
import hashlib
from typing import Any, Dict, List, Optional
from threading import Lock
import re

from .config import DEFAULT_TOP_K, DEFAULT_RETRIEVAL_K, RANKING_CONFIG
from .utils import (
    normalize_text, is_meaningful_text, normalize_terms, merge_unique_terms,
    build_explicit_preference_profile, tokenize, shared_items, cosine_similarity, clamp
)
from .embedding import EmbeddingManager

class MatchingService:
    def __init__(self) -> None:
        self._embedding_manager = EmbeddingManager()
        self._profile_embedding_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()

    def _build_profile_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        explicit_preferences = build_explicit_preference_profile(profile)
        skills = merge_unique_terms(profile.get("skills"), profile.get("likedSkills"), profile.get("acceptedSkills"))
        interests = merge_unique_terms(profile.get("interests"), profile.get("likedInterests"), profile.get("acceptedInterests"))
        details = profile.get("details") if isinstance(profile.get("details"), dict) else {}

        headline = ""
        for key in ("description", "aboutYou", "experience", "name"):
            if is_meaningful_text(profile.get(key)):
                headline = normalize_text(profile.get(key))
                break

        text_fields = [
            profile.get("description"), profile.get("aboutYou"), profile.get("experience"),
            profile.get("projects"), profile.get("achievements"), profile.get("historySummary"),
            profile.get("preferenceSummary"), profile.get("positiveSummary"), profile.get("negativeSummary"),
            profile.get("connectionHeadlines"), profile.get("likedThemes"), profile.get("avoidedThemes"),
            profile.get("explicitPreferenceSummary"), explicit_preferences.get("summary"),
            profile.get("activitySummary"), profile.get("responsivenessSummary"), profile.get("outcomeSummary"),
            profile.get("role"), profile.get("roles"), profile.get("industry"), profile.get("industries"),
            profile.get("languages"), profile.get("education"), profile.get("goals"),
            profile.get("lookingFor"), profile.get("openTo"), profile.get("availability"),
            details.get("address"),
        ]
        meaningful_text_fields = [normalize_text(value) for value in text_fields if is_meaningful_text(value)]

        sections = []
        if headline: sections.append(f"Headline: {headline}")
        if skills: sections.append(f"Skills: {', '.join(skills[:8])}")
        if interests: sections.append(f"Interests: {', '.join(interests[:8])}")
        if meaningful_text_fields: sections.append("Profile: " + " | ".join(meaningful_text_fields[:5]))

        tokens = tokenize([headline, *skills, *interests, *meaningful_text_fields])
        completeness = len([headline, *skills, *interests, *meaningful_text_fields])

        return {
            "headline": headline, "skills": skills, "interests": interests,
            "text_fields": meaningful_text_fields, "tokens": tokens,
            "completeness": completeness, "text": "\n".join(sections).strip(),
            "explicit_preferences": explicit_preferences,
        }

    def _detect_profile_type(self, features: Dict[str, Any]) -> str:
        text = " ".join([features.get("headline", ""), *features.get("text_fields", [])]).lower()
        if any(x in text for x in ["designer", "ui/ux", "ux "]): return "design"
        if any(x in text for x in ["manager", "lead", "strategy"]): return "leadership"
        if any(x in text for x in ["data", "analyst", "analytics"]): return "data"
        if any(x in text for x in ["devops", "platform", "cloud"]): return "platform"
        if any(x in text for x in ["developer", "engineer", "software"]): return "engineering"
        if features.get("skills"): return str(features["skills"][0]).lower()
        if features.get("interests"): return str(features["interests"][0]).lower()
        return "generalist"

    def _detect_location_bucket(self, profile: Dict[str, Any], features: Dict[str, Any]) -> str:
        details = profile.get("details", {}) if isinstance(profile.get("details"), dict) else {}
        raw_location = normalize_text(details.get("address") or (features.get("text_fields") or [""])[0])
        if not raw_location: return "location:unknown"
        parts = [normalize_text(part).lower() for part in re.split(r"[,/|-]+", raw_location) if normalize_text(part)]
        return f"location:{parts[-1]}" if parts else "location:unknown"

    def _detect_experience_level(self, profile: Dict[str, Any], features: Dict[str, Any]) -> str:
        source = " ".join([
            normalize_text(profile.get("experienceLevel")), normalize_text(profile.get("yearsOfExperience")),
            normalize_text(profile.get("experience")), features.get("headline", ""), *features.get("text_fields", []),
        ]).lower()
        if not source: return "experience:unknown"
        if "student" in source or "intern" in source: return "experience:student"
        if any(x in source for x in ["senior", "lead", "principal", "10+", "8+"]): return "experience:senior"
        if "junior" in source or "entry" in source: return "experience:junior"
        if any(x in source for x in ["mid", "3+", "4+"]): return "experience:mid"
        return "experience:general"

    def _profile_quality_score(self, features: Dict[str, Any]) -> float:
        quality = (
            (0.32 if features["headline"] else 0.0)
            + min(len(features["skills"]), 5) * 0.09
            + min(len(features["interests"]), 4) * 0.06
            + min(len(features["text_fields"]), 4) * 0.08
        )
        return round(clamp(quality, 0.0, 1.0), 4)

    def _profile_quality_label(self, quality_score: float) -> str:
        if quality_score >= 0.82: return "High-signal profile"
        if quality_score >= 0.62: return "Strong profile"
        if quality_score >= 0.42: return "Moderate profile"
        return "Limited profile signal"

    def _is_discoverable_profile(self, profile: Dict[str, Any]) -> bool:
        features = self._build_profile_features(profile)
        has_core_signal = bool(features["headline"]) or features["skills"] or features["interests"] or features["text_fields"]
        return has_core_signal and features["completeness"] >= RANKING_CONFIG["min_discoverable_completeness"]

    def _profile_signature(self, profile: Dict[str, Any]) -> str:
        encoded = json.dumps(profile or {}, sort_keys=True, default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _get_profile_bundle(self, profile: Dict[str, Any], profile_id: Optional[str] = None, allow_profile_cache: bool = True) -> Dict[str, Any]:
        signature = self._profile_signature(profile)
        if allow_profile_cache and profile_id:
            with self._cache_lock:
                cached_bundle = self._profile_embedding_cache.get(profile_id)
            if cached_bundle and cached_bundle.get("signature") == signature:
                return cached_bundle

        features = self._build_profile_features(profile)
        vector = self._embedding_manager.embed_text(features["text"]) if features["text"] else []
        bundle = {"signature": signature, "features": features, "vector": vector}
        if allow_profile_cache and profile_id:
            with self._cache_lock:
                self._profile_embedding_cache[profile_id] = bundle
        return bundle

    def _build_history_context_profile(self, history_profile: Optional[Dict[str, Any]], mode: str = "positive") -> Dict[str, Any]:
        hp = history_profile or {}
        if mode == "negative":
            return {
                "skills": hp.get("avoidedSkills", []), "interests": hp.get("avoidedInterests", []),
                "aboutYou": hp.get("negativeSummary", ""), "projects": hp.get("avoidedThemes", []),
                "achievements": hp.get("negativeSummary", ""),
            }
        return {
            "skills": merge_unique_terms(hp.get("acceptedSkills", []), hp.get("likedSkills", [])),
            "interests": merge_unique_terms(hp.get("acceptedInterests", []), hp.get("likedInterests", [])),
            "aboutYou": hp.get("positiveSummary", ""), "projects": hp.get("acceptedHeadlines", []),
            "achievements": hp.get("likedThemes", []), "historySummary": hp.get("positiveSummary", ""),
        }

    def _merge_feature_sets(self, *feature_sets: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged_skills, merged_interests, merged_text_fields, merged_tokens = [], [], [], set()
        headline = ""
        for features in feature_sets:
            if not features: continue
            if not headline and features.get("headline"): headline = features["headline"]
            merged_skills = merge_unique_terms(merged_skills, features.get("skills", []))
            merged_interests = merge_unique_terms(merged_interests, features.get("interests", []))
            merged_text_fields = merge_unique_terms(merged_text_fields, features.get("text_fields", []))
            merged_tokens.update(features.get("tokens", []))
        return {
            "headline": headline, "skills": merged_skills, "interests": merged_interests,
            "text_fields": merged_text_fields, "tokens": sorted(merged_tokens),
            "completeness": len([headline, *merged_skills, *merged_interests, *merged_text_fields]),
        }

    def _retrieval_score(self, user_features: Dict[str, Any], candidate_features: Dict[str, Any], vector_score: float, negative_features: Optional[Dict[str, Any]] = None) -> float:
        shared_s = len(shared_items(user_features["skills"], candidate_features["skills"]))
        shared_i = len(shared_items(user_features["interests"], candidate_features["interests"]))
        shared_t = len(shared_items(user_features["tokens"], candidate_features["tokens"]))
        neg_s = len(shared_items((negative_features or {}).get("skills", []), candidate_features["skills"]))
        neg_i = len(shared_items((negative_features or {}).get("interests", []), candidate_features["interests"]))
        neg_t = len(shared_items((negative_features or {}).get("tokens", []), candidate_features["tokens"]))
        pq_score = self._profile_quality_score(candidate_features) * RANKING_CONFIG["profile_quality_weight"]

        return clamp(
            vector_score * RANKING_CONFIG["retrieval_vector_weight"]
            + shared_s * RANKING_CONFIG["retrieval_skill_weight"]
            + shared_i * RANKING_CONFIG["retrieval_interest_weight"]
            + min(shared_t, 5) * RANKING_CONFIG["retrieval_token_weight"]
            - neg_s * 0.06 - neg_i * 0.04 - min(neg_t, 5) * 0.02 + pq_score,
            0.0, 0.99
        )

    def _score_explicit_preferences(self, user_profile: Dict[str, Any], candidate_profile: Dict[str, Any], candidate_features: Dict[str, Any]) -> Dict[str, Any]:
        prefs = build_explicit_preference_profile(user_profile or {})
        if not prefs["has_signal"]: return {"score": 0.0, "reasons": []}

        c_role = self._detect_profile_type(candidate_features)
        c_loc = self._detect_location_bucket(candidate_profile or {}, candidate_features)
        c_exp = self._detect_experience_level(candidate_profile or {}, candidate_features)
        c_inds = normalize_terms([candidate_profile.get("industry"), candidate_profile.get("industries")])
        c_langs = normalize_terms(candidate_profile.get("languages"))
        c_looking = normalize_terms([candidate_profile.get("lookingFor"), candidate_profile.get("openTo"), candidate_profile.get("goals")])
        c_space = " ".join([c_role, c_loc, c_exp, *candidate_features.get("skills", []), *candidate_features.get("interests", []), *c_inds, *c_langs, *c_looking, candidate_features.get("headline", ""), *candidate_features.get("text_fields", [])]).lower()

        score, reasons = 0.0, []
        def _match(items, weight, label):
            nonlocal score
            matches = [item for item in normalize_terms(items) if item.lower() in c_space]
            if matches:
                score += min(len(matches), 2) * weight
                reasons.append(f"{label}: {', '.join(matches[:2])}")

        _match(prefs["looking_for"], 0.045, "Shared intent")
        _match(prefs["preferred_roles"], 0.055, "Preferred role fit")
        _match(prefs["preferred_industries"], 0.05, "Industry fit")
        _match(prefs["preferred_locations"], 0.035, "Location fit")
        _match(prefs["preferred_languages"], 0.03, "Language fit")
        _match(prefs["preferred_experience_levels"], 0.04, "Experience fit")

        db = [item for item in prefs["deal_breakers"] if item.lower() in c_space]
        if db:
            score -= min(len(db), 2) * 0.07
            reasons.append(f"Possible mismatch: {', '.join(db[:2])}")
        return {"score": round(clamp(score, -0.16, 0.18), 4), "reasons": reasons[:2]}

    def _score_candidate_readiness(self, cp: Dict[str, Any]) -> Dict[str, Any]:
        text = " ".join([normalize_text(cp.get(k)) for k in ("activitySummary", "responsivenessSummary", "outcomeSummary")])
        nums = [float(m) / 100.0 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]
        if not nums: return {"score": 0.0, "reasons": []}
        score = round(clamp((sum(nums) / len(nums)) * RANKING_CONFIG["candidate_readiness_weight"], 0.0, 0.18), 4)
        reasons, low = [], text.lower()
        if "response rate" in low or "responsiveness" in low: reasons.append("Responsive candidate")
        if "success score" in low or "acceptance rate" in low: reasons.append("Good connection follow-through")
        if "high activity" in low: reasons.append("Recently active")
        return {"score": score, "reasons": reasons[:2]}

    async def precompute_profile(self, profile_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        if not profile_id: raise ValueError("profile_id is required")
        bundle = self._get_profile_bundle(profile or {}, profile_id=profile_id, allow_profile_cache=True)
        pq_score = self._profile_quality_score(bundle["features"])
        return {
            "profileId": profile_id, "cached": True, "completeness": bundle["features"]["completeness"],
            "profileQualityScore": pq_score, "profileQualityLabel": self._profile_quality_label(pq_score),
            "hasMinimumSignal": self._is_discoverable_profile(profile or {}),
        }

    async def rank_candidates(self, user_profile: Dict[str, Any], history_profile: Optional[Dict[str, Any]], candidates: List[Dict[str, Any]], *, top_k: int = DEFAULT_TOP_K, retrieval_k: int = DEFAULT_RETRIEVAL_K) -> List[Dict[str, Any]]:
        if not candidates: return []
        user_bundle = self._get_profile_bundle(user_profile or {}, allow_profile_cache=False)
        pos_bundle = self._get_profile_bundle(self._build_history_context_profile(history_profile, "positive"), allow_profile_cache=False) if history_profile else None
        neg_bundle = self._get_profile_bundle(self._build_history_context_profile(history_profile, "negative"), allow_profile_cache=False) if history_profile else None
        user_f = self._merge_feature_sets(user_bundle["features"], pos_bundle["features"] if pos_bundle else None)
        neg_f = neg_bundle["features"] if neg_bundle else None

        retrieved = []
        for c in candidates:
            c_id = normalize_text(c.get("id"))
            c_prof = c.get("expertise") or {}
            if not self._is_discoverable_profile(c_prof): continue
            c_bundle = self._get_profile_bundle(c_prof, profile_id=c_id, allow_profile_cache=True)
            b_v_s = cosine_similarity(user_bundle["vector"], c_bundle["vector"])
            p_v_s = cosine_similarity(pos_bundle["vector"], c_bundle["vector"]) if pos_bundle else 0.0
            n_v_s = cosine_similarity(neg_bundle["vector"], c_bundle["vector"]) if neg_bundle else 0.0
            v_s = clamp(b_v_s * 0.64 + p_v_s * 0.28 - n_v_s * 0.22, 0.0, 0.99)
            retrieved.append({"candidate": c, "bundle": c_bundle, "vector_score": v_s, "positive_vector_score": p_v_s, "negative_vector_score": n_v_s,
                              "retrieval_score": self._retrieval_score(user_f, c_bundle["features"], v_s, neg_f)})

        if not retrieved: return []
        retrieved.sort(key=lambda x: x["retrieval_score"], reverse=True)
        shortlisted = retrieved[: max(top_k, retrieval_k)]

        ranked = []
        for item in shortlisted:
            c, features = item["candidate"], item["bundle"]["features"]
            pq_score = self._profile_quality_score(features)
            shared_s = shared_items(user_f["skills"], features["skills"])
            shared_i = shared_items(user_f["interests"], features["interests"])
            shared_t = shared_items(user_f["tokens"], features["tokens"])
            av_s = shared_items(neg_f.get("skills", []) if neg_f else [], features["skills"])
            av_i = shared_items(neg_f.get("interests", []) if neg_f else [], features["interests"])
            av_t = shared_items(neg_f.get("tokens", []) if neg_f else [], features["tokens"])
            ex_p = self._score_explicit_preferences(user_profile or {}, c.get("expertise") or {}, features)
            c_r = self._score_candidate_readiness(c.get("expertise") or {})

            s_overlap = len(shared_s) / max(len(user_f["skills"]), len(features["skills"]), 1)
            i_overlap = len(shared_i) / max(len(user_f["interests"]), len(features["interests"]), 1)
            t_overlap = len(shared_t) / max(len(user_f["tokens"]), len(features["tokens"]), 1)
            p_q_b = pq_score * RANKING_CONFIG["profile_quality_weight"]
            
            score = clamp(item["vector_score"] * RANKING_CONFIG["rank_vector_weight"] + s_overlap * RANKING_CONFIG["rank_skill_weight"] +
                          i_overlap * RANKING_CONFIG["rank_interest_weight"] + t_overlap * RANKING_CONFIG["rank_token_weight"] -
                          (len(av_s) * 0.06 + len(av_i) * 0.04 + min(len(av_t), 5) * 0.02) + p_q_b, 0.0, 0.99)
            score = clamp(score + ex_p["score"] * RANKING_CONFIG["explicit_preference_weight"] + c_r["score"], 0.0, 0.99)

            reasons = []
            if shared_s: reasons.append(f"Shared skills: {', '.join(shared_s[:2])}")
            if shared_i: reasons.append(f"Shared interests: {', '.join(shared_i[:2])}")
            if shared_t: reasons.append(f"Similar profile themes around {', '.join(shared_t[:3])}")
            reasons.extend(ex_p["reasons"])
            reasons.extend(c_r["reasons"])
            if not reasons and pos_bundle and item.get("positive_vector_score", 0) > 0.38: reasons.append("Aligned with the kinds of people you usually connect with")
            if not reasons and features["headline"]: reasons.append(f"Profile fit: {features['headline'][:90]}")
            
            ranked.append({
                "id": c.get("id"), "score": round(score, 4), "reasons": reasons[:3] or ["Recommended based on profile similarity"],
                "profileQualityScore": pq_score, "profileQualityLabel": self._profile_quality_label(pq_score),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
