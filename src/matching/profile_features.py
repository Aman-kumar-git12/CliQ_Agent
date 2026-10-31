"""Reduces a raw profile to the features used for embedding and scoring (headline, terms, text, tokens)."""

import hashlib
import json
from typing import Any

from .preferences import build_explicit_preference_profile
from .tokens import is_meaningful_text, merge_unique_terms, normalize_text, tokenize

type Features = dict[str, Any]

HEADLINE_KEYS = ("description", "aboutYou", "experience", "name")
# Free-text profile fields in priority order; only the first five meaningful ones reach the embedded text
TEXT_FIELD_KEYS = (
    "description", "aboutYou", "experience", "projects", "achievements", "historySummary",
    "preferenceSummary", "positiveSummary", "negativeSummary", "connectionHeadlines", "likedThemes", "avoidedThemes",
    "explicitPreferenceSummary",
)  # fmt: skip
TRAILING_TEXT_FIELD_KEYS = (
    "activitySummary", "responsivenessSummary", "outcomeSummary", "role", "roles", "industry", "industries",
    "languages", "education", "goals", "lookingFor", "openTo", "availability",
)  # fmt: skip


def _headline(profile: dict[str, Any]) -> str:
    for key in HEADLINE_KEYS:
        if is_meaningful_text(profile.get(key)):
            return normalize_text(profile.get(key))
    return ""


def _text_fields(profile: dict[str, Any], preference_summary: str, details: dict[str, Any]) -> list[str]:
    values = [
        *(profile.get(key) for key in TEXT_FIELD_KEYS),
        preference_summary,
        *(profile.get(key) for key in TRAILING_TEXT_FIELD_KEYS),
        details.get("address"),
    ]
    return [normalize_text(value) for value in values if is_meaningful_text(value)]


def _embedding_text(headline: str, skills: list[str], interests: list[str], text_fields: list[str]) -> str:
    sections = []
    if headline:
        sections.append(f"Headline: {headline}")
    if skills:
        sections.append(f"Skills: {', '.join(skills[:8])}")
    if interests:
        sections.append(f"Interests: {', '.join(interests[:8])}")
    if text_fields:
        sections.append("Profile: " + " | ".join(text_fields[:5]))
    return "\n".join(sections).strip()


def build_profile_features(profile: dict[str, Any]) -> Features:
    explicit_preferences = build_explicit_preference_profile(profile)
    skills = merge_unique_terms(profile.get("skills"), profile.get("likedSkills"), profile.get("acceptedSkills"))
    interests = merge_unique_terms(
        profile.get("interests"), profile.get("likedInterests"), profile.get("acceptedInterests")
    )
    details = profile.get("details") if isinstance(profile.get("details"), dict) else {}
    headline = _headline(profile)
    text_fields = _text_fields(profile, explicit_preferences.get("summary"), details)

    return {
        "headline": headline,
        "skills": skills,
        "interests": interests,
        "text_fields": text_fields,
        "tokens": tokenize([headline, *skills, *interests, *text_fields]),
        # Counts the headline slot even when empty, so every profile scores at least 1
        "completeness": len([headline, *skills, *interests, *text_fields]),
        "text": _embedding_text(headline, skills, interests, text_fields),
        "explicit_preferences": explicit_preferences,
    }


def has_minimum_signal(features: Features, min_completeness: int) -> bool:
    """Whether a profile has enough signal to be shown to others."""
    has_core_signal = (
        bool(features["headline"]) or features["skills"] or features["interests"] or features["text_fields"]
    )
    return bool(has_core_signal) and features["completeness"] >= min_completeness


def profile_signature(profile: dict[str, Any] | None) -> str:
    """Stable hash of the profile content, used to tell whether a cached embedding is stale."""
    encoded = json.dumps(profile or {}, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def build_history_context_profile(history_profile: dict[str, Any] | None, mode: str = "positive") -> dict[str, Any]:
    """Builds a pseudo-profile from connection history: what the user accepted ("positive") or avoided."""
    hp = history_profile or {}
    if mode == "negative":
        return {
            "skills": hp.get("avoidedSkills", []),
            "interests": hp.get("avoidedInterests", []),
            "aboutYou": hp.get("negativeSummary", ""),
            "projects": hp.get("avoidedThemes", []),
            "achievements": hp.get("negativeSummary", ""),
        }
    return {
        "skills": merge_unique_terms(hp.get("acceptedSkills", []), hp.get("likedSkills", [])),
        "interests": merge_unique_terms(hp.get("acceptedInterests", []), hp.get("likedInterests", [])),
        "aboutYou": hp.get("positiveSummary", ""),
        "projects": hp.get("acceptedHeadlines", []),
        "achievements": hp.get("likedThemes", []),
        "historySummary": hp.get("positiveSummary", ""),
    }


def merge_feature_sets(*feature_sets: Features | None) -> Features:
    """Combines the user's own features with their history features; the first headline wins."""
    merged_skills: list[str] = []
    merged_interests: list[str] = []
    merged_text_fields: list[str] = []
    merged_tokens: set[str] = set()
    headline = ""
    for features in feature_sets:
        if not features:
            continue
        if not headline and features.get("headline"):
            headline = features["headline"]
        merged_skills = merge_unique_terms(merged_skills, features.get("skills", []))
        merged_interests = merge_unique_terms(merged_interests, features.get("interests", []))
        merged_text_fields = merge_unique_terms(merged_text_fields, features.get("text_fields", []))
        merged_tokens.update(features.get("tokens", []))
    return {
        "headline": headline,
        "skills": merged_skills,
        "interests": merged_interests,
        "text_fields": merged_text_fields,
        "tokens": sorted(merged_tokens),
        "completeness": len([headline, *merged_skills, *merged_interests, *merged_text_fields]),
    }
