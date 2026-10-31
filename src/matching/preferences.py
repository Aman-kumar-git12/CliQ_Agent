"""Explicit connection preferences and the coarse candidate traits (role, location, seniority) they are matched on."""

import re
from typing import Any

from .tokens import normalize_terms, normalize_text

# Preferences may sit on the profile itself or in any of these nested objects, depending on the client version
PREFERENCE_CONTAINERS = ("preferences", "matchPreferences", "connectionPreferences")

# (result key, summary label, accepted source keys in priority order)
PREFERENCE_FIELDS = [
    ("looking_for", "Looking for", ("lookingFor", "openTo", "connectionGoals")),
    ("preferred_roles", "Preferred roles", ("preferredRoles", "roles", "desiredRoles")),
    ("preferred_industries", "Preferred industries", ("preferredIndustries", "industries", "desiredIndustries")),
    ("preferred_locations", "Preferred locations", ("preferredLocations", "locations", "targetLocations")),
    ("preferred_languages", "Preferred languages", ("preferredLanguages", "languages")),
    ("preferred_experience_levels", "Preferred experience levels", ("preferredExperienceLevels", "experienceLevels")),
    ("deal_breakers", "Avoid", ("dealBreakers", "avoid", "avoidTypes")),
]

# Checked in order; the first group with a keyword in the profile text decides the type
PROFILE_TYPE_KEYWORDS = [
    ("design", ["designer", "ui/ux", "ux "]),
    ("leadership", ["manager", "lead", "strategy"]),
    ("data", ["data", "analyst", "analytics"]),
    ("platform", ["devops", "platform", "cloud"]),
    ("engineering", ["developer", "engineer", "software"]),
]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_preference_value(profile: dict[str, Any], *keys: str) -> Any:
    """First non-None value among `keys`, looking at the profile before its nested preference objects."""
    containers = [_as_dict(profile), *(_as_dict(profile.get(name)) for name in PREFERENCE_CONTAINERS)]
    for key in keys:
        for container in containers:
            if key in container and container.get(key) is not None:
                return container.get(key)
    return None


def build_explicit_preference_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Collects the stated preferences plus a one-line summary that is also embedded with the profile."""
    result: dict[str, Any] = {}
    summary_parts: list[str] = []
    for key, label, source_keys in PREFERENCE_FIELDS:
        terms = normalize_terms(get_preference_value(profile, *source_keys))
        result[key] = terms
        if terms:
            summary_parts.append(f"{label}: {', '.join(terms[:6])}")
    result["summary"] = ". ".join(summary_parts)
    result["has_signal"] = len(summary_parts) > 0
    return result


def detect_profile_type(features: dict[str, Any]) -> str:
    text = " ".join([features.get("headline", ""), *features.get("text_fields", [])]).lower()
    for profile_type, keywords in PROFILE_TYPE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return profile_type
    if features.get("skills"):
        return str(features["skills"][0]).lower()
    if features.get("interests"):
        return str(features["interests"][0]).lower()
    return "generalist"


def detect_location_bucket(profile: dict[str, Any], features: dict[str, Any]) -> str:
    """Uses the last part of the address ("Pune, India" -> "india") as a coarse location."""
    details = profile.get("details", {}) if isinstance(profile.get("details"), dict) else {}
    raw_location = normalize_text(details.get("address") or (features.get("text_fields") or [""])[0])
    if not raw_location:
        return "location:unknown"
    parts = [normalize_text(part).lower() for part in re.split(r"[,/|-]+", raw_location) if normalize_text(part)]
    return f"location:{parts[-1]}" if parts else "location:unknown"


def detect_experience_level(profile: dict[str, Any], features: dict[str, Any]) -> str:
    source = " ".join(
        [
            normalize_text(profile.get("experienceLevel")),
            normalize_text(profile.get("yearsOfExperience")),
            normalize_text(profile.get("experience")),
            features.get("headline", ""),
            *features.get("text_fields", []),
        ]
    ).lower()
    if not source:
        return "experience:unknown"
    if "student" in source or "intern" in source:
        return "experience:student"
    if any(x in source for x in ["senior", "lead", "principal", "10+", "8+"]):
        return "experience:senior"
    if "junior" in source or "entry" in source:
        return "experience:junior"
    if any(x in source for x in ["mid", "3+", "4+"]):
        return "experience:mid"
    return "experience:general"
