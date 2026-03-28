import re
import math
from typing import Any, Dict, List

PLACEHOLDER_VALUES = {
    "",
    "****",
    "professional seeker",
    "n/a",
    "na",
    "null",
    "undefined",
}

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def is_meaningful_text(value: Any) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return False
    return normalized.lower() not in PLACEHOLDER_VALUES

def normalize_terms(value: Any) -> List[str]:
    raw_items = value if isinstance(value, list) else re.split(r"[,|\n;/]+", normalize_text(value))
    deduped: Dict[str, str] = {}
    for item in raw_items:
        text = normalize_text(item)
        if not is_meaningful_text(text):
            continue
        key = text.lower()
        if key not in deduped:
            deduped[key] = text
    return list(deduped.values())

def merge_unique_terms(*value_groups: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in value_groups:
        for item in normalize_terms(group):
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(item)
    return merged

def get_nested_object(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}

def get_preference_value(profile: Dict[str, Any], *keys: str) -> Any:
    containers = [
        get_nested_object(profile),
        get_nested_object(profile.get("preferences")),
        get_nested_object(profile.get("matchPreferences")),
        get_nested_object(profile.get("connectionPreferences")),
    ]
    for key in keys:
        for container in containers:
            if key in container and container.get(key) is not None:
                return container.get(key)
    return None

def build_explicit_preference_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    looking_for = normalize_terms(get_preference_value(profile, "lookingFor", "openTo", "connectionGoals"))
    preferred_roles = normalize_terms(get_preference_value(profile, "preferredRoles", "roles", "desiredRoles"))
    preferred_industries = normalize_terms(get_preference_value(profile, "preferredIndustries", "industries", "desiredIndustries"))
    preferred_locations = normalize_terms(get_preference_value(profile, "preferredLocations", "locations", "targetLocations"))
    preferred_languages = normalize_terms(get_preference_value(profile, "preferredLanguages", "languages"))
    preferred_experience_levels = normalize_terms(get_preference_value(profile, "preferredExperienceLevels", "experienceLevels"))
    deal_breakers = normalize_terms(get_preference_value(profile, "dealBreakers", "avoid", "avoidTypes"))

    summary_parts = []
    if looking_for: summary_parts.append(f"Looking for: {', '.join(looking_for[:6])}")
    if preferred_roles: summary_parts.append(f"Preferred roles: {', '.join(preferred_roles[:6])}")
    if preferred_industries: summary_parts.append(f"Preferred industries: {', '.join(preferred_industries[:6])}")
    if preferred_locations: summary_parts.append(f"Preferred locations: {', '.join(preferred_locations[:6])}")
    if preferred_languages: summary_parts.append(f"Preferred languages: {', '.join(preferred_languages[:6])}")
    if preferred_experience_levels: summary_parts.append(f"Preferred experience levels: {', '.join(preferred_experience_levels[:6])}")
    if deal_breakers: summary_parts.append(f"Avoid: {', '.join(deal_breakers[:6])}")

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

def tokenize(values: List[str]) -> List[str]:
    tokens = set()
    for value in values:
        text = normalize_text(value).lower()
        for token in re.split(r"[^a-z0-9+#.]+", text):
            cleaned = token.strip()
            if len(cleaned) >= 3:
                tokens.add(cleaned)
    return sorted(tokens)

def shared_items(left: List[str], right: List[str]) -> List[str]:
    right_lookup = {item.lower(): item for item in right}
    return [item for item in left if item.lower() in right_lookup]

def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
