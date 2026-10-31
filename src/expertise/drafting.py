"""Turns a model reply (or nothing, on failure) into a complete expertise profile that keeps the protected facts."""

from typing import Any

from .facts import DEFAULT_TITLE, MAX_TITLE_LENGTH, extract_protected_facts
from .normalization import clean_text, normalize_generated_skills

DEFAULT_EXPERIENCE = "Building practical work with modern tools and collaborative teams."
DEFAULT_PROJECTS = "Selected work and practical contributions."
DEFAULT_ACHIEVEMENTS = "Consistent delivery across meaningful work."
DEFAULT_INTERESTS = "Technology, collaboration, continuous learning"
DEFAULT_ABOUT = "A motivated professional focused on building thoughtful, practical work."
FALLBACK_ABOUT = "A motivated professional focused on meaningful work."
# Profile layouts the frontend can render; anything else falls back to layout 1
VALID_FORMATS = {1, 2, 3, 4}


def _append_sentence(base: str, sentence: str) -> str:
    sentence = clean_text(sentence)
    if not sentence:
        return base
    if not base:
        return sentence
    if sentence.lower() in base.lower():
        return base
    connector = "" if base.endswith((".", "!", "?")) else "."
    return f"{base}{connector} {sentence}".strip()


def _ensure_fact_presence(text: str, facts: list[str], prefix: str) -> str:
    """Appends any protected fact the model dropped, so numbers and names are never lost."""
    missing = [fact for fact in facts if fact and fact.lower() not in text.lower()]
    if not missing:
        return text
    return _append_sentence(text, f"{prefix}: {', '.join(missing)}.")


def _experience_with_years(result: dict[str, Any], existing: dict[str, Any], facts: dict[str, Any]) -> str:
    experience = clean_text(result.get("experience")) or clean_text(existing.get("experience"))
    if facts["years_phrase"]:
        # The model sometimes rounds or inflates years; the user's exact phrase must win
        if facts["years_phrase"] not in experience:
            experience = f"{facts['title_hint']} with {facts['years_phrase']} of experience."
    elif not experience:
        experience = DEFAULT_EXPERIENCE
    return experience


def _first_text(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def validate_and_finalize(
    result: dict[str, Any], existing_expertise: dict[str, Any], answers: dict[str, Any], protected_facts: dict[str, Any]
) -> dict[str, Any]:
    """Fills gaps from the existing profile and answers, then re-inserts any protected fact the model dropped."""
    existing = existing_expertise
    existing_details = existing.get("details", {}) if isinstance(existing, dict) else {}
    description = clean_text(result.get("description")) or protected_facts["title_hint"]
    description = description[:MAX_TITLE_LENGTH] if description else DEFAULT_TITLE

    experience = _experience_with_years(result, existing, protected_facts)
    about_you = _first_text(result.get("aboutYou"), existing.get("aboutYou"), answers.get("about")) or DEFAULT_ABOUT
    projects = _first_text(result.get("projects"), existing.get("projects"), answers.get("highlights"))
    achievements = _first_text(result.get("achievements"), existing.get("achievements")) or DEFAULT_ACHIEVEMENTS
    interests = (
        clean_text(result.get("interests"))
        or protected_facts.get("interests")
        or clean_text(existing.get("interests"))
        or DEFAULT_INTERESTS
    )
    experience = _ensure_fact_presence(experience, protected_facts["company_names"], "Worked with")
    projects = _ensure_fact_presence(projects, protected_facts["counts"], "Selected scope")
    achievements = _ensure_fact_presence(achievements, protected_facts["percentages"], "Measured impact")
    achievements = _ensure_fact_presence(achievements, protected_facts["certifications"], "Certifications")

    return {
        "name": protected_facts["name"],
        "description": description,
        "experience": experience,
        "skills": normalize_generated_skills(result, protected_facts),
        "projects": projects or DEFAULT_PROJECTS,
        "achievements": achievements,
        "interests": interests,
        "aboutYou": about_you,
        "details": {
            "email": protected_facts["email"] or clean_text(existing_details.get("email")),
            "address": protected_facts["location"] or clean_text(existing_details.get("address")) or "Remote",
        },
        "format": existing.get("format", 1) if existing.get("format") in VALID_FORMATS else 1,
    }


def build_minimal_fallback(
    user_info: dict[str, Any], answers: dict[str, Any], existing_expertise: dict[str, Any]
) -> dict[str, Any]:
    """A deterministic profile built without the model, used when generation fails."""
    existing_expertise = existing_expertise or {}
    answers = answers or {}
    protected_facts = extract_protected_facts(user_info, answers, existing_expertise)
    years_phrase = protected_facts["years_phrase"]
    fallback = {
        "description": protected_facts["title_hint"],
        "experience": (
            f"{protected_facts['title_hint']} with {years_phrase} of experience."
            if years_phrase
            else DEFAULT_EXPERIENCE
        ),
        "skills": protected_facts["skills"][:5],
        "projects": _first_text(answers.get("highlights"), existing_expertise.get("projects")) or DEFAULT_PROJECTS,
        "achievements": clean_text(existing_expertise.get("achievements")) or DEFAULT_ACHIEVEMENTS,
        "interests": clean_text(existing_expertise.get("interests")) or DEFAULT_INTERESTS,
        "aboutYou": _first_text(answers.get("about"), existing_expertise.get("aboutYou")) or FALLBACK_ABOUT,
    }
    return validate_and_finalize(fallback, existing_expertise, answers, protected_facts)
