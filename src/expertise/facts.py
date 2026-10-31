"""Extracts the facts (numbers, companies, certifications) the generated profile must preserve exactly."""

import re
from typing import Any

from .normalization import CANONICAL_SKILLS, canonicalize_skills, clean_text, dedupe_keep_order, split_terms

# Capitalised tech names that the company patterns would otherwise mistake for employers
KNOWN_COMPANY_STOPWORDS = {
    "React",
    "Node",
    "Node.js",
    "JavaScript",
    "TypeScript",
    "Python",
    "Java",
    "Docker",
    "AWS",
    "CliQ",
}

CERTIFICATION_PATTERNS = [
    re.compile(r"\bAWS Certified [A-Za-z0-9 .-]+\b", re.IGNORECASE),
    re.compile(r"\bGoogle Cloud Certified [A-Za-z0-9 .-]+\b", re.IGNORECASE),
    re.compile(r"\bMicrosoft Certified: [A-Za-z0-9 .-]+\b", re.IGNORECASE),
    re.compile(
        r"\bAzure (?:Administrator|Developer|Solutions Architect|Data Engineer)[A-Za-z0-9 .-]*\b", re.IGNORECASE
    ),
    re.compile(r"\bCertified Kubernetes (?:Administrator|Application Developer|Security Specialist)\b", re.IGNORECASE),
    re.compile(r"\b(?:PMP|CSM|PSM I|PSM II|CKA|CKAD|CKS|RHCSA|RHCE)\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9 .-]+ certification\b", re.IGNORECASE),
]

# First matching keyword group wins, so more specific roles come before generic ones like "developer"
TITLE_PATTERNS = [
    (("frontend", "front end"), "Frontend Developer"),
    (("backend", "back end"), "Backend Developer"),
    (("full stack", "fullstack"), "Full Stack Developer"),
    (("product designer",), "Product Designer"),
    (("ui/ux", "ui ux", "ux designer"), "UI/UX Designer"),
    (("software engineer",), "Software Engineer"),
    (("developer",), "Software Developer"),
    (("data analyst",), "Data Analyst"),
    (("data scientist",), "Data Scientist"),
    (("devops",), "DevOps Engineer"),
]
DEFAULT_TITLE = "Professional @ CliQ"
MAX_TITLE_LENGTH = 50
MAX_FACT_SKILLS = 8

YEARS_PATTERN = re.compile(r"\b(\d+\+?)\s*(?:years?|yrs?)\b", re.IGNORECASE)
PERCENTAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?%\b(?:\s+(?:growth|increase|improvement|reduction|boost|uptime|accuracy|conversion|retention"
    r"|engagement|efficiency|performance))?",
    re.IGNORECASE,
)
COUNT_PATTERN = re.compile(
    r"\b\d+\+?\s+(?:projects?|apps?|applications?|websites?|platforms?|products?|teams?|clients?|users?|engineers?"
    r"|developers?|features?|certifications?|awards?|campaigns?|dashboards?)\b",
    re.IGNORECASE,
)
COMPANY_PATTERNS = [
    re.compile(
        r"\b(?:at|for|with|from|worked at|working at|experience at|built for)\s+"
        r"([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})"
    ),
    re.compile(r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})\s+(?:team|platform|product|company)\b"),
]


def _non_empty_texts(values: tuple[Any, ...]) -> list[str]:
    return [text for text in (clean_text(value) for value in values) if text]


def extract_years_phrase(*values: Any) -> str:
    """Returns the first "N years" mention, normalised to "N years" (keeping a trailing "+")."""
    for text in _non_empty_texts(values):
        match = YEARS_PATTERN.search(text)
        if match:
            return f"{match.group(1)} years"
    return ""


def extract_percentages(*values: Any) -> list[str]:
    percentages = [m.group(0).strip() for text in _non_empty_texts(values) for m in PERCENTAGE_PATTERN.finditer(text)]
    return dedupe_keep_order(percentages)[:5]


def extract_counts(*values: Any) -> list[str]:
    counts = [m.group(0).strip() for text in _non_empty_texts(values) for m in COUNT_PATTERN.finditer(text)]
    return dedupe_keep_order(counts)[:6]


def extract_company_names(*values: Any) -> list[str]:
    companies: list[str] = []
    for text in _non_empty_texts(values):
        for pattern in COMPANY_PATTERNS:
            for match in pattern.finditer(text):
                company = match.group(1).strip(" ,.")
                if company and company not in KNOWN_COMPANY_STOPWORDS:
                    companies.append(company)
    return dedupe_keep_order(companies)[:5]


def extract_certifications(*values: Any) -> list[str]:
    certifications = [
        m.group(0).strip(" ,.")
        for text in _non_empty_texts(values)
        for pattern in CERTIFICATION_PATTERNS
        for m in pattern.finditer(text)
    ]
    return dedupe_keep_order(certifications)[:5]


def extract_title(role: str, existing_expertise: dict[str, Any]) -> str:
    """Picks a job title from the role answer, else the existing title, else a generic one."""
    text = clean_text(role).lower()
    for keywords, title in TITLE_PATTERNS:
        if any(keyword in text for keyword in keywords):
            return title

    existing_title = clean_text(existing_expertise.get("description", ""))
    if existing_title:
        return existing_title[:MAX_TITLE_LENGTH]
    return DEFAULT_TITLE


def _extract_skills(
    answers: dict[str, Any], existing_expertise: dict[str, Any], role: str, highlights: str
) -> list[str]:
    raw_skills = split_terms(answers.get("skills"), existing_expertise.get("skills", []), role, highlights)
    return dedupe_keep_order(canonicalize_skills(raw_skills))[:MAX_FACT_SKILLS]


def _extract_measurable_facts(role: str, about: str, highlights: str, existing: dict[str, Any]) -> dict[str, list[str]]:
    projects, achievements = existing.get("projects", ""), existing.get("achievements", "")
    return {
        "percentages": extract_percentages(role, about, highlights, projects, achievements),
        "counts": extract_counts(role, about, highlights, projects, achievements),
        "company_names": extract_company_names(role, about, highlights, projects, existing.get("experience", "")),
        "certifications": extract_certifications(role, about, highlights, achievements),
    }


def extract_protected_facts(
    user_info: dict[str, Any], answers: dict[str, Any], existing_expertise: dict[str, Any]
) -> dict[str, Any]:
    """Collects everything the model must keep verbatim, plus the hints it may rephrase."""
    role = clean_text(answers.get("role"))
    interests = clean_text(answers.get("interests"))
    about = clean_text(answers.get("about"))
    highlights = clean_text(answers.get("highlights"))
    location = clean_text(answers.get("location")) or clean_text(existing_expertise.get("details", {}).get("address"))
    full_name = f"{clean_text(user_info.get('firstname'))} {clean_text(user_info.get('lastname'))}".strip()

    return {
        "name": full_name or "CliQ User",
        "email": clean_text(user_info.get("email")),
        "title_hint": extract_title(role, existing_expertise or {}),
        "years_phrase": extract_years_phrase(role, about, highlights, existing_expertise.get("experience", "")),
        "location": location or "Remote",
        "skills": _extract_skills(answers, existing_expertise, role, highlights),
        "interests": interests,
        "about": about,
        "highlights": highlights,
        "role": role,
        **_extract_measurable_facts(role, about, highlights, existing_expertise),
        "curated_skill_taxonomy": CANONICAL_SKILLS,
    }
