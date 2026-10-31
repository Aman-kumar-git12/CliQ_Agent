"""Skill taxonomy and the text normalization used to map free-text skills onto it."""

import re
from difflib import get_close_matches
from typing import Any

# The only skills a generated profile may list; the prompt also shows this list to the model
CANONICAL_SKILLS = [
    "Communication",
    "Problem Solving",
    "React",
    "Node.js",
    "JavaScript",
    "TypeScript",
    "Next.js",
    "MongoDB",
    "PostgreSQL",
    "Express.js",
    "HTML",
    "CSS",
    "Tailwind CSS",
    "Figma",
    "UI Design",
    "UX Design",
    "UI/UX Design",
    "Python",
    "Java",
    "C++",
    "Prisma",
    "Redis",
    "Socket.IO",
    "AWS",
    "Docker",
    "Git",
    "Go",
    "Kubernetes",
    "GraphQL",
    "FastAPI",
    "CI/CD",
    "Linux",
    "REST APIs",
    "Microservices",
    "System Design",
    "DevOps",
    "Data Analysis",
    "SQL",
    "Firebase",
    "React Native",
]

# Lower-cased spellings users type, mapped to the canonical name
SKILL_ALIASES = {
    "reactjs": "React",
    "react js": "React",
    "react": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "nextjs": "Next.js",
    "next js": "Next.js",
    "mongodb": "MongoDB",
    "mongo db": "MongoDB",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "expressjs": "Express.js",
    "express js": "Express.js",
    "html": "HTML",
    "css": "CSS",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "figma": "Figma",
    "ui ux": "UI/UX Design",
    "ui/ux": "UI/UX Design",
    "ux": "UX Design",
    "ui": "UI Design",
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "prisma": "Prisma",
    "redis": "Redis",
    "socketio": "Socket.IO",
    "socket.io": "Socket.IO",
    "aws": "AWS",
    "docker": "Docker",
    "git": "Git",
    "golang": "Go",
    "go": "Go",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "graphql": "GraphQL",
    "fast api": "FastAPI",
    "fastapi": "FastAPI",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "linux": "Linux",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "api design": "REST APIs",
    "microservices": "Microservices",
    "system design": "System Design",
    "devops": "DevOps",
    "data analysis": "Data Analysis",
    "sql": "SQL",
    "firebase": "Firebase",
    "react native": "React Native",
}

CANONICAL_SKILL_LOOKUP = {skill.lower(): skill for skill in CANONICAL_SKILLS}
# Shown when neither the user nor the model supplied a recognisable skill
DEFAULT_SKILLS = ["Communication", "Problem Solving"]
MAX_PROFILE_SKILLS = 5
# High cutoff: only near-identical typos ("postgress") are corrected, never different skills
SKILL_MATCH_CUTOFF = 0.92
TERM_SEPARATORS = re.compile(r"[,|\n;/]+")


def clean_text(value: Any) -> str:
    """Strips a value to text; lists become a comma-separated string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def split_terms(*values: Any) -> list[str]:
    """Splits free-text lists ("React, Node / SQL") into individual terms."""
    parts: list[str] = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        parts.extend(TERM_SEPARATORS.split(text))
    return [part.strip() for part in parts if part.strip()]


def dedupe_keep_order(items: list[str]) -> list[str]:
    """Drops case-insensitive duplicates and blanks, keeping the first spelling."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.lower().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item.strip())
    return result


def canonicalize_skill(term: str) -> str:
    """Maps a term to its canonical skill name, or "" when it is not in the taxonomy."""
    cleaned = re.sub(r"\s+", " ", term.strip().lower())
    if not cleaned:
        return ""
    if cleaned in CANONICAL_SKILL_LOOKUP:
        return CANONICAL_SKILL_LOOKUP[cleaned]
    if cleaned in SKILL_ALIASES:
        return SKILL_ALIASES[cleaned]
    candidates = list(CANONICAL_SKILL_LOOKUP.keys()) + list(SKILL_ALIASES.keys())
    match = get_close_matches(cleaned, candidates, n=1, cutoff=SKILL_MATCH_CUTOFF)
    if match:
        return CANONICAL_SKILL_LOOKUP.get(match[0]) or SKILL_ALIASES.get(match[0], "")
    return ""


def canonicalize_skills(terms: list[Any]) -> list[str]:
    """Canonicalizes each term and drops the ones outside the taxonomy (order and duplicates kept)."""
    canonical: list[str] = []
    for term in terms:
        skill = canonicalize_skill(term)
        if skill:
            canonical.append(skill)
    return canonical


def normalize_generated_skills(result: dict[str, Any], protected_facts: dict[str, Any]) -> list[str]:
    """Merges the user's skills with the model's, both mapped onto the taxonomy, user's first."""
    raw_generated = result.get("skills", [])
    if isinstance(raw_generated, str):
        raw_generated = split_terms(raw_generated)
    if not isinstance(raw_generated, list):
        raw_generated = []

    merged = dedupe_keep_order(protected_facts["skills"] + canonicalize_skills(raw_generated))
    if not merged:
        merged = list(DEFAULT_SKILLS)
    return merged[:MAX_PROFILE_SKILLS]
