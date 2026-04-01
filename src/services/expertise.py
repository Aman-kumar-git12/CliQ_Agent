import json
import logging
import re
from difflib import get_close_matches

from langchain_core.output_parsers import StrOutputParser

from ..core.llm import response_llm
from ..models.expertise import ExpertiseGenerateResponse, ExpertisePayload
from ..prompts.expertise import GENERATION_PROMPT

logger = logging.getLogger(__name__)

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
    re.compile(r"\bAzure (?:Administrator|Developer|Solutions Architect|Data Engineer)[A-Za-z0-9 .-]*\b", re.IGNORECASE),
    re.compile(r"\bCertified Kubernetes (?:Administrator|Application Developer|Security Specialist)\b", re.IGNORECASE),
    re.compile(r"\b(?:PMP|CSM|PSM I|PSM II|CKA|CKAD|CKS|RHCSA|RHCE)\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9 .-]+ certification\b", re.IGNORECASE),
]

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


def _parse_json_object(response: str):
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in expertise response")
    return json.loads(response[start:end])


def _clean_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _split_terms(*values):
    parts = []
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        parts.extend(re.split(r"[,|\n;/]+", text))
    return [part.strip() for part in parts if part.strip()]


def _canonicalize_skill(term: str):
    cleaned = re.sub(r"\s+", " ", term.strip().lower())
    if not cleaned:
        return ""
    if cleaned in CANONICAL_SKILL_LOOKUP:
        return CANONICAL_SKILL_LOOKUP[cleaned]
    if cleaned in SKILL_ALIASES:
        return SKILL_ALIASES[cleaned]
    match = get_close_matches(cleaned, list(CANONICAL_SKILL_LOOKUP.keys()) + list(SKILL_ALIASES.keys()), n=1, cutoff=0.92)
    if match:
        return CANONICAL_SKILL_LOOKUP.get(match[0]) or SKILL_ALIASES.get(match[0], "")
    return ""


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        normalized = item.lower().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item.strip())
    return result


def _extract_years_phrase(*values):
    pattern = re.compile(r"\b(\d+\+?)\s*(?:years?|yrs?)\b", re.IGNORECASE)
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        match = pattern.search(text)
        if match:
            return f"{match.group(1)} years"
    return ""


def _extract_percentages(*values):
    percentages = []
    pattern = re.compile(r"\b\d+(?:\.\d+)?%\b(?:\s+(?:growth|increase|improvement|reduction|boost|uptime|accuracy|conversion|retention|engagement|efficiency|performance))?", re.IGNORECASE)
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        percentages.extend(match.group(0).strip() for match in pattern.finditer(text))
    return _dedupe_keep_order(percentages)[:5]


def _extract_counts(*values):
    counts = []
    pattern = re.compile(
        r"\b\d+\+?\s+(?:projects?|apps?|applications?|websites?|platforms?|products?|teams?|clients?|users?|engineers?|developers?|features?|certifications?|awards?|campaigns?|dashboards?)\b",
        re.IGNORECASE,
    )
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        counts.extend(match.group(0).strip() for match in pattern.finditer(text))
    return _dedupe_keep_order(counts)[:6]


def _extract_company_names(*values):
    companies = []
    patterns = [
        re.compile(r"\b(?:at|for|with|from|worked at|working at|experience at|built for)\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})"),
        re.compile(r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})\s+(?:team|platform|product|company)\b"),
    ]
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        for pattern in patterns:
            for match in pattern.finditer(text):
                company = match.group(1).strip(" ,.")
                if company and company not in KNOWN_COMPANY_STOPWORDS:
                    companies.append(company)
    return _dedupe_keep_order(companies)[:5]


def _extract_certifications(*values):
    certifications = []
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        for pattern in CERTIFICATION_PATTERNS:
            certifications.extend(match.group(0).strip(" ,.") for match in pattern.finditer(text))
    return _dedupe_keep_order(certifications)[:5]


def _extract_title(role: str, existing_expertise: dict):
    text = _clean_text(role).lower()
    for keywords, title in TITLE_PATTERNS:
        if any(keyword in text for keyword in keywords):
            return title

    existing_title = _clean_text(existing_expertise.get("description", ""))
    if existing_title:
        return existing_title[:50]

    return "Professional @ CliQ"


def _extract_protected_facts(user_info: dict, answers: dict, existing_expertise: dict):
    role = _clean_text(answers.get("role"))
    interests = _clean_text(answers.get("interests"))
    about = _clean_text(answers.get("about"))
    highlights = _clean_text(answers.get("highlights"))
    location = _clean_text(answers.get("location")) or _clean_text(existing_expertise.get("details", {}).get("address"))
    years_phrase = _extract_years_phrase(role, about, highlights, existing_expertise.get("experience", ""))

    raw_skills = _split_terms(
        answers.get("skills"),
        existing_expertise.get("skills", []),
        role,
        highlights,
    )
    canonical_skills = []
    for skill in raw_skills:
        canonical_skill = _canonicalize_skill(skill)
        if canonical_skill:
            canonical_skills.append(canonical_skill)
    canonical_skills = _dedupe_keep_order(canonical_skills)[:8]
    percentages = _extract_percentages(role, about, highlights, existing_expertise.get("projects", ""), existing_expertise.get("achievements", ""))
    counts = _extract_counts(role, about, highlights, existing_expertise.get("projects", ""), existing_expertise.get("achievements", ""))
    company_names = _extract_company_names(role, about, highlights, existing_expertise.get("projects", ""), existing_expertise.get("experience", ""))
    certifications = _extract_certifications(role, about, highlights, existing_expertise.get("achievements", ""))

    return {
        "name": f"{_clean_text(user_info.get('firstname'))} {_clean_text(user_info.get('lastname'))}".strip() or "CliQ User",
        "email": _clean_text(user_info.get("email")),
        "title_hint": _extract_title(role, existing_expertise or {}),
        "years_phrase": years_phrase,
        "location": location or "Remote",
        "skills": canonical_skills,
        "interests": interests,
        "about": about,
        "highlights": highlights,
        "role": role,
        "percentages": percentages,
        "counts": counts,
        "company_names": company_names,
        "certifications": certifications,
        "curated_skill_taxonomy": CANONICAL_SKILLS,
    }


def _normalize_generated_skills(result: dict, protected_facts: dict):
    raw_generated = result.get("skills", [])
    if isinstance(raw_generated, str):
        raw_generated = _split_terms(raw_generated)
    if not isinstance(raw_generated, list):
        raw_generated = []

    generated_skills = []
    for skill in raw_generated:
        canonical_skill = _canonicalize_skill(skill)
        if canonical_skill:
            generated_skills.append(canonical_skill)
    merged = protected_facts["skills"] + generated_skills
    merged = _dedupe_keep_order(merged)

    if not merged:
        merged = ["Communication", "Problem Solving"]

    return merged[:5]


def _append_sentence(base: str, sentence: str):
    sentence = _clean_text(sentence)
    if not sentence:
        return base
    if not base:
        return sentence
    if sentence.lower() in base.lower():
        return base
    connector = "" if base.endswith((".", "!", "?")) else "."
    return f"{base}{connector} {sentence}".strip()


def _ensure_fact_presence(text: str, facts: list[str], prefix: str):
    missing = [fact for fact in facts if fact and fact.lower() not in text.lower()]
    if not missing:
        return text
    return _append_sentence(text, f"{prefix}: {', '.join(missing)}.")


def _validate_and_finalize(result: dict, existing_expertise: dict, answers: dict, protected_facts: dict):
    existing_details = existing_expertise.get("details", {}) if isinstance(existing_expertise, dict) else {}
    description = _clean_text(result.get("description")) or protected_facts["title_hint"]
    description = description[:50] if description else "Professional @ CliQ"

    experience = _clean_text(result.get("experience")) or _clean_text(existing_expertise.get("experience"))
    if protected_facts["years_phrase"]:
        if protected_facts["years_phrase"] not in experience:
            title_part = protected_facts["title_hint"]
            experience = f"{title_part} with {protected_facts['years_phrase']} of experience."
    elif not experience:
        experience = "Building practical work with modern tools and collaborative teams."

    about_you = _clean_text(result.get("aboutYou")) or _clean_text(existing_expertise.get("aboutYou")) or _clean_text(answers.get("about"))
    if not about_you:
        about_you = "A motivated professional focused on building thoughtful, practical work."

    projects = _clean_text(result.get("projects")) or _clean_text(existing_expertise.get("projects")) or _clean_text(answers.get("highlights"))
    achievements = _clean_text(result.get("achievements")) or _clean_text(existing_expertise.get("achievements")) or "Consistent delivery across meaningful work."
    interests = _clean_text(result.get("interests")) or protected_facts.get("interests") or _clean_text(existing_expertise.get("interests")) or "Technology, collaboration, continuous learning"
    experience = _ensure_fact_presence(experience, protected_facts["company_names"], "Worked with")
    projects = _ensure_fact_presence(projects, protected_facts["counts"], "Selected scope")
    achievements = _ensure_fact_presence(achievements, protected_facts["percentages"], "Measured impact")
    achievements = _ensure_fact_presence(achievements, protected_facts["certifications"], "Certifications")

    return {
        "name": protected_facts["name"],
        "description": description,
        "experience": experience,
        "skills": _normalize_generated_skills(result, protected_facts),
        "projects": projects or "Selected work and practical contributions.",
        "achievements": achievements,
        "interests": interests,
        "aboutYou": about_you,
        "details": {
            "email": protected_facts["email"] or _clean_text(existing_details.get("email")),
            "address": protected_facts["location"] or _clean_text(existing_details.get("address")) or "Remote",
        },
        "format": existing_expertise.get("format", 1) if existing_expertise.get("format") in {1, 2, 3, 4} else 1,
    }


def _build_minimal_fallback(user_info: dict, answers: dict, existing_expertise: dict):
    existing_expertise = existing_expertise or {}
    protected_facts = _extract_protected_facts(user_info, answers or {}, existing_expertise)
    fallback = {
        "description": protected_facts["title_hint"],
        "experience": (
            f"{protected_facts['title_hint']} with {protected_facts['years_phrase']} of experience."
            if protected_facts["years_phrase"]
            else "Building practical work with modern tools and collaborative teams."
        ),
        "skills": protected_facts["skills"][:5],
        "projects": _clean_text(answers.get("highlights")) or _clean_text(existing_expertise.get("projects")) or "Selected work and practical contributions.",
        "achievements": _clean_text(existing_expertise.get("achievements")) or "Consistent delivery across meaningful work.",
        "interests": _clean_text(existing_expertise.get("interests")) or "Technology, collaboration, continuous learning",
        "aboutYou": _clean_text(answers.get("about")) or _clean_text(existing_expertise.get("aboutYou")) or "A motivated professional focused on meaningful work.",
    }
    return _validate_and_finalize(fallback, existing_expertise, answers or {}, protected_facts)


class ExpertiseService:
    def __init__(self):
        self.chain = GENERATION_PROMPT | response_llm | StrOutputParser()

    async def generate_expertise(self, user_info: dict, answers: dict, existing_expertise: dict, mode: str = "fresh") -> dict:
        try:
            use_existing = mode == "refine_existing"
            existing_expertise = existing_expertise if use_existing else {}
            protected_facts = _extract_protected_facts(user_info, answers or {}, existing_expertise)
            response = await self.chain.ainvoke({
                "protected_facts": json.dumps(protected_facts, ensure_ascii=True),
                "existing_expertise": json.dumps(existing_expertise, ensure_ascii=True),
            })
            generated = _parse_json_object(response)
            return ExpertiseGenerateResponse.model_validate({
                "expertise": ExpertisePayload.model_validate(
                    _validate_and_finalize(generated, existing_expertise, answers or {}, protected_facts)
                ).model_dump(),
                "generation_source": "llm",
            }).model_dump()
        except Exception as exc:
            logger.exception("Expertise generation fell back to deterministic draft: %s", exc)
            return ExpertiseGenerateResponse.model_validate({
                "expertise": ExpertisePayload.model_validate(
                    _build_minimal_fallback(user_info, answers or {}, existing_expertise or {})
                ).model_dump(),
                "generation_source": "fallback",
            }).model_dump()


expertise_service = ExpertiseService()
