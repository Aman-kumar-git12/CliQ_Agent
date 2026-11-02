"""Tests for the expertise generator: fact extraction, normalization, drafting, service fallback and the route."""

import json
from typing import Any

import groq
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

from src.core.security import require_internal_key
from src.expertise.dependencies import get_expertise_service
from src.expertise.drafting import build_minimal_fallback, validate_and_finalize
from src.expertise.facts import (
    extract_company_names,
    extract_percentages,
    extract_protected_facts,
    extract_title,
    extract_years_phrase,
)
from src.expertise.normalization import canonicalize_skill, normalize_generated_skills, split_terms
from src.expertise.router import router
from src.expertise.schemas import ExpertiseGenerateRequest
from src.expertise.service import ExpertiseService, parse_json_object

USER = {"firstname": "Asha", "lastname": "Rao", "email": "asha@example.com", "imageUrl": ""}
ANSWERS = {
    "role": "Senior frontend engineer at Razorpay with 6+ yrs",
    "skills": "reactjs, node js, tailwind, Rust",
    "interests": "design systems",
    "about": "I build fast UIs",
    "highlights": "Shipped 12 projects, improved conversion by 35% growth, CKA",
    "location": "Bengaluru",
}


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("reactjs", "React"),
        ("React", "React"),
        ("postgress", "PostgreSQL"),
        ("Rust", ""),
        ("", ""),
        ("ui/ux", "UI/UX Design"),
    ],
)
def test_canonicalize_skill(term: str, expected: str) -> None:
    assert canonicalize_skill(term) == expected


def test_split_terms_handles_all_separators() -> None:
    assert split_terms("a, b | c\nd; e/f", ["g"], None) == ["a", "b", "c", "d", "e", "f", "g"]


def test_extract_protected_facts() -> None:
    facts = extract_protected_facts(USER, ANSWERS, {})
    assert facts["name"] == "Asha Rao"
    assert facts["title_hint"] == "Frontend Developer"
    assert facts["years_phrase"] == "6+ years"
    assert facts["skills"] == ["React", "Node.js", "Tailwind CSS"]
    assert facts["counts"] == ["12 projects"]
    assert facts["company_names"] == ["Razorpay"]
    assert facts["certifications"] == ["CKA"]
    assert facts["location"] == "Bengaluru"


def test_fact_helpers_edge_cases() -> None:
    assert extract_years_phrase("", None, "3 yrs") == "3 years"
    assert extract_company_names("worked at React") == []
    assert extract_title("", {"description": "x" * 60}) == "x" * 50
    assert extract_title("", {}) == "Professional @ CliQ"


def test_validate_and_finalize_restores_dropped_facts() -> None:
    facts = extract_protected_facts(USER, ANSWERS, {})
    generated = {
        "description": "Frontend Dev",
        "experience": "Builds UIs",
        "skills": "Vue, docker",
        "achievements": "Won",
    }
    result = validate_and_finalize(generated, {}, ANSWERS, facts)
    assert result["experience"] == "Frontend Developer with 6+ years of experience. Worked with: Razorpay."
    assert result["skills"] == ["React", "Node.js", "Tailwind CSS", "Docker"]
    # "12 projects" is already in the highlights the projects text came from, so it is not appended twice
    assert result["projects"] == ANSWERS["highlights"]
    # "35% growth" was dropped by the model, so it is restored along with the certification
    assert result["achievements"] == "Won. Measured impact: 35% growth. Certifications: CKA."
    assert result["format"] == 1


def test_normalize_generated_skills_defaults() -> None:
    assert normalize_generated_skills({"skills": 5}, {"skills": []}) == ["Communication", "Problem Solving"]


def test_build_minimal_fallback_keeps_existing_format() -> None:
    existing = {"description": "Data Scientist", "format": 3, "details": {"address": "Berlin"}}
    result = build_minimal_fallback(USER, {}, existing)
    assert result["description"] == "Data Scientist"
    assert result["format"] == 3
    assert result["details"] == {"email": "asha@example.com", "address": "Berlin"}


@pytest.mark.parametrize(("text", "expected"), [('x {"a": 1} y', {"a": 1}), ('```json\n{"k": [1]}\n```', {"k": [1]})])
def test_parse_json_object(text: str, expected: dict[str, Any]) -> None:
    assert parse_json_object(text) == expected


@pytest.mark.parametrize("text", ["no json", "} {"])
def test_parse_json_object_rejects_non_json(text: str) -> None:
    with pytest.raises(ValueError):
        parse_json_object(text)


def _fake_llm(reply: str, prompts: list[str] | None = None) -> RunnableLambda:
    def respond(prompt: PromptValue) -> str:
        if prompts is not None:
            prompts.append(prompt.to_string())
        return reply

    return RunnableLambda(respond)


async def test_service_uses_llm_reply() -> None:
    prompts: list[str] = []
    service = ExpertiseService(_fake_llm(json.dumps({"description": "UI Engineer", "skills": ["css"]}), prompts))
    result = await service.generate_expertise(USER, ANSWERS, {"description": "Old"}, "fresh")
    assert result["generation_source"] == "llm"
    assert result["expertise"]["description"] == "UI Engineer"
    assert "CSS" in result["expertise"]["skills"]
    # A fresh build must not send the old profile to the model
    assert '"Old"' not in prompts[0]


async def test_service_falls_back_on_bad_reply() -> None:
    service = ExpertiseService(_fake_llm("sorry"))
    existing = {"description": "Data Scientist", "format": 2}
    result = await service.generate_expertise(USER, {}, existing, "refine_existing")
    assert result["generation_source"] == "fallback"
    assert result["expertise"]["description"] == "Data Scientist"
    assert result["expertise"]["format"] == 2


class FakeExpertiseService:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, []

    async def generate_expertise(self, *args: Any) -> dict[str, Any] | None:
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.result


def _client(service: FakeExpertiseService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/expertise")
    app.dependency_overrides[require_internal_key] = lambda: None
    app.dependency_overrides[get_expertise_service] = lambda: service
    return TestClient(app)


PAYLOAD = {
    "name": "Asha Rao",
    "description": "Frontend Developer",
    "experience": "6 years",
    "skills": ["React"],
    "projects": "p",
    "achievements": "a",
    "interests": "i",
    "aboutYou": "b",
    "details": {"email": "asha@example.com", "address": "Remote"},
    "format": 1,
}
REQUEST = {"user_info": USER, "answers": ANSWERS}


def test_route_returns_service_result() -> None:
    service = FakeExpertiseService({"expertise": PAYLOAD, "generation_source": "llm"})
    response = _client(service).post("/api/expertise/generate", json={**REQUEST, "existing_expertise": PAYLOAD})
    assert response.status_code == 200
    assert response.json() == {"expertise": PAYLOAD, "generation_source": "llm"}
    user_info, answers, existing, mode = service.calls[0]
    assert existing == PAYLOAD and mode == "fresh" and answers == ANSWERS and user_info == USER


def test_route_passes_partial_existing_as_dict() -> None:
    service = FakeExpertiseService({"expertise": PAYLOAD, "generation_source": "fallback"})
    _client(service).post("/api/expertise/generate", json={**REQUEST, "existing_expertise": {"description": "x"}})
    assert service.calls[0][2] == {"description": "x"}


def test_route_empty_result_is_500() -> None:
    response = _client(FakeExpertiseService(None)).post("/api/expertise/generate", json=REQUEST)
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to generate expertise JSON"}


def test_route_error_is_500_with_message() -> None:
    response = _client(FakeExpertiseService(error=RuntimeError("boom"))).post("/api/expertise/generate", json=REQUEST)
    assert response.status_code == 500
    assert response.json() == {"detail": "boom"}


def test_route_rejects_unknown_fields() -> None:
    response = _client(FakeExpertiseService()).post("/api/expertise/generate", json={**REQUEST, "extra": 1})
    assert response.status_code == 422


def test_percentages_followed_by_words_are_protected() -> None:
    text = "Drove 35% growth, cut costs by 12.5% and kept 99.9% uptime."
    assert extract_percentages(text) == ["35% growth", "12.5%", "99.9% uptime"]


async def test_service_retries_a_provider_5xx_once() -> None:
    calls: list[int] = []

    def respond(_prompt: Any) -> str:
        calls.append(1)
        if len(calls) == 1:
            request = httpx.Request("POST", "https://api.groq.com")
            raise groq.InternalServerError("down", response=httpx.Response(503, request=request), body=None)
        return json.dumps({"description": "UI Engineer", "skills": ["css"]})

    service = ExpertiseService(RunnableLambda(respond))
    result = await service.generate_expertise({"firstname": "A", "email": "a@x.io"}, {"about": "I build UIs"}, {})
    assert result["generation_source"] == "llm"
    assert len(calls) == 2


def test_long_answers_are_shortened_not_rejected() -> None:
    request = ExpertiseGenerateRequest.model_validate({**REQUEST, "answers": {"about": "x" * 10_000}})
    assert len(request.answers.about) == 4000
