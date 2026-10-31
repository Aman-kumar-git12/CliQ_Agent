"""Tests for matching: tokens, features, scoring, the service (hash embedder, offline) and the routes."""

import sys
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.config import Settings
from src.matching import embedding
from src.matching.dependencies import get_matching_service
from src.matching.embedding import TextEmbedder, cosine_similarity, hash_embedding
from src.matching.preferences import build_explicit_preference_profile, detect_location_bucket
from src.matching.profile_features import build_profile_features, has_minimum_signal
from src.matching.router import router
from src.matching.scoring import RankingWeights, profile_quality_label, profile_quality_score, score_candidate_readiness
from src.matching.service import MatchingService
from src.matching.tokens import merge_unique_terms, normalize_terms, shared_items, tokenize

CANDIDATES = [
    {
        "id": "c1",
        "expertise": {"description": "Frontend Developer", "skills": ["React", "TypeScript"], "interests": ["design"]},
    },
    {"id": "c2", "expertise": {"description": "Data Analyst", "skills": "SQL, Python", "interests": "analytics"}},
    {"id": "c3", "expertise": {"description": "N/A", "skills": []}},
    {"id": "c4", "expertise": {"description": "Designer", "skills": ["Figma"], "activitySummary": "Response rate 90%"}},
    {"id": "", "expertise": {"description": "No id", "skills": ["React"]}},
]
USER = {"description": "Full Stack Developer", "skills": ["React", "Python"], "interests": ["design"]}


@pytest.fixture
def service() -> MatchingService:
    return MatchingService(TextEmbedder(None), RankingWeights())


def test_term_helpers() -> None:
    assert normalize_terms("React, react | null ; Node") == ["React", "Node"]
    assert merge_unique_terms(["a", "B"], "b, c") == ["a", "B", "c"]
    assert tokenize(["Node.js and C++", "go"]) == ["and", "c++", "node.js"]
    assert shared_items(["React", "SQL"], ["react"]) == ["React"]


def test_hash_embedding_is_normalised_and_deterministic() -> None:
    vector = hash_embedding("react developer")
    assert len(vector) == 96
    assert vector == hash_embedding("react developer")
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert cosine_similarity([], vector) == 0.0


def test_load_embedding_backend_falls_back_to_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise OSError("model download failed")

    monkeypatch.setattr(embedding, "get_embeddings", broken)
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `from fastembed import ...` raise ImportError
    assert embedding.load_embedding_backend() is None


def test_text_embedder_caches_and_prefixes_queries_only_for_models() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.texts: list[str] = []

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.texts.extend(texts)
            return [[1.0, 0.0]]

    model = FakeModel()
    embedder = TextEmbedder(model)
    embedder.embed_text("x")
    embedder.embed_text("x")
    embedder.embed_query("q")
    assert model.texts == ["x", "Represent this sentence for searching relevant passages: q"]
    assert TextEmbedder(None).embed_query("q") == hash_embedding("q")


def test_profile_features_and_quality() -> None:
    features = build_profile_features({**USER, "details": {"address": "Pune, India"}})
    assert features["headline"] == "Full Stack Developer"
    assert features["text"].startswith("Headline: Full Stack Developer\nSkills: React, Python")
    assert profile_quality_score(features) == 0.72  # headline 0.32 + 2 skills + 1 interest + 2 text fields
    assert profile_quality_label(0.9) == "High-signal profile"
    assert detect_location_bucket({"details": {"address": "Pune, India"}}, features) == "location:india"


def test_has_minimum_signal_keeps_legacy_empty_list() -> None:
    assert has_minimum_signal(build_profile_features(USER), 1) is True
    assert has_minimum_signal(build_profile_features({}), 1) is False


def test_explicit_preferences_read_nested_containers() -> None:
    prefs = build_explicit_preference_profile({"lookingFor": "cofounder", "preferences": {"dealBreakers": ["student"]}})
    assert prefs["looking_for"] == ["cofounder"]
    assert prefs["deal_breakers"] == ["student"]
    assert prefs["summary"] == "Looking for: cofounder. Avoid: student"


def test_candidate_readiness() -> None:
    result = score_candidate_readiness(RankingWeights(), {"activitySummary": "Response rate 50%"})
    assert result == {"score": 0.06, "reasons": ["Responsive candidate"]}


def test_ranking_weights_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMART_RANK_SKILL_WEIGHT", "0.5")
    weights = RankingWeights.from_settings(Settings(_env_file=None))
    assert weights.rank_skill_weight == 0.5
    assert weights.rank_vector_weight == RankingWeights().rank_vector_weight


async def test_rank_candidates(service: MatchingService) -> None:
    ranked = await service.rank_candidates(USER, None, CANDIDATES, top_k=10, retrieval_k=10)
    ids = [item["id"] for item in ranked]
    assert "c3" not in ids  # placeholder-only profile is not discoverable
    assert ids[0] == "c1"
    assert ranked[0]["reasons"][0] == "Shared skills: React"
    assert set(ranked[0]) == {"id", "score", "reasons", "profileQualityScore", "profileQualityLabel"}
    assert await service.rank_candidates(USER, None, []) == []
    assert len(await service.rank_candidates(USER, None, CANDIDATES, top_k=2, retrieval_k=0)) == 2


async def test_rank_candidates_penalises_avoided_history(service: MatchingService) -> None:
    history = {"avoidedSkills": ["TypeScript"], "acceptedSkills": ["SQL"]}
    plain = {r["id"]: r["score"] for r in await service.rank_candidates(USER, None, CANDIDATES)}
    with_history = {r["id"]: r["score"] for r in await service.rank_candidates(USER, history, CANDIDATES)}
    assert with_history["c1"] < plain["c1"]
    assert with_history["c2"] > plain["c2"]


async def test_semantic_search(service: MatchingService) -> None:
    results = await service.semantic_search("react typescript", CANDIDATES, min_score=0.0)
    assert results[0]["id"] == "c1"
    assert results[0]["reasons"] == ["Skills: React, TypeScript"]
    assert all(r["id"] for r in results)  # candidates without an id are skipped
    assert await service.semantic_search("  ", CANDIDATES) == []
    assert len(await service.semantic_search("react", CANDIDATES, top_k=1, min_score=-1.0)) == 1


async def test_precompute(service: MatchingService) -> None:
    result = await service.precompute_profile("p1", USER)
    assert result == {
        "profileId": "p1",
        "cached": True,
        "completeness": 5,
        "profileQualityScore": 0.64,
        "profileQualityLabel": "Strong profile",
        "hasMinimumSignal": True,
    }
    with pytest.raises(ValueError, match="profile_id is required"):
        await service.precompute_profile("", {})


class FakeMatchingService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((name, args, kwargs))
        if self.error:
            raise self.error

    async def rank_candidates(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("rank", args, kwargs)
        return [{"id": "c1", "score": 0.5}]

    async def precompute_profile(self, *args: Any) -> dict[str, Any]:
        self._record("precompute", args, {})
        return {"profileId": args[0], "cached": True}

    async def semantic_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("search", args, kwargs)
        return [{"id": "c2", "score": 0.7, "reasons": []}]


def _client(service: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/match")
    app.dependency_overrides[get_matching_service] = lambda: service
    return TestClient(app)


def test_match_route_applies_defaults() -> None:
    fake = FakeMatchingService()
    response = _client(fake).post("/api/match/match", json={"userProfile": USER, "candidates": [], "topK": 0})
    assert response.status_code == 200
    assert response.json() == {"rankedResults": [{"id": "c1", "score": 0.5}]}
    assert fake.calls[0] == ("rank", (USER, None, []), {"top_k": 60, "retrieval_k": 180})


def test_precompute_route() -> None:
    fake = FakeMatchingService()
    response = _client(fake).post("/api/match/precompute", json={"profileId": "p1", "profile": {}})
    assert response.json() == {"profileId": "p1", "cached": True}


def test_search_route_applies_defaults() -> None:
    fake = FakeMatchingService()
    response = _client(fake).post("/api/match/search", json={"query": "react", "candidates": []})
    assert response.json() == {"results": [{"id": "c2", "score": 0.7, "reasons": []}]}
    assert fake.calls[0] == ("search", ("react", []), {"top_k": 20, "min_score": None})


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/match/match", {"userProfile": {}, "candidates": []}),
        ("/api/match/precompute", {"profileId": "p", "profile": {}}),
        ("/api/match/search", {"query": "q", "candidates": []}),
    ],
)
def test_routes_turn_errors_into_500(path: str, body: dict[str, Any]) -> None:
    response = _client(FakeMatchingService(error=ValueError("bad input"))).post(path, json=body)
    assert response.status_code == 500
    assert response.json() == {"detail": "bad input"}


def test_routes_use_app_state_service() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/match")
    app.state.matching = MatchingService(TextEmbedder(None), RankingWeights())
    response = TestClient(app).post("/api/match/precompute", json={"profileId": "", "profile": {}})
    assert response.status_code == 500
    assert response.json() == {"detail": "profile_id is required"}


def test_match_route_validates_body() -> None:
    assert _client(FakeMatchingService()).post("/api/match/match", json={"candidates": []}).status_code == 422
