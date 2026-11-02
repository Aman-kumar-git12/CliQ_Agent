"""Tests for matching: tokens, features, scoring, the service (hash embedder, offline) and the routes."""

import asyncio
import hashlib
import sys
import threading
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.config import Settings
from src.core.security import require_internal_key
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


class BlockingBackend:
    """An embedding model whose inference blocks until released, like a long ONNX call."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.thread_ids: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.thread_ids.append(threading.get_ident())
        self.started.set()
        self.release.wait(timeout=2)
        return [[1.0, 0.0] for _ in texts]


async def test_precompute_runs_off_the_event_loop() -> None:
    backend = BlockingBackend()
    service = MatchingService(TextEmbedder(backend), RankingWeights())
    task = asyncio.create_task(service.precompute_profile("p1", USER))
    # Run inline, the embedding would block the loop until the release timeout and the task would be done here
    await asyncio.to_thread(backend.started.wait, 2)
    assert not task.done()
    backend.release.set()
    assert (await task)["profileId"] == "p1"
    assert backend.thread_ids and threading.get_ident() not in backend.thread_ids


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
    app.dependency_overrides[require_internal_key] = lambda: None
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
    app.dependency_overrides[require_internal_key] = lambda: None
    app.state.matching = MatchingService(TextEmbedder(None), RankingWeights())
    client = TestClient(app)
    response = client.post("/api/match/precompute", json={"profileId": "p1", "profile": {"skills": ["React"]}})
    assert response.status_code == 200 and response.json()["profileId"] == "p1"
    # An empty id is the caller's mistake: 422, not a server error
    assert client.post("/api/match/precompute", json={"profileId": "", "profile": {}}).status_code == 422


def test_match_route_validates_body() -> None:
    assert _client(FakeMatchingService()).post("/api/match/match", json={"candidates": []}).status_code == 422


class FakeCollection:
    """Just enough of a pymongo collection for ProfileVectorStore."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def find(self, query: dict, _projection: dict) -> list[dict]:
        return [{"_id": key, **doc} for key, doc in self.docs.items() if doc["model"] == query["model"]]

    def update_one(self, query: dict, update: dict, upsert: bool = False) -> None:
        self.docs[query["_id"]] = {**self.docs.get(query["_id"], {}), **update["$set"]}


class CountingEmbedder(TextEmbedder):
    def __init__(self) -> None:
        super().__init__(None)
        self.calls = 0

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed_text(text)


PROFILE = {"skills": ["React", "Node"], "description": "Full stack developer building web apps"}


def test_stored_embeddings_survive_a_restart() -> None:
    from src.matching.vector_store import ProfileVectorStore

    collection = FakeCollection()
    first = CountingEmbedder()
    MatchingService(first, RankingWeights(), ProfileVectorStore(collection, "hash-96"))._get_profile_bundle(
        PROFILE, profile_id="u1"
    )
    assert first.calls == 1
    assert len(bytes(collection.docs["u1"]["vector"])) == 4 * 96  # packed float32

    # A new process (restart) reads the vector back instead of embedding again
    second = CountingEmbedder()
    bundle = MatchingService(second, RankingWeights(), ProfileVectorStore(collection, "hash-96"))._get_profile_bundle(
        PROFILE, profile_id="u1"
    )
    assert second.calls == 0
    assert bundle["vector"] == pytest.approx(first.embed_text(bundle["features"]["text"]), abs=1e-6)


def test_stored_embeddings_are_redone_when_the_profile_or_model_changes() -> None:
    from src.matching.vector_store import ProfileVectorStore

    collection = FakeCollection()
    MatchingService(
        CountingEmbedder(), RankingWeights(), ProfileVectorStore(collection, "hash-96")
    )._get_profile_bundle(PROFILE, profile_id="u1")

    edited = CountingEmbedder()
    MatchingService(edited, RankingWeights(), ProfileVectorStore(collection, "hash-96"))._get_profile_bundle(
        {**PROFILE, "skills": ["Python"]}, profile_id="u1"
    )
    assert edited.calls == 1

    other_model = CountingEmbedder()
    MatchingService(other_model, RankingWeights(), ProfileVectorStore(collection, "bge"))._get_profile_bundle(
        PROFILE, profile_id="u1"
    )
    assert other_model.calls == 1


def test_matching_still_works_when_the_database_is_down() -> None:
    from src.matching.vector_store import ProfileVectorStore

    class BrokenCollection:
        def find(self, *_args: object) -> list:
            raise ConnectionError("database unreachable")

        def update_one(self, *_args: object, **_kwargs: object) -> None:
            raise ConnectionError("database unreachable")

    embedder = CountingEmbedder()
    service = MatchingService(embedder, RankingWeights(), ProfileVectorStore(BrokenCollection(), "hash-96"))
    assert service._get_profile_bundle(PROFILE, profile_id="u1")["vector"]
    assert embedder.calls == 1


# --- Bounded memory and request limits ------------------------------------------------------------


def test_text_embedder_cache_is_bounded() -> None:
    embedder = CountingEmbedder()
    embedder._max_cached = 2
    for text in ("a", "b", "a", "c"):  # "a" is used again, so "b" is the one evicted
        embedder.embed_text(text)
    assert list(embedder._cache) == [hashlib.sha1(t.encode()).hexdigest() for t in ("a", "c")]


def test_profile_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.matching import service as matching_service

    monkeypatch.setattr(matching_service, "MAX_CACHED_PROFILES", 2)
    service = MatchingService(TextEmbedder(None), RankingWeights())
    for profile_id in ("u1", "u2", "u1", "u3"):
        service._get_profile_bundle(PROFILE, profile_id=profile_id)
    assert list(service._profile_cache) == ["u1", "u3"]


def test_vector_store_retries_a_failed_load_and_keeps_newer_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from array import array

    from src.matching import vector_store
    from src.matching.vector_store import ProfileVectorStore

    collection = FakeCollection()
    ProfileVectorStore(collection, "hash-96").save("stored", "sig", [0.5, 0.25])

    class FlakyCollection(FakeCollection):
        down = True

        def find(self, query: dict, projection: dict) -> list[dict]:
            if self.down:
                raise ConnectionError("database unreachable")
            return super().find(query, projection)

    flaky = FlakyCollection()
    flaky.docs = collection.docs
    store = ProfileVectorStore(flaky, "hash-96")
    store.preload()  # fails: the database is down at startup
    store.save("new", "sig", [1.0, 0.0])
    flaky.down = False
    assert store.get("stored", "sig") is None  # not retried before the retry interval

    monkeypatch.setattr(vector_store.time, "monotonic", lambda: 10**9)
    assert store.get("stored", "sig") == [0.5, 0.25]  # loaded on the next attempt
    assert store.get("new", "sig") == [1.0, 0.0]  # and the vector saved meanwhile survives
    assert isinstance(store._vectors["stored"][1], array)  # float32, not a list of Python floats


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/match/match", {"userProfile": {}, "candidates": [], "topK": -1}),
        ("/api/match/match", {"userProfile": {}, "candidates": [{}] * 5001}),
        ("/api/match/search", {"query": "q" * 501, "candidates": []}),
        ("/api/match/search", {"query": "q", "candidates": [], "minScore": 2}),
        ("/api/match/precompute", {"profileId": "", "profile": {}}),
    ],
)
def test_routes_reject_out_of_range_input(path: str, body: dict[str, Any]) -> None:
    fake = FakeMatchingService()
    assert _client(fake).post(path, json=body).status_code == 422
    assert fake.calls == []
