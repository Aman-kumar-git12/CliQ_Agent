"""Shared infrastructure: internal key rotation, request ids, log formats and the optional Sentry setup."""

import json
import logging
from collections.abc import Iterator

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from src.core import monitoring
from src.core.config import Settings
from src.core.logging import JsonFormatter
from src.core.request_id import NO_REQUEST_ID, RequestIdFilter, RequestIdMiddleware, current_request_id, request_id_var
from src.core.security import require_internal_key

# --- Internal key and rotation ---------------------------------------------------------------


def _status(key: str | None) -> int:
    try:
        require_internal_key(key)
    except HTTPException as exc:
        return exc.status_code
    return 200


def test_internal_key_accepts_current_and_previous_during_rotation(settings_env) -> None:
    settings_env(AGENT_INTERNAL_KEY="new-key", AGENT_INTERNAL_KEY_PREVIOUS="old-key")
    assert _status("new-key") == 200
    assert _status("old-key") == 200
    assert _status("other") == 401
    assert _status(None) == 401
    assert _status("") == 401


def test_previous_key_stops_working_once_removed(settings_env) -> None:
    settings_env(AGENT_INTERNAL_KEY="new-key", AGENT_INTERNAL_KEY_PREVIOUS="")
    assert _status("new-key") == 200
    assert _status("old-key") == 401
    assert _status("") == 401


def test_previous_key_alone_does_not_open_the_agent(settings_env) -> None:
    settings_env(AGENT_INTERNAL_KEY="", AGENT_INTERNAL_KEY_PREVIOUS="old-key")
    assert _status("old-key") == 503


def test_non_ascii_key_is_rejected_not_a_server_error(settings_env) -> None:
    settings_env(AGENT_INTERNAL_KEY="new-key")
    assert _status("clé") == 401


# --- Request ids ------------------------------------------------------------------------------


@pytest.fixture
def request_id_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/id")
    def echo_id() -> dict[str, str]:
        return {"requestId": current_request_id()}

    @app.get("/stream")
    def stream() -> StreamingResponse:
        def chunks() -> Iterator[str]:
            yield current_request_id()

        return StreamingResponse(chunks(), media_type="text/plain")

    return TestClient(app)


def test_incoming_request_id_is_used_and_echoed(request_id_client: TestClient) -> None:
    response = request_id_client.get("/id", headers={"X-Request-Id": "backend-123"})
    assert response.json() == {"requestId": "backend-123"}
    assert response.headers["x-request-id"] == "backend-123"
    assert current_request_id() == NO_REQUEST_ID  # reset after the request


def test_missing_or_unsafe_request_id_is_replaced(request_id_client: TestClient) -> None:
    generated = request_id_client.get("/id").headers["x-request-id"]
    assert len(generated) == 32
    response = request_id_client.get("/id", headers={"X-Request-Id": "has spaces;and<html>"})
    assert response.headers["x-request-id"] != "has spaces;and<html>"
    assert response.json()["requestId"] == response.headers["x-request-id"]


def test_streaming_responses_carry_the_request_id(request_id_client: TestClient) -> None:
    response = request_id_client.get("/stream", headers={"X-Request-Id": "abc"})
    assert response.headers["x-request-id"] == "abc"
    assert response.text == "abc"


def _record() -> logging.LogRecord:
    return logging.LogRecord("cliq.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)


def test_log_records_get_the_request_id() -> None:
    record = _record()
    token = request_id_var.set("req-1")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)
    assert record.request_id == "req-1"


def test_json_log_format() -> None:
    record = _record()
    RequestIdFilter().filter(record)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "hello world"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "cliq.test"
    assert entry["request_id"] == NO_REQUEST_ID


# --- Sentry -----------------------------------------------------------------------------------


def test_sentry_is_off_without_a_dsn() -> None:
    assert monitoring.setup_sentry(Settings(_env_file=None, sentry_dsn="")) is False


def test_sentry_init_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentry_sdk

    captured: dict = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: captured.update(kwargs))
    settings = Settings(_env_file=None, sentry_dsn="https://key@example.invalid/1", sentry_traces_sample_rate=0.25)
    assert monitoring.setup_sentry(settings) is True
    assert captured["send_default_pii"] is False
    assert captured["traces_sample_rate"] == 0.25
    assert captured["max_request_body_size"] == "never"
    assert "x-internal-key" in captured["event_scrubber"].denylist


def test_sentry_events_are_tagged_with_the_request_id() -> None:
    token = request_id_var.set("req-9")
    try:
        event = monitoring._tag_request_id({}, {})
    finally:
        request_id_var.reset(token)
    assert event["tags"]["request_id"] == "req-9"
    assert "tags" not in monitoring._tag_request_id({}, {})


# --- LLM limits, embeddings cache, database timeouts, keep-alive -------------------------------


def test_groq_calls_are_bounded_and_skip_sdk_retries(settings_env) -> None:
    from src.core.llm import groq_options

    settings_env(LLM_TIMEOUT_SECONDS="20", LLM_CLASSIFIER_TIMEOUT_SECONDS="8", LLM_MAX_RETRIES="0")
    settings_env(LLM_REASONING_EFFORT="low")
    assert groq_options("openai/gpt-oss-120b") == {"timeout": 20, "max_retries": 0, "reasoning_effort": "low"}
    # Classifiers give up sooner; models without reasoning never get the parameter (Groq would reject it)
    assert groq_options("llama-3.1-8b-instant", fast=True) == {"timeout": 8, "max_retries": 0}


def test_reasoning_effort_can_be_turned_off(settings_env) -> None:
    from src.core.llm import groq_options

    settings_env(LLM_REASONING_EFFORT="")
    assert "reasoning_effort" not in groq_options("openai/gpt-oss-20b")


def test_query_embeddings_are_cached_with_a_size_limit() -> None:
    from src.core.embeddings import QueryCachedEmbeddings

    class CountingModel:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def embed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return [float(len(text))]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    model = CountingModel()
    cached = QueryCachedEmbeddings(model, max_entries=2)
    assert cached.embed_query("a") == cached.embed_query("a") == [1.0]
    assert model.queries == ["a"]
    cached.embed_query("bb")
    cached.embed_query("a")  # refreshes "a", so "bb" is now the oldest
    cached.embed_query("ccc")  # evicts "bb"
    assert len(cached._cache) == 2
    cached.embed_query("bb")
    assert model.queries == ["a", "bb", "ccc", "bb"]
    assert cached.embed_documents(["x", "y"]) == [[0.0], [0.0]]


def test_mongo_client_fails_fast_when_atlas_is_unreachable(settings_env) -> None:
    from src.core import database

    settings_env(DATABASE_URL="mongodb://localhost:27017")  # never a mongodb+srv URL: that would query DNS
    database.close_mongo_client()
    client = database.get_mongo_client()  # pymongo connects lazily: nothing is contacted here
    try:
        assert client.options.server_selection_timeout == database.SERVER_SELECTION_TIMEOUT_MS / 1000
        assert client.options.pool_options.connect_timeout == database.CONNECT_TIMEOUT_MS / 1000
        assert client.options.pool_options.socket_timeout == database.SOCKET_TIMEOUT_MS / 1000
    finally:
        database.close_mongo_client()


def test_keepalive_pings_the_public_url_when_render_provides_one() -> None:
    from src.system.keepalive import KeepAlive

    assert KeepAlive(8000, "https://agent.onrender.com/").url == "https://agent.onrender.com/api/health"
    assert KeepAlive(8000).url == "http://localhost:8000/api/health"
