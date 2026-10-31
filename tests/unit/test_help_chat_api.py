"""HTTP contract of the help chat endpoints the backend calls."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.help_chat.dependencies import get_help_chat_service
from src.help_chat.router import router
from src.help_chat.schemas import ChatResult


class FakeService:
    def __init__(self):
        self.feedback = []
        self.history = []

    async def ask(self, request):
        return ChatResult(answer=f"answer to {request.message}", suggestions=["Next?"])

    async def stream(self, request):
        yield "token", {"text": "Hi"}
        yield "done", {"cacheable": False}

    async def kb_version(self):
        return "abc123"

    async def record_feedback(self, feedback):
        self.feedback.append(feedback)

    async def append_history(self, session_id, question, answer):
        self.history.append((session_id, question, answer))


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def client(service: FakeService, settings_env) -> TestClient:
    settings_env(AGENT_INTERNAL_KEY="secret")
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_help_chat_service] = lambda: service
    return TestClient(app, headers={"X-Internal-Key": "secret"})


FEEDBACK = {"messageId": "m1", "userHash": "u1", "rating": 1, "question": "q", "answer": "a"}


def test_chat_returns_question_answer_and_suggestions(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "hello", "sessionId": "s1"})
    assert response.json() == {"question": "hello", "answer": "answer to hello", "suggestions": ["Next?"]}


def test_chat_rejects_overlong_messages(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": "x" * 1001, "sessionId": "s1"}).status_code == 422


def test_stream_sends_server_sent_events(client: TestClient) -> None:
    response = client.post("/api/chat/stream", json={"message": "hi", "sessionId": "s1"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: token\ndata: {"text": "Hi"}\n\n' in response.text


def test_kb_version(client: TestClient) -> None:
    assert client.get("/api/chat/kb-version").json() == {"kbVersion": "abc123"}


def test_every_endpoint_requires_the_internal_key(client: TestClient, service: FakeService) -> None:
    anonymous = TestClient(client.app)
    assert anonymous.post("/api/chat", json={"message": "hi", "sessionId": "s1"}).status_code == 401
    assert anonymous.post("/api/chat/stream", json={"message": "hi", "sessionId": "s1"}).status_code == 401
    assert anonymous.get("/api/chat/kb-version").status_code == 401
    body = {"sessionId": "s1", "question": "q", "answer": "a"}
    assert anonymous.post("/api/chat/history/append", json=body).status_code == 401
    assert anonymous.post("/api/chat/feedback", json=FEEDBACK).status_code == 401
    assert client.post("/api/chat/feedback", json=FEEDBACK, headers={"X-Internal-Key": "wrong"}).status_code == 401
    assert service.history == []
    ok = client.post("/api/chat/feedback", json=FEEDBACK)
    assert ok.json() == {"ok": True}
    assert service.feedback[0].messageId == "m1"


def test_agent_refuses_calls_without_a_configured_key(client: TestClient, settings_env) -> None:
    settings_env(AGENT_INTERNAL_KEY="")
    assert client.post("/api/chat/feedback", json=FEEDBACK, headers={"X-Internal-Key": "x"}).status_code == 503


def test_history_append(client: TestClient, service: FakeService) -> None:
    body = {"sessionId": "s1", "question": "q", "answer": "a"}
    assert client.post("/api/chat/history/append", json=body).json() == {"ok": True}
    assert service.history == [("s1", "q", "a")]
