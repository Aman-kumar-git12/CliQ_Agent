"""Unit tests for the message assistant's deterministic logic, service fallbacks and HTTP routes."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.core.security import require_internal_key
from src.message_assistant import canned_responses
from src.message_assistant.conversation import (
    build_assistant_history_section,
    build_context_sections,
    context_confidence,
    detect_intent_fallback,
    normalize_turn,
)
from src.message_assistant.dependencies import get_message_assistant_service
from src.message_assistant.json_output import ensure_list, extract_json_payload
from src.message_assistant.router import router
from src.message_assistant.routing import (
    is_capability_prompt,
    is_contextual_chat_question,
    is_greeting_prompt,
    is_transcript_or_summary_prompt,
    should_use_general_reply_mode,
)
from src.message_assistant.safety import contains_unsafe_messaging_request
from src.message_assistant.schemas import MessageAssistantRequest
from src.message_assistant.service import MessageAssistantService
from src.message_assistant.suggestions import normalize_grouped_replies, normalize_reply_payload
from src.message_assistant.tone import get_emoji_instruction, normalize_tone

Responder = Callable[[str, str], str]


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Stands in for ChatGroq; `responder(system_prompt, user_prompt)` returns text or raises."""

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.calls: list[str] = []

    def invoke(self, messages: list[Any]) -> _Message:
        self.calls.append(messages[0].content)
        return _Message(self.responder(messages[0].content, messages[1].content))

    async def astream(self, messages: list[Any]) -> AsyncIterator[_Message]:
        self.calls.append(messages[0].content)
        for part in self.responder(messages[0].content, messages[1].content).split("|"):
            if part == "RAISE":
                raise RuntimeError("stream broke")
            yield _Message(part)


def _fail(_system: str, _user: str) -> str:
    raise RuntimeError("model down")


def _service(responder: Responder) -> tuple[MessageAssistantService, list[FakeLLM]]:
    llms = [FakeLLM(responder) for _ in range(3)]
    return MessageAssistantService(*llms), llms


CHAT = [
    {"role": "other", "text": "Hey! How was your weekend?", "kind": "text"},
    {"role": "me", "text": "Great, went hiking", "kind": "text"},
    {"role": "other", "text": "Want to grab coffee sometime", "kind": "text"},
    {"role": "me", "text": "sure", "kind": "text"},
]


# --- JSON extraction -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 2}\n```', {"a": 2}),
        ('Sure! {"x": "y"}', {"x": "y"}),
        # An object followed by more prose is skipped in favour of one that ends the reply.
        ('pre {"a": {"b": 1}} and {"c": 2}', {"c": 2}),
        ('{"s": "}"} trailing words', {"s": "}"}),
    ],
)
def test_extract_json_payload_finds_object(raw: str, expected: dict[str, Any]) -> None:
    assert extract_json_payload(raw) == expected


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2]", '{"broken": '])
def test_extract_json_payload_raises_without_object(raw: str) -> None:
    with pytest.raises(ValueError):
        extract_json_payload(raw)


def test_ensure_list_strips_dedupes_and_caps() -> None:
    assert ensure_list(["a", "a", " b ", None, "", "c", "d"], 3) == ["a", "b", "c"]
    assert ensure_list("not a list", 3) == []


# --- Conversation formatting -----------------------------------------------------------------


def test_normalize_turn_uses_placeholders_for_media() -> None:
    assert normalize_turn({"role": "me", "text": " hi "}) == "You: hi"
    assert normalize_turn({"role": "other", "text": "", "kind": "image"}) == "Other person: [Image]"
    assert normalize_turn({"role": "other", "text": None, "kind": "VOICE"}) == "Other person: [Voice message]"
    assert normalize_turn({"role": "other", "text": "", "kind": None}) == "Other person: [Empty]"


def test_build_context_sections_keeps_last_twenty_turns() -> None:
    turns = [{"role": "me", "text": f"m{i}"} for i in range(30)]
    conversation_text, older_text = build_context_sections(turns, "  ")
    lines = conversation_text.splitlines()
    assert len(lines) == 20 and lines[0] == "You: m10" and lines[-1] == "You: m29"
    assert older_text == "No older context available."
    assert build_context_sections(None, " old ") == ("No prior messages.", "old")


def test_assistant_history_keeps_last_twelve_non_empty() -> None:
    history = [{"role": "user" if i % 2 else "assistant", "text": f"t{i}"} for i in range(20)]
    assert build_assistant_history_section(history).splitlines()[0] == "Assistant: t8"
    assert build_assistant_history_section([{"role": "user", "text": " "}]) == "No prior Ask AI follow-up context."


def test_context_confidence_levels() -> None:
    long_last = CHAT[:2] + [{"role": "other", "text": "Want to grab coffee sometime?"}, CHAT[3]]
    assert context_confidence(long_last) == "high"
    assert context_confidence(CHAT[:2]) == "medium"
    assert context_confidence([], "older") == "medium"
    assert context_confidence([]) == "low"


@pytest.mark.parametrize(
    ("turns", "intent"),
    [
        ([], "opener"),
        ([{"role": "other", "text": ""}], "general"),
        ([{"role": "other", "text": "how are you?"}], "question"),
        ([{"role": "other", "text": "dinner on friday"}], "invitation"),
        ([{"role": "other", "text": "so sorry"}], "apology"),
        ([{"role": "other", "text": "cool"}], "casual"),
    ],
)
def test_detect_intent_fallback(turns: list[dict[str, Any]], intent: str) -> None:
    assert detect_intent_fallback(turns) == intent


# --- Routing heuristics, tone and safety ------------------------------------------------------


def test_prompt_recognisers() -> None:
    assert is_greeting_prompt("  Good   Morning ")
    assert not is_greeting_prompt("hi there")
    assert is_capability_prompt("What can you do?")
    assert is_transcript_or_summary_prompt("please summarize this chat")
    assert not is_transcript_or_summary_prompt("")


@pytest.mark.parametrize(
    ("question", "contextual"),
    [("", True), ("what did she say", True), ("how to text a girl", False), ("flirty lines", False), ("x", True)],
)
def test_is_contextual_chat_question(question: str, contextual: bool) -> None:
    assert is_contextual_chat_question(question) is contextual


def test_should_use_general_reply_mode() -> None:
    assert should_use_general_reply_mode([], "") is True
    assert should_use_general_reply_mode(CHAT, "hi") is False
    assert should_use_general_reply_mode([{"role": "me", "text": "a"}, {"role": "me", "text": "b"}], "start chat")
    assert should_use_general_reply_mode([{"role": "me", "text": "a"}, {"role": "me", "text": "b"}], "ok") is False


def test_tone_and_emoji_normalisation() -> None:
    assert normalize_tone(" WITTY ") == "witty"
    assert normalize_tone("rude") == "polite"
    assert normalize_tone(None) == "polite"
    assert get_emoji_instruction("odd").startswith("Emoji instruction for 'odd': Generate two parallel styles")


def test_contains_unsafe_messaging_request() -> None:
    assert contains_unsafe_messaging_request("help me", "BLACKMAIL him")
    assert not contains_unsafe_messaging_request("", "say hi nicely")


# --- Suggestion normalisation ----------------------------------------------------------------


def test_normalize_grouped_replies_falls_back_to_rewrites_and_draft() -> None:
    grouped = normalize_grouped_replies({"follow_up": "fu"}, {"warm": "w", "confident": "c"}, {"top_reply": "t"}, "d")
    assert grouped["top_reply"] == "t"
    assert grouped["safe_reply"] == "d"
    assert grouped["warm_reply"] == "w"
    assert grouped["playful_reply"] == grouped["curious_reply"] == "fu"
    assert grouped["direct_reply"] == grouped["confident_reply"] == "c"
    assert grouped["short_reply"] == grouped["opener"] == "d"


def _normalize(payload: dict[str, Any], emoji_preference: str | None = "both") -> dict[str, Any] | None:
    return normalize_reply_payload(
        payload,
        draft="",
        tone="polite",
        max_suggestions=2,
        emoji_preference=emoji_preference,
        fallback_intent="opener",
        generation_source="response_llm",
        routing_source="general_mode",
    )


def test_normalize_reply_payload_applies_emoji_preference() -> None:
    payload = {"reply_suggestions": ["a", "b", "c"], "emoji_replies": ["e 😀"], "emoji_replies_without_emojis": ["e"]}
    both = _normalize(payload)
    assert both is not None
    assert both["reply_suggestions"] == ["a", "b"]
    assert both["top_reply"] == "a"
    assert both["emoji_replies_with_emojis"] == ["e 😀"]
    assert both["detected_intent"] == "opener"
    assert _normalize(payload, "with")["reply_suggestions"] == ["e 😀"]
    assert _normalize(payload, "without")["reply_suggestions"] == ["e"]


def test_normalize_reply_payload_returns_none_without_suggestions() -> None:
    assert _normalize({"reply_suggestions": [" "]}) is None


# --- Service fallbacks (fake models, no network) ---------------------------------------------


def test_generate_response_keyword_block_skips_models() -> None:
    service, llms = _service(_fail)
    result = service.generate_response(conversation=CHAT, draft="help me stalk her")
    assert result["error"] == "safety_block"
    assert all(not llm.calls for llm in llms)


def test_generate_response_all_models_down_returns_error_payload() -> None:
    service, _ = _service(_fail)
    result = service.generate_response(conversation=CHAT, draft="sounds fun", tone="Witty")
    assert result == canned_responses.reply_generation_error_payload("llm_unavailable", "witty")


def test_generate_response_falls_back_to_next_model() -> None:
    def responder(system: str, _user: str) -> str:
        if "safety classifier" in system:
            return '{"allow": true}'
        if "You classify" in system:
            return '{"mode": "contextual"}'
        return json.dumps({"reply_suggestions": ["See you there"], "detected_intent": ""})

    service = MessageAssistantService(
        classifier_llm=FakeLLM(_fail), response_llm=FakeLLM(_fail), fallback_llm=FakeLLM(responder)
    )
    result = service.generate_response(conversation=CHAT, draft="")
    assert result["generation_source"] == "fallback_response_llm"
    assert result["routing_source"] == "fallback_response_llm"
    assert result["reply_suggestions"] == ["See you there"]
    assert result["detected_intent"] == "casual"


def test_answer_routes_without_models() -> None:
    service, _ = _service(_fail)
    assert service.answer_about_conversation(question="hi")["mode"] == "assistant_greeting"
    transcript = service.answer_about_conversation(conversation=CHAT, question="show my chat")
    assert transcript["answer"].endswith("- You: sure")
    # Models down: the heuristic sends this to general help, which then falls back to canned advice.
    assert service.answer_about_conversation(question="how to text")["mode"] == "general_message_help"


async def _collect(chunks: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in chunks]


async def test_stream_answer_streams_then_falls_back() -> None:
    def responder(system: str, _user: str) -> str:
        if "Do not return JSON" in system:
            return "Hel|lo"
        return '{"allow": true, "mode": "contextual"}'

    service, _ = _service(responder)
    assert await _collect(service.stream_answer(conversation=CHAT, question="what did they ask")) == ["Hel", "lo"]

    service, _ = _service(_fail)
    chunks = await _collect(service.stream_answer(conversation=CHAT, question="what did they ask"))
    assert chunks == [canned_responses.fallback_answer("what did they ask")["answer"]]


# --- Schemas ---------------------------------------------------------------------------------


def test_request_schema_defaults_and_validation() -> None:
    request = MessageAssistantRequest()
    assert request.mode == "replies" and request.tone == "polite" and request.emojiPreference == "both"
    assert request.max_suggestions == 5 and request.other_name == "the other person"
    turn = MessageAssistantRequest(conversation=[{"role": "me"}]).conversation[0]
    assert turn.model_dump() == {"role": "me", "text": "", "kind": "text", "timestamp": None}
    with pytest.raises(ValidationError):
        MessageAssistantRequest(conversation=[{"text": "missing role"}])


# --- Router ----------------------------------------------------------------------------------


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate_response(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("generate", kwargs))
        return {"top_reply": "hey"}

    def answer_about_conversation(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("answer", kwargs))
        return {"answer": "42"}

    async def stream_answer(self, **kwargs: Any) -> AsyncIterator[str]:
        self.calls.append(("stream", kwargs))
        for chunk in ("a", "b"):
            yield chunk


@pytest.fixture
def client_and_service() -> tuple[TestClient, FakeService]:
    app = FastAPI()
    app.include_router(router, prefix="/api/message-ai")
    app.dependency_overrides[require_internal_key] = lambda: None
    fake = FakeService()
    app.dependency_overrides[get_message_assistant_service] = lambda: fake
    return TestClient(app), fake


def test_respond_generates_replies(client_and_service: tuple[TestClient, FakeService]) -> None:
    client, fake = client_and_service
    body = {"conversation": [{"role": "other", "text": "yo"}], "draft": "d", "emojiPreference": "with"}
    response = client.post("/api/message-ai/respond", json=body)
    assert response.status_code == 200 and response.json() == {"top_reply": "hey"}
    name, kwargs = fake.calls[0]
    assert name == "generate"
    assert kwargs["emoji_preference"] == "with" and kwargs["draft"] == "d" and kwargs["max_suggestions"] == 5
    assert kwargs["conversation"] == [{"role": "other", "text": "yo", "kind": "text", "timestamp": None}]


def test_respond_ask_mode(client_and_service: tuple[TestClient, FakeService]) -> None:
    client, fake = client_and_service
    body = {"mode": "ask", "question": "q", "assistant_history": [{"role": "user"}]}
    response = client.post("/api/message-ai/respond", json=body)
    assert response.json() == {"answer": "42"}
    assert fake.calls[0][0] == "answer"
    assert fake.calls[0][1]["assistant_history"] == [{"role": "user", "text": ""}]


def test_respond_stream(client_and_service: tuple[TestClient, FakeService]) -> None:
    client, fake = client_and_service
    response = client.post("/api/message-ai/respond/stream", json={"question": "q"})
    assert response.status_code == 200 and response.text == "ab"
    assert response.headers["content-type"].startswith("text/plain")
    assert fake.calls[0][1]["question"] == "q"


def test_respond_rejects_invalid_body(client_and_service: tuple[TestClient, FakeService]) -> None:
    client, _ = client_and_service
    response = client.post("/api/message-ai/respond", json={"max_suggestions": "lots"})
    assert response.status_code == 422
