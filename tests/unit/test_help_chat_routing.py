"""Router output parsing and the greeting shortcut."""

import pytest
from langchain_core.runnables import RunnableLambda

from src.help_chat.routing import QuestionRouter, normalize

SECTIONS = ["auth", "cliq_details", "create"]


def _failing_llm(_input: object) -> str:
    raise AssertionError("the LLM must not be called")


@pytest.fixture
def router() -> QuestionRouter:
    return QuestionRouter(RunnableLambda(_failing_llm), SECTIONS)


def test_greeting_skips_the_llm(router: QuestionRouter) -> None:
    language, parts = router.route("Hello :)")
    assert parts[0].intent == "greeting"
    assert language == "other"


def test_parse_valid_json_inside_code_fence(router: QuestionRouter) -> None:
    part = '{"question": "Reset password?", "intent": "steps", "section": "auth"}'
    raw = f'```json\n{{"language": "english", "parts": [{part}]}}\n```'
    language, parts = router.parse(raw, "original")
    assert language == "english"
    assert (parts[0].question, parts[0].intent, parts[0].section) == ("Reset password?", "steps", "auth")


def test_parse_replaces_unknown_values(router: QuestionRouter) -> None:
    raw = '{"language": "klingon", "parts": [{"question": " ", "intent": "dance", "section": "nowhere"}]}'
    language, parts = router.parse(raw, "original")
    assert language == "other"
    assert (parts[0].question, parts[0].intent, parts[0].section) == ("original", "information", "cliq_details")


def test_parse_keeps_at_most_three_parts(router: QuestionRouter) -> None:
    raw = '{"language": "english", "parts": [' + ",".join(['{"question": "q"}'] * 5) + "]}"
    assert len(router.parse(raw, "m")[1]) == 3


def test_unparseable_output_still_searches_the_raw_message(router: QuestionRouter) -> None:
    language, parts = router.parse("not json", "मेरा पासवर्ड")
    assert language == "hindi"
    assert parts[0].question == "मेरा पासवर्ड"
    assert parts[0].intent == "information"


def test_normalize_strips_punctuation_and_case() -> None:
    assert normalize("  Hey   THERE!! ") == "hey there"
