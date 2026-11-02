"""MessageAssistantService: reply suggestions and Ask AI answers for a 1:1 chat."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel

from . import canned_responses, prompts
from .classification import ModeClassifier
from .conversation import (
    Turn,
    build_assistant_history_section,
    build_context_sections,
    build_contextual_reply_prompts,
    detect_intent_fallback,
)
from .json_output import ensure_list
from .model_chain import LLMUnavailableError, NamedModel, invoke_json_model, stream_with_fallback
from .routing import (
    is_capability_prompt,
    is_greeting_prompt,
    is_transcript_or_summary_prompt,
    normalized_user_text,
)
from .safety import contains_unsafe_messaging_request
from .suggestions import normalize_reply_payload
from .tone import DEFAULT_TONE, get_emoji_instruction, get_tone_instruction, normalize_tone

logger = logging.getLogger(__name__)

DEFAULT_OTHER_NAME = "the other person"
SUGGESTED_ACTION_LIMIT = 3
SUGGESTED_REPLY_LIMIT = 3


class MessageAssistantService:
    """Generates reply suggestions and answers questions about a conversation using Groq chat models."""

    def __init__(self, classifier_llm: BaseChatModel, response_llm: BaseChatModel, fallback_llm: BaseChatModel) -> None:
        # Priority order for every call; the labels are returned to the client as `generation_source`.
        self._models = (
            NamedModel("response_llm", response_llm),
            NamedModel("fallback_response_llm", fallback_llm),
            NamedModel("classifier_llm", classifier_llm),
        )
        # Safety and routing answer with a few JSON tokens: the small, fast model first, never the large
        # response model. When every classifier fails, routing uses its heuristic and safety fails open.
        self._classifier = ModeClassifier(
            (NamedModel("classifier_llm", classifier_llm), NamedModel("fallback_response_llm", fallback_llm))
        )

    def generate_response(
        self,
        conversation: list[Turn] | None = None,
        draft: str = "",
        other_name: str = DEFAULT_OTHER_NAME,
        older_context: str = "",
        tone: str = DEFAULT_TONE,
        max_suggestions: int = 5,
        emoji_preference: str | None = "both",
    ) -> dict[str, Any]:
        """Reply suggestions grounded in the conversation, or opener suggestions when the chat is too sparse."""
        tone = normalize_tone(tone)
        if contains_unsafe_messaging_request(draft):
            return canned_responses.safety_reply_payload(tone)
        routed = self._classifier.classify_unless_blocked(
            draft, lambda: self._classifier.classify_reply_mode(conversation, draft, older_context)
        )
        if routed is None:
            return canned_responses.safety_reply_payload(tone)

        reply_mode, routing_source = routed
        if reply_mode == "general":
            # The draft was already checked above, so the opener path does not ask the safety model again
            return self._general_replies(draft, tone, max_suggestions, emoji_preference)

        system_prompt, user_prompt = build_contextual_reply_prompts(
            conversation, draft, other_name, older_context, tone, max_suggestions, emoji_preference
        )
        return self._generate_replies(
            system_prompt,
            user_prompt,
            draft=draft,
            tone=tone,
            max_suggestions=max_suggestions,
            emoji_preference=emoji_preference,
            fallback_intent=detect_intent_fallback(conversation),
            routing_source=routing_source,
        )

    def generate_general_replies(
        self,
        draft: str = "",
        tone: str = DEFAULT_TONE,
        max_suggestions: int = 5,
        emoji_preference: str | None = "both",
    ) -> dict[str, Any]:
        """Opener-style suggestions for an empty or sparse chat."""
        tone = normalize_tone(tone)
        if self._classifier.should_block(draft):
            return canned_responses.safety_reply_payload(tone)
        return self._general_replies(draft, tone, max_suggestions, emoji_preference)

    def _general_replies(
        self, draft: str, tone: str, max_suggestions: int, emoji_preference: str | None
    ) -> dict[str, Any]:
        user_prompt = prompts.general_reply_user_prompt(
            draft, tone, get_tone_instruction(tone), get_emoji_instruction(emoji_preference), max_suggestions
        )
        return self._generate_replies(
            prompts.GENERAL_REPLY_SYSTEM_PROMPT,
            user_prompt,
            draft=draft,
            tone=tone,
            max_suggestions=max_suggestions,
            emoji_preference=emoji_preference,
            fallback_intent="opener",
            routing_source="general_mode",
        )

    def _generate_replies(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        draft: str,
        tone: str,
        max_suggestions: int,
        emoji_preference: str | None,
        fallback_intent: str,
        routing_source: str,
    ) -> dict[str, Any]:
        try:
            payload, generation_source = invoke_json_model(self._models, system_prompt, user_prompt)
        except LLMUnavailableError:
            logger.warning("Reply suggestions unavailable: every model failed")
            return canned_responses.reply_generation_error_payload("llm_unavailable", tone)

        normalized = normalize_reply_payload(
            payload,
            draft=draft,
            tone=tone,
            max_suggestions=max_suggestions,
            emoji_preference=emoji_preference,
            fallback_intent=fallback_intent,
            generation_source=generation_source,
            routing_source=routing_source,
        )
        if normalized is None:
            logger.warning("Model %s returned no reply suggestions", generation_source)
            return canned_responses.reply_generation_error_payload("empty_llm_reply", tone)
        return normalized

    def answer_about_conversation(
        self,
        conversation: list[Turn] | None = None,
        assistant_history: list[Turn] | None = None,
        question: str = "",
        other_name: str = DEFAULT_OTHER_NAME,
        older_context: str = "",
        tone: str = DEFAULT_TONE,
    ) -> dict[str, Any]:
        """Ask AI answer about the conversation, routed to general advice when the question is not chat-specific."""
        tone = normalize_tone(tone)
        canned, question_mode, routing_source = self._route_question(conversation, question, older_context, tone)
        if canned is not None:
            return canned
        if question_mode != "contextual":
            return self._general_help_answer(question, tone)

        conversation_text, older_text = build_context_sections(conversation, older_context)
        user_prompt = prompts.conversation_answer_user_prompt(
            older_text,
            other_name,
            conversation_text,
            build_assistant_history_section(assistant_history),
            question,
            tone,
        )
        return self._answer_from_conversation(user_prompt, question, tone, routing_source)

    def _answer_from_conversation(
        self, user_prompt: str, question: str, tone: str, routing_source: str
    ) -> dict[str, Any]:
        try:
            payload, generation_source = invoke_json_model(
                self._models, prompts.CONVERSATION_ANSWER_SYSTEM_PROMPT, user_prompt
            )
        except LLMUnavailableError:
            logger.warning("Conversation answer unavailable: every model failed")
            return canned_responses.fallback_answer(question)

        answer = str(payload.get("answer") or "").strip()
        if not answer:
            return canned_responses.fallback_answer(question)
        return {
            "answer": answer,
            "suggested_actions": ensure_list(payload.get("suggested_actions"), SUGGESTED_ACTION_LIMIT),
            "mode": "conversation_context",
            "tone": tone,
            "generation_source": generation_source,
            "routing_source": routing_source,
        }

    def answer_general_help(self, question: str = "", tone: str = DEFAULT_TONE) -> dict[str, Any]:
        """Ask AI answer for general texting advice that does not depend on the chat."""
        tone = normalize_tone(tone)
        rule_answer = self._rule_based_answer(question, tone)
        if rule_answer is not None:
            return rule_answer
        return self._general_help_answer(question, tone)

    def _general_help_answer(self, question: str, tone: str) -> dict[str, Any]:
        """General help for a question that has already passed the safety checks."""
        if not normalized_user_text(question):
            return canned_responses.fallback_general_clarifier(tone)

        user_prompt = prompts.general_help_user_prompt(question, tone)
        try:
            payload, generation_source = invoke_json_model(
                self._models, prompts.GENERAL_HELP_SYSTEM_PROMPT, user_prompt
            )
        except LLMUnavailableError:
            logger.warning("General messaging help unavailable: every model failed")
            return canned_responses.fallback_general_message_help(tone)

        answer = str(payload.get("answer") or "").strip()
        if not answer:
            return canned_responses.fallback_general_message_help(tone)
        return {
            "answer": answer,
            "suggested_actions": ensure_list(payload.get("suggested_actions"), SUGGESTED_ACTION_LIMIT),
            "suggested_replies": ensure_list(payload.get("suggested_replies"), SUGGESTED_REPLY_LIMIT),
            "mode": "general_message_help",
            "tone": str(payload.get("tone") or tone).strip() or tone,
            "generation_source": generation_source,
            "routing_source": "general_mode",
        }

    async def stream_answer(
        self,
        conversation: list[Turn] | None = None,
        assistant_history: list[Turn] | None = None,
        question: str = "",
        other_name: str = DEFAULT_OTHER_NAME,
        older_context: str = "",
        tone: str = DEFAULT_TONE,
    ) -> AsyncIterator[str]:
        """Streaming plain-text version of answer_about_conversation.

        The safety and routing classifiers are blocking LLM calls, so they run in a worker thread.
        """
        tone = normalize_tone(tone)
        canned, question_mode, _ = await asyncio.to_thread(
            self._route_question, conversation, question, older_context, tone
        )
        if canned is not None:
            yield canned["answer"]
            return
        if question_mode != "contextual":
            async for chunk in self._stream_general_help_answer(question, tone):
                yield chunk
            return

        conversation_text, older_text = build_context_sections(conversation, older_context)
        user_prompt = prompts.stream_conversation_user_prompt(
            older_text,
            other_name,
            conversation_text,
            build_assistant_history_section(assistant_history),
            question,
            tone,
        )
        fallback = canned_responses.fallback_answer(question)["answer"]
        async for chunk in stream_with_fallback(
            self._models, prompts.STREAM_CONVERSATION_SYSTEM_PROMPT, user_prompt, fallback
        ):
            yield chunk

    async def stream_general_help(self, question: str = "", tone: str = DEFAULT_TONE) -> AsyncIterator[str]:
        """Streaming plain-text version of answer_general_help."""
        tone = normalize_tone(tone)
        rule_answer = await asyncio.to_thread(self._rule_based_answer, question, tone)
        if rule_answer is not None:
            yield rule_answer["answer"]
            return
        async for chunk in self._stream_general_help_answer(question, tone):
            yield chunk

    async def _stream_general_help_answer(self, question: str, tone: str) -> AsyncIterator[str]:
        """Streamed general help for a question that has already passed the safety checks."""
        if not normalized_user_text(question):
            yield canned_responses.fallback_general_clarifier(tone)["answer"]
            return

        user_prompt = prompts.stream_general_help_user_prompt(question, tone)
        fallback = canned_responses.fallback_general_message_help(tone)["answer"]
        async for chunk in stream_with_fallback(
            self._models, prompts.STREAM_GENERAL_HELP_SYSTEM_PROMPT, user_prompt, fallback
        ):
            yield chunk

    def _route_question(
        self, conversation: list[Turn] | None, question: str, older_context: str, tone: str
    ) -> tuple[dict[str, Any] | None, str, str]:
        """(canned answer, question mode, routing source) for an Ask AI question.

        A canned answer needs no generation: a safety refusal, a greeting, a capability blurb or the transcript.
        A bare greeting can't be unsafe, so it skips the safety model; everything else is checked by it while
        the question-mode classifier runs alongside.
        """
        if contains_unsafe_messaging_request(question):
            return canned_responses.safety_answer_payload(tone), "", ""
        if is_greeting_prompt(question):
            return canned_responses.greeting_answer_payload(tone), "", ""

        if is_capability_prompt(question) or is_transcript_or_summary_prompt(question):
            if self._classifier.should_block(question):
                return canned_responses.safety_answer_payload(tone), "", ""
            if is_capability_prompt(question):
                return canned_responses.capability_answer_payload(tone), "", ""
            return canned_responses.conversation_transcript_payload(conversation, older_context, tone), "", ""

        routed = self._classifier.classify_unless_blocked(
            question, lambda: self._classifier.classify_question_mode(question)
        )
        if routed is None:
            return canned_responses.safety_answer_payload(tone), "", ""
        return None, *routed

    def _rule_based_answer(self, question: str, tone: str) -> dict[str, Any] | None:
        """Answers that need no generation: safety refusals, greetings and capability questions."""
        if self._classifier.should_block(question):
            return canned_responses.safety_answer_payload(tone)
        if is_greeting_prompt(question):
            return canned_responses.greeting_answer_payload(tone)
        if is_capability_prompt(question):
            return canned_responses.capability_answer_payload(tone)
        return None
