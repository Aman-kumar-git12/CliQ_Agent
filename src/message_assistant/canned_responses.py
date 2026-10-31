"""Fixed responses for safety blocks, intro questions, transcripts and LLM failures (no model call needed)."""

from typing import Any

from .conversation import NO_OLDER_CONTEXT, NO_PRIOR_MESSAGES, Turn, build_context_sections
from .tone import DEFAULT_TONE, normalize_tone

SAFETY_MESSAGE = "I can help with respectful, safe messaging, but not with coercive, abusive, or exploitative requests."

# Keep the transcript recap short enough to read in the chat side panel.
MAX_TRANSCRIPT_LINES = 14

GROUPED_REPLY_KEYS = (
    "top_reply",
    "safe_reply",
    "warm_reply",
    "playful_reply",
    "curious_reply",
    "direct_reply",
    "opener",
    "follow_up",
    "short_reply",
    "confident_reply",
)


def _empty_reply_payload(tone: str) -> dict[str, Any]:
    """Reply-mode response with every suggestion field present but empty."""
    return {
        "top_reply": "",
        "reply_suggestions": [],
        "emoji_replies": [],
        "emoji_replies_with_emojis": [],
        "emoji_replies_without_emojis": [],
        "rewrites": {"clean": "", "short": "", "warm": "", "confident": ""},
        "same_message_variants": [],
        "grouped_replies": dict.fromkeys(GROUPED_REPLY_KEYS, ""),
        "context_focus": "",
        "conversation_summary": "",
        "detected_intent": "unknown",
        "tone": tone,
    }


def safety_reply_payload(tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Reply-mode response when the draft asks for something harmful."""
    payload = {"error": "safety_block", "generation_source": "safety_guard", "routing_source": "safety_guard"}
    payload.update(_empty_reply_payload(normalize_tone(tone)))
    payload["detected_intent"] = "safety_block"
    payload["safety_message"] = SAFETY_MESSAGE
    return payload


def reply_generation_error_payload(reason: str = "llm_unavailable", tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Reply-mode response when the models fail or return nothing usable."""
    payload = {"error": reason, "generation_source": "unavailable"}
    payload.update(_empty_reply_payload(normalize_tone(tone)))
    return payload


def safety_answer_payload(tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Ask-mode response when the question asks for something harmful."""
    return {
        "answer": SAFETY_MESSAGE,
        "suggested_actions": [
            "Ask for a respectful rewrite.",
            "Ask for a calm boundary-setting message.",
            "Ask for a safer way to respond.",
        ],
        "mode": "safety_block",
        "tone": normalize_tone(tone),
    }


def fallback_answer(question: str = "") -> dict[str, Any]:
    """Ask-mode response when the conversation answer could not be generated."""
    answer = (
        "I could not fully analyze this conversation right now, but I can still help once you try again."
        if question.strip()
        else "Ask me anything about this conversation and I will help."
    )
    return {
        "answer": answer,
        "suggested_actions": [
            "Ask for a summary of the last few messages.",
            "Ask for a polite or witty reply.",
            "Ask what the other person is requesting.",
        ],
    }


def greeting_answer_payload(tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Ask-mode response to a bare greeting."""
    return {
        "answer": (
            "Hello, I can help you with this chat. I can summarize the conversation, explain what the other person "
            "means, suggest replies, or help you decide what to say next."
        ),
        "suggested_actions": ["Summarize this chat.", "What should I reply next?", "What can you do?"],
        "mode": "assistant_greeting",
        "tone": normalize_tone(tone),
        "generation_source": "rules_greeting",
        "routing_source": "assistant_intro",
    }


def capability_answer_payload(tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Ask-mode response to 'what can you do'."""
    return {
        "answer": (
            "I can help you understand this chat, summarize what happened, explain the last message, suggest polite "
            "or natural replies, rewrite your message in different tones, and help you figure out what to say next."
        ),
        "suggested_actions": ["Summarize this chat.", "Explain the last message.", "Give me a polite reply."],
        "suggested_replies": [
            "Summarize this chat for me.",
            "What should I reply here?",
            "Explain what the other person means.",
        ],
        "mode": "assistant_capabilities",
        "tone": normalize_tone(tone),
        "generation_source": "rules_capabilities",
        "routing_source": "assistant_intro",
    }


def conversation_transcript_payload(
    conversation: list[Turn] | None = None, older_context: str = "", tone: str = DEFAULT_TONE
) -> dict[str, Any]:
    """Ask-mode response listing the latest chat lines, built without a model call."""
    tone = normalize_tone(tone)
    conversation_text, older_text = build_context_sections(conversation, older_context)
    older_snippets = _non_empty_lines(older_text) if older_text != NO_OLDER_CONTEXT else []
    recent_lines = _non_empty_lines(conversation_text) if conversation_text != NO_PRIOR_MESSAGES else []
    transcript_lines = (older_snippets + recent_lines)[-MAX_TRANSCRIPT_LINES:]

    if not transcript_lines:
        return {
            "answer": "I could not find enough messages to show a useful chat summary yet.",
            "suggested_actions": [
                "Ask me to explain the last message.",
                "Ask for a reply suggestion.",
                "Ask what the other person means.",
            ],
            "mode": "conversation_transcript",
            "tone": tone,
            "generation_source": "rules_transcript_empty",
            "routing_source": "conversation_context",
        }

    answer = "Here is a clean view of your chat conversation:\n\n" + "\n".join(f"- {line}" for line in transcript_lines)
    return {
        "answer": answer,
        "suggested_actions": ["Summarize it briefly.", "What should I reply next?", "Explain the last message."],
        "mode": "conversation_transcript",
        "tone": tone,
        "generation_source": "rules_transcript",
        "routing_source": "conversation_context",
    }


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def fallback_general_clarifier(tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Ask-mode response when the general question is empty."""
    return {
        "answer": (
            "I can help with this chat. You can ask me to summarize the conversation, explain the last message, "
            "suggest a reply, or tell you what to say next."
        ),
        "suggested_actions": ["Summarize this chat.", "What should I reply?", "Explain the last message."],
        "suggested_replies": [
            "Summarize this chat for me.",
            "Give me a polite reply.",
            "What does the last message mean?",
        ],
        "mode": "assistant_clarifier",
        "tone": normalize_tone(tone),
        "generation_source": "rules_clarifier",
        "routing_source": "assistant_intro",
    }


def fallback_general_message_help(tone: str = DEFAULT_TONE) -> dict[str, Any]:
    """Ask-mode response when general messaging advice could not be generated."""
    return {
        "answer": (
            "For general messaging, keep it light, specific, and easy to reply to. "
            "A short, warm opener usually works better than something overly long."
        ),
        "suggested_actions": [
            "Give me 5 flirty opening messages.",
            "Give me 5 friendly hi or hello messages.",
            "Rewrite a greeting in a more confident or witty tone.",
        ],
        "suggested_replies": [
            "Hey, you seem fun to talk to. How's your day going?",
            "Hi, I wanted to say hello properly instead of just liking your profile.",
            "Hey, what kind of conversations do you actually enjoy here?",
        ],
        "mode": "general_message_help",
        "tone": normalize_tone(tone),
    }
