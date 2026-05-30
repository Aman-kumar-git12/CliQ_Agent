import json
import re
from typing import Any

from .config import ALLOWED_TONES, SAFETY_BLOCK_PATTERNS

def _normalize_turn(turn: dict[str, Any]) -> str:
    role = "You" if turn.get("role") == "me" else "Other person"
    text = (turn.get("text") or "").strip()
    kind = (turn.get("kind") or "text").strip().lower()

    if not text:
        if kind == "image":
            text = "[Image]"
        elif kind == "voice":
            text = "[Voice message]"
        elif kind == "file":
            text = "[File]"
        else:
            text = "[Empty]"

    return f"{role}: {text}"


def _normalize_tone(tone: str = "polite") -> str:
    normalized = str(tone or "polite").strip().lower()
    return normalized if normalized in ALLOWED_TONES else "polite"


def _normalized_user_text(text: str = "") -> str:
    return " ".join(str(text or "").strip().lower().split())


def _build_assistant_history_section(assistant_history: list[dict[str, Any]] | None = None) -> str:
    turns = assistant_history or []
    if not turns:
        return "No prior Ask AI follow-up context."

    lines = []
    for turn in turns[-12:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines) if lines else "No prior Ask AI follow-up context."


def _is_greeting_prompt(question: str = "") -> bool:
    normalized = _normalized_user_text(question)
    if not normalized:
        return False
    return normalized in {
        "hi",
        "hii",
        "hiii",
        "hello",
        "hey",
        "heyy",
        "good morning",
        "good evening",
    }


def _is_capability_prompt(question: str = "") -> bool:
    normalized = _normalized_user_text(question)
    if not normalized:
        return False
    capability_markers = [
        "what can you do",
        "what will you do",
        "how can you help",
        "what do you do",
        "help me",
        "how do you help",
    ]
    return any(marker in normalized for marker in capability_markers)


def _is_transcript_or_summary_prompt(question: str = "") -> bool:
    normalized = _normalized_user_text(question)
    if not normalized:
        return False
    markers = [
        "give me my chat conversation",
        "give me my conversation",
        "show my chat",
        "show me the conversation",
        "show conversation",
        "show chat",
        "chat conversation",
        "give my chat",
        "summarize this chat",
        "summary of this chat",
        "summarize conversation",
        "conversation summary",
        "what happened in this chat",
    ]
    return any(marker in normalized for marker in markers)


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    text = (candidate or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_balanced_json_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    stack = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(raw_text):
        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if stack == 0:
                start_index = index
            stack += 1
        elif char == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start_index is not None:
                candidate = raw_text[start_index:index + 1].strip()
                if candidate and candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
                start_index = None

    return candidates


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise ValueError("Empty assistant response")

    stripped = raw_text.strip()
    direct_match = _try_parse_json_object(stripped)
    if direct_match is not None:
        return direct_match

    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        parsed = _try_parse_json_object(block)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, end_index = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            trailing = raw_text[index + end_index:].strip()
            if not trailing or trailing.startswith("```"):
                return parsed

    for candidate in _extract_balanced_json_candidates(raw_text):
        parsed = _try_parse_json_object(candidate)
        if parsed is not None:
            return parsed

    raise ValueError("Assistant response did not contain valid JSON")


def _ensure_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _contains_unsafe_messaging_request(*parts: str) -> bool:
    text = " ".join(part.strip().lower() for part in parts if part).strip()
    if not text:
        return False
    return any(pattern in text for pattern in SAFETY_BLOCK_PATTERNS)


def _safety_reply_payload(tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
    return {
        "error": "safety_block",
        "generation_source": "safety_guard",
        "routing_source": "safety_guard",
        "top_reply": "",
        "reply_suggestions": [],
        "emoji_replies": [],
        "rewrites": {
            "clean": "",
            "short": "",
            "warm": "",
            "confident": "",
        },
        "same_message_variants": [],
        "grouped_replies": {
            "top_reply": "",
            "safe_reply": "",
            "warm_reply": "",
            "playful_reply": "",
            "curious_reply": "",
            "direct_reply": "",
            "opener": "",
            "follow_up": "",
            "short_reply": "",
            "confident_reply": "",
        },
        "context_focus": "",
        "conversation_summary": "",
        "detected_intent": "safety_block",
        "tone": tone,
        "safety_message": "I can help with respectful, safe messaging, but not with coercive, abusive, or exploitative requests.",
    }


def _safety_answer_payload(tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
    return {
        "answer": "I can help with respectful, safe messaging, but not with coercive, abusive, or exploitative requests.",
        "suggested_actions": [
            "Ask for a respectful rewrite.",
            "Ask for a calm boundary-setting message.",
            "Ask for a safer way to respond.",
        ],
        "mode": "safety_block",
        "tone": tone,
    }


def _is_contextual_chat_question(question: str = "") -> bool:
    normalized = (question or "").strip().lower()
    if not normalized:
        return True

    contextual_markers = [
        "this chat",
        "this conversation",
        "our chat",
        "what did",
        "what does",
        "did they",
        "did he",
        "did she",
        "what is the other person",
        "summarize",
        "summary",
        "reply to this",
        "reply to that",
        "what should i reply",
        "what should i say next",
        "what do i say next",
        "what are they asking",
        "context",
        "last message",
        "recent message",
        "their message",
        "his message",
        "her message",
        "from the chat",
    ]

    general_markers = [
        "how to chat",
        "how do i chat",
        "how to text",
        "how do i text",
        "flirty messages",
        "flirty message",
        "romantic messages",
        "pickup line",
        "pick up line",
        "say hi",
        "say hello",
        "hello message",
        "hi message",
        "good morning message",
        "general messages",
        "opening message",
        "starter message",
        "conversation starter",
        "ice breaker",
        "icebreaker",
    ]

    if any(marker in normalized for marker in contextual_markers):
        return True

    if any(marker in normalized for marker in general_markers):
        return False

    if any(word in normalized for word in ["flirty", "romantic", "cute", "greeting", "hello", "hi"]) and "reply" not in normalized:
        return False

    return True


def _reply_generation_error_payload(reason: str = "llm_unavailable", tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
    return {
        "error": reason,
        "generation_source": "unavailable",
        "top_reply": "",
        "reply_suggestions": [],
        "emoji_replies": [],
        "rewrites": {
            "clean": "",
            "short": "",
            "warm": "",
            "confident": "",
        },
        "same_message_variants": [],
        "grouped_replies": {
            "top_reply": "",
            "safe_reply": "",
            "warm_reply": "",
            "playful_reply": "",
            "curious_reply": "",
            "direct_reply": "",
            "opener": "",
            "follow_up": "",
            "short_reply": "",
            "confident_reply": "",
        },
        "context_focus": "",
        "conversation_summary": "",
        "detected_intent": "unknown",
        "tone": tone,
    }


def _fallback_answer(question: str = "") -> dict[str, Any]:
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


def _greeting_answer_payload(tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
    return {
        "answer": (
            "Hello, I can help you with this chat. I can summarize the conversation, explain what the other person means, "
            "suggest replies, or help you decide what to say next."
        ),
        "suggested_actions": [
            "Summarize this chat.",
            "What should I reply next?",
            "What can you do?",
        ],
        "mode": "assistant_greeting",
        "tone": tone,
        "generation_source": "rules_greeting",
        "routing_source": "assistant_intro",
    }


def _capability_answer_payload(tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
    return {
        "answer": (
            "I can help you understand this chat, summarize what happened, explain the last message, suggest polite or natural replies, "
            "rewrite your message in different tones, and help you figure out what to say next."
        ),
        "suggested_actions": [
            "Summarize this chat.",
            "Explain the last message.",
            "Give me a polite reply.",
        ],
        "suggested_replies": [
            "Summarize this chat for me.",
            "What should I reply here?",
            "Explain what the other person means.",
        ],
        "mode": "assistant_capabilities",
        "tone": tone,
        "generation_source": "rules_capabilities",
        "routing_source": "assistant_intro",
    }


def _conversation_transcript_payload(
    conversation: list[dict[str, Any]] | None = None,
    older_context: str = "",
    tone: str = "polite",
) -> dict[str, Any]:
    tone = _normalize_tone(tone)
    conversation_text, older_text = _build_context_sections(conversation, older_context)
    older_snippets = [line.strip() for line in older_text.splitlines() if line.strip() and older_text != "No older context available."]
    recent_lines = [line.strip() for line in conversation_text.splitlines() if line.strip() and conversation_text != "No prior messages."]
    transcript_lines = (older_snippets + recent_lines)[-14:]

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
        "suggested_actions": [
            "Summarize it briefly.",
            "What should I reply next?",
            "Explain the last message.",
        ],
        "mode": "conversation_transcript",
        "tone": tone,
        "generation_source": "rules_transcript",
        "routing_source": "conversation_context",
    }


def _fallback_general_clarifier(tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
    return {
        "answer": (
            "I can help with this chat. You can ask me to summarize the conversation, explain the last message, "
            "suggest a reply, or tell you what to say next."
        ),
        "suggested_actions": [
            "Summarize this chat.",
            "What should I reply?",
            "Explain the last message.",
        ],
        "suggested_replies": [
            "Summarize this chat for me.",
            "Give me a polite reply.",
            "What does the last message mean?",
        ],
        "mode": "assistant_clarifier",
        "tone": tone,
        "generation_source": "rules_clarifier",
        "routing_source": "assistant_intro",
    }


def _fallback_general_message_help(question: str = "", tone: str = "polite") -> dict[str, Any]:
    tone = _normalize_tone(tone)
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
        "tone": tone,
    }


def _should_use_general_reply_mode(
    conversation: list[dict[str, Any]] | None = None,
    draft: str = "",
) -> bool:
    turns = conversation or []
    meaningful_turns = [
        (turn.get("text") or "").strip().lower()
        for turn in turns
        if (turn.get("text") or "").strip()
    ]

    normalized_draft = (draft or "").strip().lower()

    opener_markers = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening",
        "start the chat",
        "start chat",
        "opening message",
        "open the chat",
        "first message",
        "flirty opener",
        "ice breaker",
        "icebreaker",
        "introduce myself",
    ]

    if len(meaningful_turns) <= 1:
        return True

    if not meaningful_turns and not normalized_draft:
        return True

    if any(marker in normalized_draft for marker in opener_markers):
        return True

    if normalized_draft in {"hi", "hello", "hey", "hii", "heyy"}:
        return True

    return False


def _build_context_sections(
    conversation: list[dict[str, Any]] | None = None,
    older_context: str = "",
) -> tuple[str, str]:
    turns = conversation or []
    recent_turns = turns[-20:]
    conversation_text = "\n".join(_normalize_turn(turn) for turn in recent_turns) or "No prior messages."
    older_text = older_context.strip() or "No older context available."
    return conversation_text, older_text


def _get_last_other_person_message(conversation: list[dict[str, Any]] | None = None) -> str:
    for turn in reversed(conversation or []):
        if turn.get("role") != "me":
            text = (turn.get("text") or "").strip()
            if text:
                return text
    return ""


def _context_confidence(conversation: list[dict[str, Any]] | None = None, older_context: str = "") -> str:
    turns = conversation or []
    meaningful_turns = [
        (turn.get("text") or "").strip()
        for turn in turns
        if (turn.get("text") or "").strip()
    ]
    last_other = _get_last_other_person_message(turns).strip()

    if len(meaningful_turns) >= 4 and len(last_other) >= 12:
        return "high"
    if len(meaningful_turns) >= 2 or older_context.strip():
        return "medium"
    return "low"


def _normalize_grouped_replies(
    grouped_replies: dict[str, Any] | None = None,
    rewrites: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    draft: str = "",
) -> dict[str, str]:
    grouped_replies = grouped_replies if isinstance(grouped_replies, dict) else {}
    rewrites = rewrites if isinstance(rewrites, dict) else {}
    payload = payload if isinstance(payload, dict) else {}

    top_reply = str(grouped_replies.get("top_reply") or payload.get("top_reply") or draft or "").strip()
    safe_reply = str(
        grouped_replies.get("safe_reply")
        or grouped_replies.get("opener")
        or grouped_replies.get("short_reply")
        or rewrites.get("clean")
        or draft
        or ""
    ).strip()
    warm_reply = str(grouped_replies.get("warm_reply") or rewrites.get("warm") or "").strip()
    playful_reply = str(grouped_replies.get("playful_reply") or grouped_replies.get("follow_up") or "").strip()
    curious_reply = str(grouped_replies.get("curious_reply") or grouped_replies.get("follow_up") or "").strip()
    direct_reply = str(grouped_replies.get("direct_reply") or grouped_replies.get("confident_reply") or rewrites.get("confident") or "").strip()
    short_reply = str(grouped_replies.get("short_reply") or rewrites.get("short") or safe_reply or "").strip()
    opener = str(grouped_replies.get("opener") or safe_reply or "").strip()
    follow_up = str(grouped_replies.get("follow_up") or curious_reply or playful_reply or "").strip()
    confident_reply = str(grouped_replies.get("confident_reply") or direct_reply or "").strip()

    return {
        "top_reply": top_reply,
        "safe_reply": safe_reply,
        "warm_reply": warm_reply,
        "playful_reply": playful_reply,
        "curious_reply": curious_reply,
        "direct_reply": direct_reply,
        "opener": opener,
        "follow_up": follow_up,
        "short_reply": short_reply,
        "confident_reply": confident_reply,
    }


def _detect_intent_fallback(conversation: list[dict[str, Any]] | None = None) -> str:
    turns = conversation or []
    if not turns:
        return "opener"
    
    last_text = ""
    for turn in reversed(turns):
        txt = (turn.get("text") or "").strip()
        if txt:
            last_text = txt.lower()
            break
            
    if not last_text:
        return "general"
        
    if "?" in last_text:
        return "question"
        
    for kw in ["invite", "meet", "go out", "hang out", "date", "dinner", "coffee", "see you"]:
        if kw in last_text:
            return "invitation"
            
    for kw in ["sorry", "apologize", "forgive"]:
        if kw in last_text:
            return "apology"
            
    return "casual"
