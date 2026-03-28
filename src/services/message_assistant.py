import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import classifier_llm, fallback_response_llm, response_llm

ALLOWED_TONES = {
    "casual",
    "polite",
    "formal",
    "flirty",
    "professional",
    "witty",
    "direct",
    "friendly",
    "empathetic",
    "confident",
}

SAFETY_BLOCK_PATTERNS = [
    "blackmail",
    "extort",
    "threaten",
    "stalk",
    "doxx",
    "revenge porn",
    "force them",
    "manipulate",
    "gaslight",
    "make her jealous",
    "make him jealous",
    "underage sex",
    "minor sex",
    "sexual minor",
    "pressure them",
    "coerce",
    "humiliate",
    "degrade them",
    "control them",
    "obsess over",
    "won't take no",
    "won't accept no",
    "lie to them",
    "catfish",
    "trick them",
    "cheat on",
    "hide that i'm",
]


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


def _invoke_json_model(system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
    attempts = [
        ("response_llm", response_llm),
        ("fallback_response_llm", fallback_response_llm),
        ("classifier_llm", classifier_llm),
    ]
    last_error: Exception | None = None

    for source, llm in attempts:
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return _extract_json_payload(getattr(response, "content", "") or ""), source
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("No model attempts were executed")


def _invoke_text_model(system_prompt: str, user_prompt: str) -> str:
    last_error: Exception | None = None
    for llm in (response_llm, fallback_response_llm, classifier_llm):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            content = str(getattr(response, "content", "") or "").strip()
            if content:
                return content
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("Empty model response")


def _contains_unsafe_messaging_request(*parts: str) -> bool:
    text = " ".join(part.strip().lower() for part in parts if part).strip()
    if not text:
        return False
    return any(pattern in text for pattern in SAFETY_BLOCK_PATTERNS)


def _model_flags_unsafe_request(*parts: str) -> bool:
    text = " ".join(part.strip() for part in parts if part).strip()
    if not text:
        return False

    system_prompt = (
        "You are a safety classifier for a private messaging assistant. "
        "Return JSON only with this shape: "
        "{\"allow\": boolean, \"reason\": string}. "
        "Set allow to false if the user is asking for help with coercion, abuse, harassment, manipulation, sexual exploitation, deceit in a harmful way, stalking, threats, or evading consent. "
        "Set allow to true for benign messaging help, boundary-setting, de-escalation, or safety-seeking requests."
    )
    user_prompt = f"User request:\n{text}"

    try:
        payload, _ = _invoke_json_model(system_prompt, user_prompt)
        allow = payload.get("allow")
        if isinstance(allow, bool):
            return not allow
    except Exception:
        pass

    return False


def _should_block_messaging_request(*parts: str) -> bool:
    return _contains_unsafe_messaging_request(*parts) or _model_flags_unsafe_request(*parts)


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


def _classify_question_mode(question: str = "") -> tuple[str, str]:
    normalized = (question or "").strip()
    if not normalized:
        return "contextual", "fallback_empty_question"

    system_prompt = (
        "You classify whether a user question in a chat assistant is about a specific conversation "
        "or is general messaging advice. Return JSON only with this shape: "
        "{\"mode\": string, \"reason\": string}. "
        "mode must be either \"contextual\" or \"general\"."
    )
    user_prompt = (
        f"Question: {normalized}\n\n"
        "Choose contextual if the answer should depend on the user's current chat history. "
        "Choose general if it is broad texting advice, openers, greetings, pickup lines, or generic message-writing help."
    )

    try:
        payload, source = _invoke_json_model(system_prompt, user_prompt)
        mode = str(payload.get("mode") or "").strip().lower()
        if mode in {"contextual", "general"}:
            return mode, source
    except Exception:
        pass

    return ("contextual" if _is_contextual_chat_question(question) else "general"), "heuristic_fallback"


def _classify_reply_mode(
    conversation: list[dict[str, Any]] | None = None,
    draft: str = "",
    older_context: str = "",
) -> tuple[str, str]:
    turns = conversation or []
    conversation_text, older_text = _build_context_sections(turns, older_context)
    last_other = _get_last_other_person_message(turns)
    confidence = _context_confidence(turns, older_context)

    system_prompt = (
        "You classify reply-generation strategy for a chat assistant. Return JSON only with this shape: "
        "{\"mode\": string, \"reason\": string}. "
        "mode must be either \"contextual\" or \"general\". "
        "Choose contextual when the best reply depends on the actual chat context. "
        "Choose general only for sparse first-message/opening-message situations with too little context."
    )
    user_prompt = (
        f"Older context:\n{older_text}\n\n"
        f"Recent conversation:\n{conversation_text}\n\n"
        f"Last other-person message: {last_other or '[Not available]'}\n"
        f"Draft: {draft.strip() or '[No draft provided]'}\n"
        f"Context confidence: {confidence}\n"
    )

    try:
        payload, source = _invoke_json_model(system_prompt, user_prompt)
        mode = str(payload.get("mode") or "").strip().lower()
        if mode in {"contextual", "general"}:
            return mode, source
    except Exception:
        pass

    return ("general" if _should_use_general_reply_mode(turns, draft) else "contextual"), "heuristic_fallback"


def _detect_intent_fallback(conversation: list[dict[str, Any]] | None = None) -> str:
    turns = conversation or []
    text = " ".join((turn.get("text") or "").lower() for turn in turns)
    last_other = _get_last_other_person_message(turns).lower()

    if any(word in text for word in ["meet", "tomorrow", "schedule", "time", "book", "reserve", "date"]):
        return "planning"
    if any(word in text for word in ["come with me", "join me", "would you like", "want to come", "let's go", "come along"]):
        return "invitation"
    if any(word in text for word in ["want to go", "wish i could", "someday", "one day", "dream", "bucket list"]):
        return "dreaming"
    if "?" in last_other or any(word in last_other for word in ["what", "when", "where", "why", "how", "would you", "do you"]):
        return "question"
    if any(word in text for word in ["interested", "love", "like", "sounds fun", "excited", "into that"]):
        return "interest"
    if any(word in text for word in ["mentioned", "saw", "heard", "thinking about", "talking about"]):
        return "casual_mention"
    if any(word in text for word in ["sorry", "apolog", "my bad"]):
        return "apology"
    if any(word in text for word in ["follow up", "checking in", "update"]):
        return "follow_up"
    if any(word in text for word in ["price", "budget", "deal", "cost", "negot"]):
        return "negotiation"
    return "general"


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

    # Short generic writing asks should default to general messaging help.
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


def generate_message_assistant_response(
    conversation: list[dict[str, Any]] | None = None,
    draft: str = "",
    other_name: str = "the other person",
    older_context: str = "",
    tone: str = "polite",
    max_suggestions: int = 5,
) -> dict[str, Any]:
    tone = _normalize_tone(tone)
    if _should_block_messaging_request(draft):
        return _safety_reply_payload(tone)

    reply_mode, routing_source = _classify_reply_mode(conversation, draft, older_context)
    if reply_mode == "general":
        return generate_general_reply_suggestions(draft=draft, tone=tone, max_suggestions=max_suggestions)

    conversation_text, older_text = _build_context_sections(conversation, older_context)
    last_other_message = _get_last_other_person_message(conversation)
    confidence = _context_confidence(conversation, older_context)

    system_prompt = (
        "You are a private messaging assistant inside a chat app. "
        "Use only the recent conversation to craft helpful replies for the user. "
        "Your job is to generate natural, human-sounding, ready-to-send replies, not generic advice. "
        "When the other person asks a simple question, answer it directly first and optionally add a light follow-up. "
        "Avoid robotic, vague, repetitive, or placeholder wording. "
        "Do not explain your reasoning. Do not give advice bullets. Only generate actual messages the user can send. "
        "Make replies feel like modern real chat messages. "
        "Rank replies by realism, not positivity. "
        "The best reply should sound like what a real person would actually send next in this exact conversation. "
        "Avoid filler-first replies like 'That's great', 'Sounds good', 'Nice', or 'Awesome' unless they are expanded with specific context. "
        "Emoji replies should be standalone emoji-only replies when that would feel natural in the conversation. "
        "If emoji-only would feel awkward or unclear, then use short text-plus-emoji messages instead. "
        "Return valid JSON only with this exact shape: "
        "{"
        "\"top_reply\": string, "
        "\"reply_suggestions\": string[], "
        "\"emoji_replies\": string[], "
        "\"rewrites\": {\"clean\": string, \"short\": string, \"warm\": string, \"confident\": string}, "
        "\"same_message_variants\": string[], "
        "\"grouped_replies\": {\"top_reply\": string, \"safe_reply\": string, \"warm_reply\": string, \"playful_reply\": string, \"curious_reply\": string, \"direct_reply\": string}, "
        "\"context_focus\": string, "
        "\"conversation_summary\": string, "
        "\"detected_intent\": string, "
        "\"tone\": string"
        "}. "
        f"Provide exactly {max_suggestions} items in reply_suggestions when possible, 3 items in emoji_replies, "
        "and 4 items in same_message_variants. "
        "Make the replies specific to the conversation with "
        f"{other_name}. Keep them natural, concise, and ready to send. "
        f"Default tone for replies should be {tone} unless a rewrite field naturally needs a different nuance. "
        "Do not invent actions, plans, bookings, travel arrangements, promises, or commitments unless they are clearly supported by the conversation. "
        "If the conversation only shows interest or casual discussion, keep replies low-assumption. "
        "If context confidence is low, be extra conservative and avoid introducing any commitment, logistics, or emotional intensity that is not clearly present. "
        "Classify the conversation using one of these labels when possible: interest, invitation, planning, dreaming, casual_mention, question, apology, follow_up, negotiation, support, casual, general. "
        "Grouped replies must be genuinely different from each other: "
        "top_reply should be the best overall send-now reply, "
        "safe_reply should be low-risk and grounded only in explicit context, "
        "warm_reply should feel friendly and emotionally open, "
        "playful_reply should be light and a little fun without changing the facts, "
        "curious_reply should move the conversation forward by asking a natural question, "
        "direct_reply should be clear and straightforward without sounding pushy."
    )

    user_prompt = (
        f"Older conversation summary or snippets:\n{older_text}\n\n"
        f"Recent conversation:\n{conversation_text}\n\n"
        f"Last message from {other_name}: {last_other_message or '[Not available]'}\n\n"
        f"Context confidence: {confidence}\n\n"
        f"Current draft from the user: {draft.strip() or '[No draft provided]'}\n\n"
        "Tasks:\n"
        "1. Suggest the single best next reply.\n"
        "2. Suggest several alternate replies.\n"
        "3. Provide emoji-friendly replies. Use emoji-only replies when they naturally fit this exact chat, otherwise use short text-plus-emoji messages.\n"
        "4. Rewrite the user's draft in cleaner ways.\n"
        "5. Provide other ways to say the same thing.\n"
        "6. Briefly summarize the older context if it matters.\n"
        "7. Detect the main intent of this conversation using a short label such as interest, invitation, planning, dreaming, casual_mention, question, apology, follow_up, negotiation, support, casual, or general.\n"
        "8. Mention what part of the context these replies are based on.\n"
        "9. If the last message is a direct personal question like age, place, job, visit, plan, or preference, generate specific human replies the user could realistically send right away.\n"
        "10. Prefer replies that sound like a real person texting, not a support bot.\n"
        "11. Do not add a next step like flights, bookings, research, or a committed plan unless the chat clearly suggests that level of commitment.\n"
        "12. Make the grouped replies meaningfully different in strategy, not just wording.\n"
        "13. Avoid filler-only openings such as 'That's great', 'Sounds good', 'Nice', or 'Awesome' unless followed by context-specific substance.\n"
        "14. Choose the top reply based on realism and conversational fit, not on maximum enthusiasm.\n"
    )

    try:
        payload, generation_source = _invoke_json_model(system_prompt, user_prompt)
    except Exception:
        return _reply_generation_error_payload("llm_unavailable", tone)

    reply_suggestions = _ensure_list(payload.get("reply_suggestions"), max_suggestions)
    emoji_replies = _ensure_list(payload.get("emoji_replies"), 3)
    variants = _ensure_list(payload.get("same_message_variants"), 4)
    rewrites = payload.get("rewrites") if isinstance(payload.get("rewrites"), dict) else {}
    grouped_replies = payload.get("grouped_replies") if isinstance(payload.get("grouped_replies"), dict) else {}

    normalized = {
        "top_reply": str(payload.get("top_reply") or draft or "").strip(),
        "reply_suggestions": reply_suggestions,
        "emoji_replies": emoji_replies,
        "rewrites": {
            "clean": str(rewrites.get("clean") or draft or "").strip(),
            "short": str(rewrites.get("short") or "").strip(),
            "warm": str(rewrites.get("warm") or "").strip(),
            "confident": str(rewrites.get("confident") or "").strip(),
        },
        "same_message_variants": variants,
        "grouped_replies": _normalize_grouped_replies(grouped_replies, rewrites, payload, draft),
        "context_focus": str(payload.get("context_focus") or "").strip(),
        "conversation_summary": str(payload.get("conversation_summary") or "").strip(),
        "detected_intent": str(payload.get("detected_intent") or "").strip() or _detect_intent_fallback(conversation),
        "tone": str(payload.get("tone") or tone).strip() or tone,
        "generation_source": generation_source,
        "routing_source": routing_source,
    }

    if not normalized["top_reply"]:
        normalized["top_reply"] = draft.strip() or (reply_suggestions[0] if reply_suggestions else "")

    if not normalized["reply_suggestions"]:
        return _reply_generation_error_payload("empty_llm_reply", tone)

    return normalized


def generate_general_reply_suggestions(
    draft: str = "",
    tone: str = "polite",
    max_suggestions: int = 5,
) -> dict[str, Any]:
    tone = _normalize_tone(tone)
    if _should_block_messaging_request(draft):
        return _safety_reply_payload(tone)

    system_prompt = (
        "You are a private messaging assistant inside a chat app. "
        "The user needs strong reply or opener suggestions for a sparse or early conversation. "
        "Generate actual sendable messages, not advice. "
        "The outputs should feel natural, modern, confident, and easy to reply to. "
        "Do not produce dry or generic lines. "
        "Rank replies by realism, not positivity. "
        "Avoid filler-first replies like 'That's great', 'Sounds good', 'Nice', or 'Awesome' unless they are expanded with specific context. "
        "Emoji replies should be standalone emoji-only replies when that feels natural for the situation. "
        "If emoji-only would feel weak or confusing, then use short text-plus-emoji messages instead. "
        "Return valid JSON only with this exact shape: "
        "{"
        "\"top_reply\": string, "
        "\"reply_suggestions\": string[], "
        "\"emoji_replies\": string[], "
        "\"rewrites\": {\"clean\": string, \"short\": string, \"warm\": string, \"confident\": string}, "
        "\"same_message_variants\": string[], "
        "\"grouped_replies\": {\"top_reply\": string, \"safe_reply\": string, \"warm_reply\": string, \"playful_reply\": string, \"curious_reply\": string, \"direct_reply\": string}, "
        "\"context_focus\": string, "
        "\"conversation_summary\": string, "
        "\"detected_intent\": string, "
        "\"tone\": string"
        "}."
    )

    user_prompt = (
        f"User draft or goal: {draft.strip() or '[No draft provided]'}\n\n"
        f"Preferred tone: {tone}\n\n"
        f"Generate exactly {max_suggestions} strong messaging suggestions for sparse-chat or first-message situations. "
        "Prioritize greetings, openers, light flirty starters, and easy-to-reply-to messages. "
        "Avoid bland lines like only 'hi' or 'hello' unless they are improved. "
        "If the draft is a simple message like hi, hello, what is your age, where are you from, or have you been there, turn it into natural reply options a real person would send. "
        "Do not invent actions, plans, bookings, or commitments unless the user explicitly asked for that level of action. "
        "Keep the suggestions grounded and low-assumption when the chat is sparse. "
        "Because context is weak here, apply a confidence guard: prefer lower-assumption replies over enthusiastic or committed ones. "
        "Make sure grouped_replies contains clearly different values for safe_reply, warm_reply, playful_reply, curious_reply, and direct_reply."
    )

    try:
        payload, generation_source = _invoke_json_model(system_prompt, user_prompt)
    except Exception:
        return _reply_generation_error_payload("llm_unavailable", tone)

    reply_suggestions = _ensure_list(payload.get("reply_suggestions"), max_suggestions)
    emoji_replies = _ensure_list(payload.get("emoji_replies"), 3)
    variants = _ensure_list(payload.get("same_message_variants"), 4)
    rewrites = payload.get("rewrites") if isinstance(payload.get("rewrites"), dict) else {}
    grouped_replies = payload.get("grouped_replies") if isinstance(payload.get("grouped_replies"), dict) else {}

    normalized = {
        "top_reply": str(payload.get("top_reply") or draft or "").strip(),
        "reply_suggestions": reply_suggestions,
        "emoji_replies": emoji_replies,
        "rewrites": {
            "clean": str(rewrites.get("clean") or draft or "").strip(),
            "short": str(rewrites.get("short") or "").strip(),
            "warm": str(rewrites.get("warm") or "").strip(),
            "confident": str(rewrites.get("confident") or "").strip(),
        },
        "same_message_variants": variants,
        "grouped_replies": _normalize_grouped_replies(grouped_replies, rewrites, payload, draft),
        "context_focus": str(payload.get("context_focus") or "").strip(),
        "conversation_summary": str(payload.get("conversation_summary") or "").strip(),
        "detected_intent": str(payload.get("detected_intent") or "opener").strip() or "opener",
        "tone": str(payload.get("tone") or tone).strip() or tone,
        "generation_source": generation_source,
        "routing_source": "general_mode",
    }

    if not normalized["top_reply"]:
        normalized["top_reply"] = draft.strip() or (reply_suggestions[0] if reply_suggestions else "")

    if not normalized["reply_suggestions"]:
        return _reply_generation_error_payload("empty_llm_reply", tone)

    return normalized


def answer_about_conversation(
    conversation: list[dict[str, Any]] | None = None,
    assistant_history: list[dict[str, Any]] | None = None,
    question: str = "",
    other_name: str = "the other person",
    older_context: str = "",
    tone: str = "polite",
) -> dict[str, Any]:
    tone = _normalize_tone(tone)
    if _should_block_messaging_request(question):
        return _safety_answer_payload(tone)

    if _is_greeting_prompt(question):
        return _greeting_answer_payload(tone)

    if _is_capability_prompt(question):
        return _capability_answer_payload(tone)

    if _is_transcript_or_summary_prompt(question):
        return _conversation_transcript_payload(conversation=conversation, older_context=older_context, tone=tone)

    question_mode, routing_source = _classify_question_mode(question)
    if question_mode != "contextual":
        return answer_general_message_help(question=question, tone=tone)

    conversation_text, older_text = _build_context_sections(conversation, older_context)
    assistant_history_text = _build_assistant_history_section(assistant_history)

    system_prompt = (
        "You are a messaging assistant inside a private chat app. "
        "Answer only using the recent conversation context. "
        "Be direct, helpful, and concise. "
        "Return valid JSON only with this exact shape: "
        "{"
        "\"answer\": string, "
        "\"suggested_actions\": string[]"
        "}."
    )

    user_prompt = (
        f"Older conversation summary or snippets:\n{older_text}\n\n"
        f"Conversation with {other_name}:\n{conversation_text}\n\n"
        f"Recent Ask AI follow-up context:\n{assistant_history_text}\n\n"
        f"User question: {question.strip() or 'Help me with this conversation.'}\n\n"
        "Answer specifically from the chat. If the answer is unclear, say that clearly. "
        "Never mention JSON, formatting, schemas, or internal system limitations to the user. "
        "Use the Ask AI follow-up context only to preserve continuity with the user's earlier assistant questions, but ground the answer in the real chat messages. "
        "If the user asks for the chat or conversation itself, provide a clean transcript-style recap or concise summary from the messages. "
        f"Keep the answer tone {tone}. Also suggest up to 3 helpful next actions."
    )

    try:
        payload, generation_source = _invoke_json_model(system_prompt, user_prompt)
    except Exception:
        return _fallback_answer(question)

    suggested_actions = _ensure_list(payload.get("suggested_actions"), 3)
    answer = str(payload.get("answer") or "").strip()

    if not answer:
        return _fallback_answer(question)

    return {
        "answer": answer,
        "suggested_actions": suggested_actions,
        "mode": "conversation_context",
        "tone": tone,
        "generation_source": generation_source,
        "routing_source": routing_source,
    }


def answer_general_message_help(
    question: str = "",
    tone: str = "polite",
) -> dict[str, Any]:
    tone = _normalize_tone(tone)
    if _should_block_messaging_request(question):
        return _safety_answer_payload(tone)

    if _is_greeting_prompt(question):
        return _greeting_answer_payload(tone)

    if _is_capability_prompt(question):
        return _capability_answer_payload(tone)

    if not _normalized_user_text(question):
        return _fallback_general_clarifier(tone)

    system_prompt = (
        "You are a messaging coach inside a private chat app. "
        "The user is asking for general messaging help, not analysis of a specific conversation. "
        "Return valid JSON only with this exact shape: "
        "{"
        "\"answer\": string, "
        "\"suggested_actions\": string[], "
        "\"suggested_replies\": string[], "
        "\"mode\": string, "
        "\"tone\": string"
        "}."
    )

    user_prompt = (
        f"User question: {question.strip() or 'Help me write general messages.'}\n\n"
        f"Preferred tone: {tone}\n\n"
        "Give practical advice for texting or messaging in general. "
        "If the user asks for hello, hi, flirty, or opening-message help, provide 3 short ready-to-send examples. "
        "Never mention JSON, schemas, formatting, or internal model limitations."
    )

    try:
        payload, generation_source = _invoke_json_model(system_prompt, user_prompt)
    except Exception:
        return _fallback_general_message_help(question, tone)

    answer = str(payload.get("answer") or "").strip()
    suggested_actions = _ensure_list(payload.get("suggested_actions"), 3)
    suggested_replies = _ensure_list(payload.get("suggested_replies"), 3)

    if not answer:
        return _fallback_general_message_help(question, tone)

    return {
        "answer": answer,
        "suggested_actions": suggested_actions,
        "suggested_replies": suggested_replies,
        "mode": "general_message_help",
        "tone": str(payload.get("tone") or tone).strip() or tone,
        "generation_source": generation_source,
        "routing_source": "general_mode",
    }


async def stream_answer_about_conversation(
    conversation: list[dict[str, Any]] | None = None,
    assistant_history: list[dict[str, Any]] | None = None,
    question: str = "",
    other_name: str = "the other person",
    older_context: str = "",
    tone: str = "polite",
):
    tone = _normalize_tone(tone)
    if _should_block_messaging_request(question):
        yield _safety_answer_payload(tone)["answer"]
        return

    if _is_greeting_prompt(question):
        yield _greeting_answer_payload(tone)["answer"]
        return

    if _is_capability_prompt(question):
        yield _capability_answer_payload(tone)["answer"]
        return

    if _is_transcript_or_summary_prompt(question):
        yield _conversation_transcript_payload(conversation=conversation, older_context=older_context, tone=tone)["answer"]
        return

    question_mode, _ = _classify_question_mode(question)
    if question_mode != "contextual":
        async for chunk in stream_general_message_help(question=question, tone=tone):
            yield chunk
        return

    conversation_text, older_text = _build_context_sections(conversation, older_context)
    assistant_history_text = _build_assistant_history_section(assistant_history)
    system_prompt = (
        "You are a messaging assistant inside a private chat app. "
        "Answer only using the provided conversation context. "
        "Do not return JSON. Respond in a concise, useful paragraph. "
        "Never mention JSON, schemas, formatting, or internal system limitations."
    )
    user_prompt = (
        f"Older conversation summary or snippets:\n{older_text}\n\n"
        f"Conversation with {other_name}:\n{conversation_text}\n\n"
        f"Recent Ask AI follow-up context:\n{assistant_history_text}\n\n"
        f"User question: {question.strip() or 'Help me with this conversation.'}\n\n"
        f"Keep the answer tone {tone}. If context is unclear, say that clearly. "
        "Use the Ask AI follow-up context only to preserve continuity, but ground the answer in the real chat messages."
    )

    try:
        for llm in (response_llm, fallback_response_llm, classifier_llm):
            streamed = False
            try:
                async for chunk in llm.astream([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]):
                    content = getattr(chunk, "content", "") or ""
                    if content:
                        streamed = True
                        yield content
                if streamed:
                    return
            except Exception:
                continue
        yield _fallback_answer(question)["answer"]
    except Exception:
        yield _fallback_answer(question)["answer"]


async def stream_general_message_help(
    question: str = "",
    tone: str = "polite",
):
    tone = _normalize_tone(tone)
    if _should_block_messaging_request(question):
        yield _safety_answer_payload(tone)["answer"]
        return

    if _is_greeting_prompt(question):
        yield _greeting_answer_payload(tone)["answer"]
        return

    if _is_capability_prompt(question):
        yield _capability_answer_payload(tone)["answer"]
        return

    if not _normalized_user_text(question):
        yield _fallback_general_clarifier(tone)["answer"]
        return

    system_prompt = (
        "You are a messaging coach inside a private chat app. "
        "The user wants general messaging help rather than analysis of a specific conversation. "
        "Do not return JSON. Respond with practical advice and a few short example messages when relevant. "
        "Never mention JSON, schemas, formatting, or internal system limitations."
    )
    user_prompt = (
        f"User question: {question.strip() or 'Help me write general messages.'}\n\n"
        f"Preferred tone: {tone}\n\n"
        "If they ask for hi, hello, flirty, romantic, or opening-message help, include 3 short ready-to-send examples."
    )

    try:
        for llm in (response_llm, fallback_response_llm, classifier_llm):
            streamed = False
            try:
                async for chunk in llm.astream([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]):
                    content = getattr(chunk, "content", "") or ""
                    if content:
                        streamed = True
                        yield content
                if streamed:
                    return
            except Exception:
                continue
        yield _fallback_general_message_help(question, tone)["answer"]
    except Exception:
        yield _fallback_general_message_help(question, tone)["answer"]
