"""Formats and trims chat turns into prompt context and derives simple signals from them."""

from typing import Any

from . import prompts
from .tone import get_emoji_instruction, get_tone_instruction

type Turn = dict[str, Any]

# Only the most recent turns go into the prompt, to keep it short and focused on what matters now.
MAX_RECENT_TURNS = 20
MAX_ASSISTANT_HISTORY_TURNS = 12

NO_PRIOR_MESSAGES = "No prior messages."
NO_OLDER_CONTEXT = "No older context available."
NO_ASSISTANT_HISTORY = "No prior Ask AI follow-up context."

# Placeholders for media turns that carry no text.
EMPTY_TURN_PLACEHOLDERS = {"image": "[Image]", "voice": "[Voice message]", "file": "[File]"}

INVITATION_KEYWORDS = ("invite", "meet", "go out", "hang out", "date", "dinner", "coffee", "see you")
APOLOGY_KEYWORDS = ("sorry", "apologize", "forgive")


def turn_text(turn: Turn) -> str:
    """The stripped text of a turn, treating None as empty."""
    return (turn.get("text") or "").strip()


def normalize_turn(turn: Turn) -> str:
    """Renders one turn as 'You: ...' or 'Other person: ...'."""
    role = "You" if turn.get("role") == "me" else "Other person"
    text = turn_text(turn)
    if not text:
        kind = (turn.get("kind") or "text").strip().lower()
        text = EMPTY_TURN_PLACEHOLDERS.get(kind, "[Empty]")
    return f"{role}: {text}"


def build_context_sections(conversation: list[Turn] | None = None, older_context: str = "") -> tuple[str, str]:
    """Returns (recent conversation transcript, older context) with placeholders when empty."""
    recent_turns = (conversation or [])[-MAX_RECENT_TURNS:]
    conversation_text = "\n".join(normalize_turn(turn) for turn in recent_turns) or NO_PRIOR_MESSAGES
    older_text = older_context.strip() or NO_OLDER_CONTEXT
    return conversation_text, older_text


def build_assistant_history_section(assistant_history: list[Turn] | None = None) -> str:
    """Renders the recent Ask AI exchange so follow-up questions keep their context."""
    lines = []
    for turn in (assistant_history or [])[-MAX_ASSISTANT_HISTORY_TURNS:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) if lines else NO_ASSISTANT_HISTORY


def get_last_other_person_message(conversation: list[Turn] | None = None) -> str:
    """Text of the latest non-empty turn not sent by the user."""
    for turn in reversed(conversation or []):
        if turn.get("role") != "me":
            text = turn_text(turn)
            if text:
                return text
    return ""


def meaningful_turn_count(conversation: list[Turn] | None) -> int:
    """Number of turns that contain text."""
    return sum(1 for turn in conversation or [] if turn_text(turn))


def context_confidence(conversation: list[Turn] | None = None, older_context: str = "") -> str:
    """'high', 'medium' or 'low': how much real context the model has to ground replies in."""
    turns = conversation or []
    meaningful = meaningful_turn_count(turns)
    last_other = get_last_other_person_message(turns)
    if meaningful >= 4 and len(last_other) >= 12:
        return "high"
    if meaningful >= 2 or older_context.strip():
        return "medium"
    return "low"


def detect_intent_fallback(conversation: list[Turn] | None = None) -> str:
    """Keyword guess at the conversation intent, used when the model does not return one."""
    turns = conversation or []
    if not turns:
        return "opener"
    last_text = next((turn_text(turn).lower() for turn in reversed(turns) if turn_text(turn)), "")
    if not last_text:
        return "general"
    if "?" in last_text:
        return "question"
    if any(keyword in last_text for keyword in INVITATION_KEYWORDS):
        return "invitation"
    if any(keyword in last_text for keyword in APOLOGY_KEYWORDS):
        return "apology"
    return "casual"


def build_contextual_reply_prompts(
    conversation: list[Turn] | None,
    draft: str,
    other_name: str,
    older_context: str,
    tone: str,
    max_suggestions: int,
    emoji_preference: str | None,
) -> tuple[str, str]:
    """(system, user) prompts for reply suggestions grounded in this conversation."""
    conversation_text, older_text = build_context_sections(conversation, older_context)
    system_prompt = prompts.contextual_reply_system_prompt(
        max_suggestions, other_name, tone, get_tone_instruction(tone), get_emoji_instruction(emoji_preference)
    )
    user_prompt = prompts.contextual_reply_user_prompt(
        older_text,
        conversation_text,
        other_name,
        get_last_other_person_message(conversation),
        context_confidence(conversation, older_context),
        draft,
    )
    return system_prompt, user_prompt
