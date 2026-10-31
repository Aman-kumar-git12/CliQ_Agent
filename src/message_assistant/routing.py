"""Rule-based routing: recognises greetings, capability and transcript questions, and general-vs-contextual."""

from .conversation import Turn, turn_text

GREETING_PROMPTS = frozenset({"hi", "hii", "hiii", "hello", "hey", "heyy", "good morning", "good evening"})

CAPABILITY_MARKERS = (
    "what can you do",
    "what will you do",
    "how can you help",
    "what do you do",
    "help me",
    "how do you help",
)

TRANSCRIPT_MARKERS = (
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
)

CONTEXTUAL_QUESTION_MARKERS = (
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
)

GENERAL_QUESTION_MARKERS = (
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
)

# Words that suggest general opener help, unless the user explicitly asks for a reply.
GENERAL_QUESTION_WORDS = ("flirty", "romantic", "cute", "greeting", "hello", "hi")

OPENER_DRAFT_MARKERS = (
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
)


def normalized_user_text(text: str | None = "") -> str:
    """Lower-cased text with runs of whitespace collapsed to single spaces."""
    return " ".join(str(text or "").strip().lower().split())


def is_greeting_prompt(question: str = "") -> bool:
    """True for a bare greeting like 'hi' or 'good morning'."""
    return normalized_user_text(question) in GREETING_PROMPTS


def is_capability_prompt(question: str = "") -> bool:
    """True when the user asks what the assistant can do."""
    normalized = normalized_user_text(question)
    return bool(normalized) and any(marker in normalized for marker in CAPABILITY_MARKERS)


def is_transcript_or_summary_prompt(question: str = "") -> bool:
    """True when the user asks to see or summarise the chat itself."""
    normalized = normalized_user_text(question)
    return bool(normalized) and any(marker in normalized for marker in TRANSCRIPT_MARKERS)


def is_contextual_chat_question(question: str = "") -> bool:
    """Heuristic fallback for the question-mode classifier; ambiguous questions count as contextual."""
    normalized = (question or "").strip().lower()
    if not normalized:
        return True
    if any(marker in normalized for marker in CONTEXTUAL_QUESTION_MARKERS):
        return True
    if any(marker in normalized for marker in GENERAL_QUESTION_MARKERS):
        return False
    return not (any(word in normalized for word in GENERAL_QUESTION_WORDS) and "reply" not in normalized)


def should_use_general_reply_mode(conversation: list[Turn] | None = None, draft: str = "") -> bool:
    """Heuristic fallback for the reply-mode classifier: general mode only for openers and sparse chats."""
    turns = conversation or []
    # Any real message from the other person means there is something concrete to reply to.
    has_other_person_message = any(
        (turn.get("role") or "").strip().lower() == "other" and turn_text(turn) for turn in turns
    )
    if has_other_person_message:
        return False

    meaningful_turns = sum(1 for turn in turns if turn_text(turn))
    normalized_draft = (draft or "").strip().lower()
    if not meaningful_turns and not normalized_draft:
        return True
    if any(marker in normalized_draft for marker in OPENER_DRAFT_MARKERS):
        return True
    return meaningful_turns <= 1
