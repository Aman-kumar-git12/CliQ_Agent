"""Turns the model's reply-suggestion JSON into the stable response shape the backend expects."""

from typing import Any

from .json_output import ensure_list

EMOJI_REPLY_LIMIT = 3
VARIANT_LIMIT = 4


def _first_text(*values: Any) -> str:
    """The first truthy value as a stripped string, or ''."""
    return str(next((value for value in values if value), "")).strip()


def normalize_grouped_replies(
    grouped_replies: dict[str, Any] | None = None,
    rewrites: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    draft: str = "",
) -> dict[str, str]:
    """Fills every grouped-reply slot, borrowing from legacy keys and rewrites when the model left one out."""
    grouped = grouped_replies if isinstance(grouped_replies, dict) else {}
    rewrites = rewrites if isinstance(rewrites, dict) else {}
    payload = payload if isinstance(payload, dict) else {}

    safe_reply = _first_text(
        grouped.get("safe_reply"), grouped.get("opener"), grouped.get("short_reply"), rewrites.get("clean"), draft
    )
    playful_reply = _first_text(grouped.get("playful_reply"), grouped.get("follow_up"))
    curious_reply = _first_text(grouped.get("curious_reply"), grouped.get("follow_up"))
    direct_reply = _first_text(grouped.get("direct_reply"), grouped.get("confident_reply"), rewrites.get("confident"))
    return {
        "top_reply": _first_text(grouped.get("top_reply"), payload.get("top_reply"), draft),
        "safe_reply": safe_reply,
        "warm_reply": _first_text(grouped.get("warm_reply"), rewrites.get("warm")),
        "playful_reply": playful_reply,
        "curious_reply": curious_reply,
        "direct_reply": direct_reply,
        "opener": _first_text(grouped.get("opener"), safe_reply),
        "follow_up": _first_text(grouped.get("follow_up"), curious_reply, playful_reply),
        "short_reply": _first_text(grouped.get("short_reply"), rewrites.get("short"), safe_reply),
        "confident_reply": _first_text(grouped.get("confident_reply"), direct_reply),
    }


def _dict_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _emoji_lists(payload: dict[str, Any]) -> dict[str, list[str]]:
    """The three parallel emoji suggestion lists, each falling back to the others when the model left it out."""
    with_emojis = ensure_list(payload.get("emoji_replies_with_emojis"), EMOJI_REPLY_LIMIT)
    without_emojis = ensure_list(payload.get("emoji_replies_without_emojis"), EMOJI_REPLY_LIMIT)
    emoji_replies = ensure_list(payload.get("emoji_replies"), EMOJI_REPLY_LIMIT) or with_emojis
    return {
        "emoji_replies_with_emojis": with_emojis or emoji_replies,
        "emoji_replies_without_emojis": without_emojis or emoji_replies,
        "emoji_replies": emoji_replies,
    }


def _normalize_rewrites(rewrites: dict[str, Any], draft: str) -> dict[str, str]:
    return {
        "clean": _first_text(rewrites.get("clean"), draft),
        "short": _first_text(rewrites.get("short")),
        "warm": _first_text(rewrites.get("warm")),
        "confident": _first_text(rewrites.get("confident")),
    }


def normalize_reply_payload(
    payload: dict[str, Any],
    *,
    draft: str,
    tone: str,
    max_suggestions: int,
    emoji_preference: str | None,
    fallback_intent: str,
    generation_source: str,
    routing_source: str,
) -> dict[str, Any] | None:
    """Builds the reply response from model JSON; returns None when the model produced no suggestions."""
    reply_suggestions = ensure_list(payload.get("reply_suggestions"), max_suggestions)
    if not reply_suggestions:
        return None

    rewrites = _dict_field(payload, "rewrites")
    normalized = {
        "top_reply": _first_text(payload.get("top_reply"), draft) or draft.strip() or reply_suggestions[0],
        "reply_suggestions": reply_suggestions,
        **_emoji_lists(payload),
        "rewrites": _normalize_rewrites(rewrites, draft),
        "same_message_variants": ensure_list(payload.get("same_message_variants"), VARIANT_LIMIT),
        "grouped_replies": normalize_grouped_replies(_dict_field(payload, "grouped_replies"), rewrites, payload, draft),
        "context_focus": _first_text(payload.get("context_focus")),
        "conversation_summary": _first_text(payload.get("conversation_summary")),
        "detected_intent": _first_text(payload.get("detected_intent")) or fallback_intent,
        "tone": _first_text(payload.get("tone"), tone) or tone,
        "generation_source": generation_source,
        "routing_source": routing_source,
    }

    _apply_emoji_preference(normalized, emoji_preference)
    return normalized


def _apply_emoji_preference(normalized: dict[str, Any], emoji_preference: str | None) -> None:
    """Shows the preferred parallel emoji list as the main suggestions, when the model provided one."""
    if emoji_preference == "with" and normalized["emoji_replies_with_emojis"]:
        normalized["reply_suggestions"] = normalized["emoji_replies_with_emojis"]
    elif emoji_preference == "without" and normalized["emoji_replies_without_emojis"]:
        normalized["reply_suggestions"] = normalized["emoji_replies_without_emojis"]
