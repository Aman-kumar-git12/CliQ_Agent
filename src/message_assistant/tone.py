"""Tone and emoji preference normalisation, plus the matching prompt instructions."""

from .prompts import EMOJI_INSTRUCTIONS, TONE_INSTRUCTIONS

DEFAULT_TONE = "polite"
ALLOWED_TONES = frozenset(TONE_INSTRUCTIONS)


def normalize_tone(tone: str | None = DEFAULT_TONE) -> str:
    """Lower-cases the tone and falls back to polite for anything unknown."""
    normalized = str(tone or DEFAULT_TONE).strip().lower()
    return normalized if normalized in ALLOWED_TONES else DEFAULT_TONE


def get_tone_instruction(tone: str | None) -> str:
    """Prompt line describing how the chosen tone should sound."""
    normalized = normalize_tone(tone)
    return f"Tone instruction for '{normalized}': {TONE_INSTRUCTIONS[normalized]}"


def get_emoji_instruction(emoji_preference: str | None = "both") -> str:
    """Prompt line describing the emoji preference; unknown values keep their label but use the 'both' text."""
    normalized = str(emoji_preference or "both").strip().lower()
    instruction = EMOJI_INSTRUCTIONS.get(normalized, EMOJI_INSTRUCTIONS["both"])
    return f"Emoji instruction for '{normalized}': {instruction}"
