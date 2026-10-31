"""Pulls a JSON object out of free-form model output and cleans list fields."""

import json
import re
from typing import Any

FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    """Parses the text as JSON, returning it only if it is an object."""
    text = (candidate or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_balanced_json_candidates(raw_text: str) -> list[str]:
    """Top-level {...} spans with balanced braces, ignoring braces inside JSON strings."""
    candidates: list[str] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(raw_text):
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = raw_text[start_index : index + 1].strip()
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
                start_index = None

    return candidates


def _first_decodable_object(raw_text: str) -> dict[str, Any] | None:
    """First object that decodes from a '{' and is followed only by whitespace or a closing fence."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, end_index = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            trailing = raw_text[index + end_index :].strip()
            if not trailing or trailing.startswith("```"):
                return parsed
    return None


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Finds the JSON object in a model reply, trying progressively looser strategies.

    Raises ValueError when the reply is empty or contains no JSON object.
    """
    if not raw_text:
        raise ValueError("Empty assistant response")

    direct_match = try_parse_json_object(raw_text.strip())
    if direct_match is not None:
        return direct_match

    for block in FENCED_JSON_PATTERN.findall(raw_text):
        parsed = try_parse_json_object(block)
        if parsed is not None:
            return parsed

    decoded = _first_decodable_object(raw_text)
    if decoded is not None:
        return decoded

    for candidate in extract_balanced_json_candidates(raw_text):
        parsed = try_parse_json_object(candidate)
        if parsed is not None:
            return parsed

    raise ValueError("Assistant response did not contain valid JSON")


def ensure_list(value: Any, limit: int) -> list[str]:
    """Stripped, de-duplicated, non-empty strings from a list value, capped at `limit`."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned
