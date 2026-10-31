"""Text cleaning, term lists and tokenization shared by profile features, scoring and search."""

import re
from typing import Any

# Values the frontend stores for "not filled in"; they must not count as profile signal
PLACEHOLDER_VALUES = {
    "",
    "****",
    "professional seeker",
    "n/a",
    "na",
    "null",
    "undefined",
}
TERM_SEPARATORS = re.compile(r"[,|\n;/]+")
# Keeps "+", "#" and "." so tokens like c++, c# and node.js survive
TOKEN_SEPARATORS = re.compile(r"[^a-z0-9+#.]+")
MIN_TOKEN_LENGTH = 3


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_meaningful_text(value: Any) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return False
    return normalized.lower() not in PLACEHOLDER_VALUES


def normalize_terms(value: Any) -> list[str]:
    """Turns a list or a separated string into meaningful terms, deduplicated case-insensitively."""
    raw_items = value if isinstance(value, list) else TERM_SEPARATORS.split(normalize_text(value))
    deduped: dict[str, str] = {}
    for item in raw_items:
        text = normalize_text(item)
        if not is_meaningful_text(text):
            continue
        key = text.lower()
        if key not in deduped:
            deduped[key] = text
    return list(deduped.values())


def merge_unique_terms(*value_groups: Any) -> list[str]:
    """Concatenates term groups, keeping the first spelling of each case-insensitive term."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in value_groups:
        for item in normalize_terms(group):
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(item)
    return merged


def tokenize(values: list[Any]) -> list[str]:
    """Sorted unique lower-case tokens of at least three characters."""
    tokens: set[str] = set()
    for value in values:
        text = normalize_text(value).lower()
        for token in TOKEN_SEPARATORS.split(text):
            cleaned = token.strip()
            if len(cleaned) >= MIN_TOKEN_LENGTH:
                tokens.add(cleaned)
    return sorted(tokens)


def shared_items(left: list[str], right: list[str]) -> list[str]:
    """Items of `left` that also appear in `right`, compared case-insensitively, in `left` order."""
    right_lookup = {item.lower(): item for item in right}
    return [item for item in left if item.lower() in right_lookup]
