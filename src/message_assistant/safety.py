"""Keyword-based safety screen for coercive, abusive or exploitative messaging requests."""

SAFETY_BLOCK_PATTERNS = (
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
)


def join_request_parts(*parts: str) -> str:
    """Joins the non-empty request parts with single spaces, as the safety checks see them."""
    return " ".join(part.strip() for part in parts if part).strip()


def contains_unsafe_messaging_request(*parts: str) -> bool:
    """True when any part contains a blocked phrase (case-insensitive)."""
    text = join_request_parts(*parts).lower()
    if not text:
        return False
    return any(pattern in text for pattern in SAFETY_BLOCK_PATTERNS)
