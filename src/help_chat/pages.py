"""Knows the CliQ pages: which ones the assistant may link to and what the user's current page is."""

import re

# Routes the assistant may link to. Mirrors the <Route> paths in website_Dev/vite-project/src/App.jsx and the
# INTERNAL_LINK_ALLOWLIST in website_Dev/vite-project/src/utils/aiUtils.js — keep all three in sync.
LINKABLE_PAGES = [
    ("/home", "Home", "home"),
    ("/search", "Search", "find"),
    ("/find/getconnection", "Explore (Get Connections)", "find"),
    ("/requests", "Requests", "requests"),
    ("/create/post", "Create Post", "create"),
    ("/edit-profile", "Edit Profile", "profile"),
    ("/settings", "Settings", "account"),
    ("/messages", "Messages", "messages"),
    ("/my-experties", "Expertise Studio", "profile"),
    ("/profile", "Profile", "profile"),
]

# Pages users can be on that are not link targets, matched by prefix
OTHER_PAGES = [
    ("/post/", "a single post's page", "create"),
    ("/notifications", "Notifications", None),
    ("/saved", "Saved Posts", None),
    ("/public-profile/", "another user's public profile", "profile"),
    ("/user/", "another user's public profile", "profile"),
]

PAGE_REFERENCE_RE = re.compile(
    r"\b(this page|this screen|on here|from here|is page|iss page|ye page|yeh page|yaha|yahan)\b",
    re.IGNORECASE,
)
# Questions whose answer depends on this user's data; their answers must never be cached
PERSONAL_RE = re.compile(
    r"\b(do i have|have i got|how many|kitne|kitni|pending|unread|my (requests|notifications|expertise|stats)|"
    r"(mere|meri) (requests|notifications)|who am i|my name)\b",
    re.IGNORECASE,
)


def describe_page(path: object) -> tuple[str | None, str | None]:
    """Returns (page name, KB section) for the user's current path, or (None, None)."""
    if not isinstance(path, str):
        return None, None
    path = path.split("?")[0].split("#")[0][:200]
    if not re.fullmatch(r"[\w\-/]*", path):
        return None, None
    for route, name, section in LINKABLE_PAGES:
        if path == route or path.startswith(route + "/"):
            return f"{name} ({route})", section
    for prefix, name, section in OTHER_PAGES:
        if path.startswith(prefix):
            return name, section
    return None, None


def refers_to_page(message: str | None) -> bool:
    return bool(PAGE_REFERENCE_RE.search(message or ""))


def is_personal(message: str | None) -> bool:
    return bool(PERSONAL_RE.search(message or ""))


def clean_name(value: object) -> str:
    # Only letters and spaces reach the prompt, so a crafted name can't inject instructions
    first_line = str(value or "").strip().splitlines()[0] if str(value or "").strip() else ""
    name = re.sub(r"[^\w \-']", "", first_line)[:40].strip()
    return re.sub(r"\d|_", "", name).strip()


def _count(value: object) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def build_extra_system(user_context: dict | None, current_path: str | None) -> str:
    """System-prompt block with the user context and deep-link rules. Never contains emails, passwords or IDs."""
    user_context = user_context or {}
    lines = ["User context (facts about the person you are helping):"]

    name = clean_name(user_context.get("firstname"))
    if name:
        lines.append(f"- First name: {name}")
    page, _ = describe_page(current_path)
    if page:
        lines.append(f"- Current page: {page}")
    pending = _count(user_context.get("pendingRequests"))
    if pending is not None:
        lines.append(f"- Pending incoming connection requests: {pending}")
    unread = _count(user_context.get("unreadNotifications"))
    if unread is not None:
        lines.append(f"- Unread notifications: {unread}")
    if isinstance(user_context.get("expertiseFilled"), bool):
        lines.append(f"- Expertise profile filled in: {'yes' if user_context['expertiseFilled'] else 'no'}")

    lines.append(
        "Use these facts only when the question is about the user's own account, requests, notifications "
        'or the page they are on (e.g. "You have 3 pending requests"). '
        '"This page" or "here" means the current page.'
    )

    links = "\n".join(f"- [Open {name}]({route})" for route, name, _ in LINKABLE_PAGES)
    lines.append(
        "\nPage links (verified routes, allowed even though they are not in the reference text):\n"
        "After a part's answer you may add at most one link, on its own last line, to the page that part is about, "
        "copied exactly from this list:\n"
        f"{links}\n"
        "Never write any other link or URL."
    )
    return "\n".join(lines)
