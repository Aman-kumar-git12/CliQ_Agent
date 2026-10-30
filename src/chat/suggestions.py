import re

from ..rag.loader import CliQDocumentLoader

STARTER_PROMPTS = [
    "How do I find people?",
    "How do I create a post?",
    "How do smart connections work?",
    "I forgot my password",
]

MAX_SUGGESTIONS = 3
MAX_CHARS = 48

# Headings that describe internals or limits make poor follow-up questions
SKIP_HEADING_RE = re.compile(
    r"^(overview|summary|limits|important limits|main sections|main pages|main product areas|"
    r"common problems.*|.*problems|features not available yet|other .*items|connection states.*|"
    r"real-time behavior|request behavior|after acceptance|feed behavior|what the ai assistant is for|"
    r"where to find things|where a notification takes you|what people search matches)$",
    re.IGNORECASE,
)


def heading_to_question(heading):
    """Turns a KB heading into a short user question, e.g. "How to edit a post" → "How do I edit a post?"."""
    text = re.sub(r"\s*\(.*?\)\s*", " ", heading).strip()
    text = re.sub(r"\byour\b", "my", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou sent\b", "I sent", text, flags=re.IGNORECASE)
    if m := re.match(r"how to (.+)", text, re.IGNORECASE):
        return f"How do I {m.group(1)}?"
    if m := re.match(r"how (.+) works?$", text, re.IGNORECASE):
        subject = m.group(1)
        verb = "does" if text.lower().endswith("works") else "do"
        return f"How {verb} {subject} work?"
    if m := re.match(r"how users (.+)", text, re.IGNORECASE):
        return f"How do I {m.group(1)}?"
    if m := re.match(r"what users can (.+)", text, re.IGNORECASE):
        return f"What can I {m.group(1)}?"
    if m := re.match(r"what (the .+) shows$", text, re.IGNORECASE):
        return f"What does {m.group(1)} show?"
    if m := re.match(r"what you can (.+)", text, re.IGNORECASE):
        return f"What can I {m.group(1)}?"
    if m := re.match(r"what (.+?) can(not)? (.+)", text, re.IGNORECASE):
        return f"What can{m.group(2) or ''} {m.group(1)} {m.group(3)}?"
    if m := re.match(r"what you get (.+)", text, re.IGNORECASE):
        return f"What do I get {m.group(1)}?"
    if m := re.match(r"forgot (.+)", text, re.IGNORECASE):
        return f"I forgot my {m.group(1)}"
    if m := re.match(r"(.+) blocked$", text, re.IGNORECASE):
        return f"What if my {m.group(1).lower()} is blocked?"
    if re.match(r"(change|send|create|edit|delete|start|find|reset)\b", text, re.IGNORECASE):
        return f"How do I {text[0].lower()}{text[1:]}?"
    if re.match(r"(what|where|can|why|who)\b", text, re.IGNORECASE):
        return text.rstrip("?") + "?"
    # Keep feature names like "Get Connections" capitalised; lowercase plain phrases
    first, _, rest = text.partition(" ")
    if not any(w[:1].isupper() for w in rest.split()) and not first.isupper():
        text = first.lower() + (" " + rest if rest else "")
    return f"Tell me about {text}"


def _words(text):
    return set(re.findall(r"[a-z]+", text.lower())) - {"how", "do", "i", "a", "the", "to", "my", "tell", "me", "about"}


class FollowUpSuggester:
    """Builds follow-up chips from KB headings, without an extra LLM call."""

    def __init__(self):
        self.headings_by_section = {}
        for doc in CliQDocumentLoader().load_documents():
            self.headings_by_section.setdefault(doc.metadata["section"], []).append(doc.metadata["heading"])

    def suggest(self, question, docs_by_part):
        answered = {docs[0].metadata.get("heading") for docs in docs_by_part if docs}
        if not answered:
            return [p for p in STARTER_PROMPTS if p.lower() != (question or "").strip().lower()][:MAX_SUGGESTIONS]

        candidates = []
        # 1. The headings right after each answered one in its own file ("what's next")
        for docs in docs_by_part:
            if not docs:
                continue
            section, heading = docs[0].metadata.get("section"), docs[0].metadata.get("heading")
            headings = self.headings_by_section.get(section, [])
            if heading in headings:
                i = headings.index(heading)
                candidates += headings[i + 1:] + headings[:i]
        # 2. Related headings that retrieval found in this and other sections
        for docs in docs_by_part:
            candidates += [d.metadata.get("heading") for d in docs[1:]]

        asked = _words(question or "")
        items, seen = [], set()
        for heading in candidates:
            if not heading or heading in answered or heading in seen or SKIP_HEADING_RE.match(heading):
                continue
            seen.add(heading)
            item = heading_to_question(heading)
            words = _words(item)
            if len(item) > MAX_CHARS or (words and len(words & asked) / len(words) > 0.6):
                continue
            items.append(item)
            if len(items) == MAX_SUGGESTIONS:
                break
        return items
