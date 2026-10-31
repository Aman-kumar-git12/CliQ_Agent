"""Reads the knowledge base files and splits them into one chunk per "## " section."""

import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

FEATURE_RE = re.compile(r"^#\s+Feature:\s*(.+?)\s*$", re.MULTILINE)
ROUTE_RE = re.compile(r"^Route:\s*(.+?)\s*$", re.MULTILINE)
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# bge-small truncates input past 512 tokens (~1,800 characters of English);
# anything after that would be silently ignored when embedding.
MAX_CHUNK_CHARS = 1800


def list_sections(data_dir: Path = DATA_DIR) -> list[str]:
    """Section names are the file names, so a new file becomes a new section automatically."""
    return sorted(path.stem for path in data_dir.glob("*.txt"))


def kb_fingerprint(data_dir: Path = DATA_DIR) -> str:
    """Changes whenever a knowledge file changes, so votes and caches tied to old docs stop counting."""
    digest = hashlib.sha1()
    for path in sorted(data_dir.glob("*.txt")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def split_file(text: str, section: str, source: str) -> list[Document]:
    """Splits one file written in the template below into heading-level chunks.

        # Feature: <Name>
        Route: <path(s)>
        ## <heading>
        <body>

    Each chunk starts with "Feature: <Name> — <heading>" plus the route, so the
    embedding and the LLM both know which feature a fragment belongs to.
    """
    feature_match = FEATURE_RE.search(text)
    route_match = ROUTE_RE.search(text)
    feature = feature_match.group(1) if feature_match else section.replace("_", " ").title()
    route = route_match.group(1) if route_match else "n/a"

    headings = list(HEADING_RE.finditer(text))
    chunks = []
    for i, match in enumerate(headings):
        heading = match.group(1)
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            continue

        content = f"Feature: {feature} — {heading}\nRoute: {route}\n\n{body}"
        if len(content) > MAX_CHUNK_CHARS:
            logger.warning(
                "%s › '%s' is %d chars; text past ~%d is not embedded. Split it into smaller sections.",
                source,
                heading,
                len(content),
                MAX_CHUNK_CHARS,
            )
        metadata = {"source": source, "section": section, "feature": feature, "heading": heading, "route": route}
        chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Loads every .txt file in the data directory as heading-level chunks."""
    if not data_dir.exists():
        logger.warning("Knowledge base directory %s does not exist", data_dir)
        return []

    documents = []
    for path in sorted(data_dir.glob("*.txt")):
        chunks = split_file(path.read_text(encoding="utf-8"), path.stem, path.name)
        if not chunks:
            logger.warning("%s has no '## ' sections; skipped", path.name)
            continue
        documents.extend(chunks)
        logger.debug("Loaded %s (%d chunks)", path.name, len(chunks))
    return documents
