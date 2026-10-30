import re
from pathlib import Path
from langchain_core.documents import Document

FEATURE_RE = re.compile(r"^#\s+Feature:\s*(.+?)\s*$", re.MULTILINE)
ROUTE_RE = re.compile(r"^Route:\s*(.+?)\s*$", re.MULTILINE)
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# bge-small truncates input past 512 tokens (~1,800 characters of English);
# anything after that would be silently ignored when embedding.
MAX_CHUNK_CHARS = 1800


class CliQDocumentLoader:
    """Loads the CliQ knowledge base and splits it into one chunk per "## " section.

    Every file follows the same template:
        # Feature: <Name>
        Route: <path(s)>
        ## <heading>
        <body>
    Each chunk starts with "Feature: <Name> — <heading>" plus the route, so the
    embedding and the LLM both know which feature a fragment belongs to.
    """

    def __init__(self, data_dir: str = None):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else self.base_dir / "data"

    @staticmethod
    def split_file(text: str, section: str, source: str):
        feature_match = FEATURE_RE.search(text)
        route_match = ROUTE_RE.search(text)
        feature = feature_match.group(1) if feature_match else section.replace("_", " ").title()
        route = route_match.group(1) if route_match else "n/a"

        headings = list(HEADING_RE.finditer(text))
        chunks = []
        for i, match in enumerate(headings):
            heading = match.group(1)
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[match.end():end].strip()
            if not body:
                continue

            content = f"Feature: {feature} — {heading}\nRoute: {route}\n\n{body}"
            if len(content) > MAX_CHUNK_CHARS:
                print(f"Warning: {source} › '{heading}' is {len(content)} chars; "
                      f"text past ~{MAX_CHUNK_CHARS} is not embedded. Split it into smaller sections.")
            chunks.append(Document(
                page_content=content,
                metadata={
                    "source": source,
                    "section": section,
                    "feature": feature,
                    "heading": heading,
                    "route": route,
                },
            ))
        return chunks

    def load_documents(self):
        """Loads all .txt files from the data directory as heading-level chunks."""
        documents = []
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return documents

        for file_path in sorted(self.data_dir.glob("*.txt")):
            try:
                text = file_path.read_text(encoding="utf-8")
                section = file_path.stem
                chunks = self.split_file(text, section, file_path.name)
                if not chunks:
                    print(f"Warning: {file_path.name} has no '## ' sections; skipped.")
                    continue
                documents.extend(chunks)
                print(f"Loaded: {file_path.name} (Section: {section}, {len(chunks)} chunks)")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

        return documents
