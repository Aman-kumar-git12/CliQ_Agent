from pathlib import Path
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cliq_kb.txt"


def load_documents():
    text = DATA_PATH.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": "cliq_kb.txt"})]