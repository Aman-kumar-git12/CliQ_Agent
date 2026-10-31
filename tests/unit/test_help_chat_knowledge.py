"""Knowledge base chunking and fingerprinting."""

from pathlib import Path

from src.help_chat.knowledge.loader import kb_fingerprint, list_sections, load_documents, split_file

SAMPLE = """# Feature: Posts
Route: /create/post

## Summary
Share text and images.

## How to create a post
1. Click **New Post**.

## Empty
"""


def test_split_file_makes_one_chunk_per_heading_with_metadata() -> None:
    chunks = split_file(SAMPLE, "create", "create.txt")

    assert [c.metadata["heading"] for c in chunks] == ["Summary", "How to create a post"]  # empty section skipped
    assert chunks[1].page_content.startswith("Feature: Posts — How to create a post\nRoute: /create/post\n\n")
    assert chunks[1].metadata == {
        "source": "create.txt",
        "section": "create",
        "feature": "Posts",
        "heading": "How to create a post",
        "route": "/create/post",
    }


def test_split_file_without_header_uses_section_name() -> None:
    chunks = split_file("## Intro\nHello", "smart_connections", "smart_connections.txt")
    assert chunks[0].metadata["feature"] == "Smart Connections"
    assert chunks[0].metadata["route"] == "n/a"


def test_real_knowledge_base_loads_and_fits_the_embedding_window() -> None:
    documents = load_documents()
    assert len(list_sections()) >= 14
    assert documents
    assert all(len(d.page_content) <= 1800 for d in documents)


def test_fingerprint_changes_when_a_file_changes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("## One\nfirst")
    before = kb_fingerprint(tmp_path)
    assert kb_fingerprint(tmp_path) == before
    (tmp_path / "a.txt").write_text("## One\nchanged")
    assert kb_fingerprint(tmp_path) != before
