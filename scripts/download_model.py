"""Downloads the embedding model into the project, so the server never downloads it while starting.

Run once per build (the Dockerfile does; on Render's native Python runtime add it to the Build Command):
    python scripts/download_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.embeddings import EMBEDDING_MODEL, MODEL_CACHE_DIR, load_embedding_model  # noqa: E402


def main() -> int:
    model = load_embedding_model()
    # One real embedding proves the files are complete, not just present
    dimensions = len(model.embed_query("ready"))
    print(f"OK    {EMBEDDING_MODEL} ({dimensions} dimensions) in {MODEL_CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
