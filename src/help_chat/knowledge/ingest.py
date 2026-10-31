"""Command-line script that re-embeds the knowledge base into Atlas.

Run from WebsiteAgent/ after editing any file in data/:
    python -m src.help_chat.knowledge.ingest
"""

import logging
import sys

from ...core.config import get_settings
from ...core.database import get_database
from ...core.embeddings import get_embeddings
from ...core.logging import setup_logging
from .loader import load_documents
from .vectorstore import COLLECTION_NAME, KnowledgeStore

logger = logging.getLogger(__name__)


def ingest() -> bool:
    """Replaces every stored chunk with the current files. Returns True when the index is ready."""
    documents = load_documents()
    if not documents:
        logger.error("No documents found to ingest")
        return False

    store = KnowledgeStore(get_database()[COLLECTION_NAME], get_embeddings())
    store.replace_documents(documents)

    # Without a READY index every query falls back to the slow in-memory search
    if store.ensure_vector_index():
        logger.info("Ingestion complete: %d chunks, vector index ready", len(documents))
        return True
    logger.error("Ingestion finished, but the vector index is not ready; queries will use the in-memory fallback")
    return False


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    sys.exit(0 if ingest() else 1)
