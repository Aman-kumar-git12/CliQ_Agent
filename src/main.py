"""Creates the FastAPI app: logging, monitoring, startup/shutdown, middleware and every feature's routes."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.database import close_mongo_client, get_database
from .core.llm import get_classifier_llm, get_fallback_response_llm, get_response_llm
from .core.logging import setup_logging
from .core.monitoring import setup_sentry
from .core.request_id import RequestIdMiddleware
from .expertise.router import router as expertise_router
from .expertise.service import ExpertiseService
from .help_chat.router import router as help_chat_router
from .help_chat.service import build_help_chat_service
from .matching.embedding import TextEmbedder, load_embedding_backend
from .matching.router import router as matching_router
from .matching.scoring import RankingWeights
from .matching.service import MatchingService
from .matching.vector_store import COLLECTION_NAME as VECTOR_COLLECTION
from .matching.vector_store import ProfileVectorStore
from .message_assistant.router import router as message_assistant_router
from .message_assistant.service import MessageAssistantService
from .system.keepalive import KeepAlive
from .system.router import router as system_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Builds every service once at startup; routes reach them through app.state."""
    settings = get_settings()
    logger.info("Starting CliQ AI agent")

    app.state.help_chat = build_help_chat_service(settings)
    app.state.message_assistant = MessageAssistantService(
        get_classifier_llm(), get_response_llm(), get_fallback_response_llm()
    )
    app.state.expertise = ExpertiseService(get_response_llm())
    embedder = TextEmbedder(load_embedding_backend())
    # Embeddings from the real model and from the hash fallback live side by side and are never mixed
    vector_store = ProfileVectorStore(
        get_database()[VECTOR_COLLECTION], model="bge-small-en-v1.5" if embedder.uses_model else "hash-96"
    )
    app.state.matching = MatchingService(embedder, RankingWeights.from_settings(settings), vector_store)
    # Read the stored embeddings now, in the background, so the first ranking request doesn't wait for it
    preload_task = asyncio.create_task(asyncio.to_thread(vector_store.preload))

    app.state.keepalive = KeepAlive(settings.port, settings.keepalive_url)
    keepalive_task = asyncio.create_task(app.state.keepalive.run())
    logger.info("CliQ AI agent ready")

    yield

    preload_task.cancel()
    keepalive_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await keepalive_task
    close_mongo_client()


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    # Before the app exists, so Sentry's FastAPI integration can hook into it
    setup_sentry(settings)

    app = FastAPI(title="CliQ AI Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Added last, so it is the outermost middleware and every response carries X-Request-Id
    app.add_middleware(RequestIdMiddleware)

    app.include_router(help_chat_router, prefix="/api")
    app.include_router(message_assistant_router, prefix="/api/message-ai")
    app.include_router(expertise_router, prefix="/api/expertise")
    app.include_router(matching_router, prefix="/api/match")
    app.include_router(system_router)
    return app


app = create_app()
