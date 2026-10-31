"""Creates the FastAPI app: logging, startup/shutdown, CORS and every feature's routes."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.database import close_mongo_client
from .core.llm import get_classifier_llm, get_fallback_response_llm, get_response_llm
from .core.logging import setup_logging
from .expertise.router import router as expertise_router
from .expertise.service import ExpertiseService
from .help_chat.router import router as help_chat_router
from .help_chat.service import build_help_chat_service
from .matching.embedding import TextEmbedder, load_embedding_backend
from .matching.router import router as matching_router
from .matching.scoring import RankingWeights
from .matching.service import MatchingService
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
    app.state.matching = MatchingService(TextEmbedder(load_embedding_backend()), RankingWeights.from_settings(settings))

    app.state.keepalive = KeepAlive(settings.port)
    keepalive_task = asyncio.create_task(app.state.keepalive.run())
    logger.info("CliQ AI agent ready")

    yield

    keepalive_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await keepalive_task
    close_mongo_client()


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title="CliQ AI Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(help_chat_router, prefix="/api")
    app.include_router(message_assistant_router, prefix="/api/message-ai")
    app.include_router(expertise_router, prefix="/api/expertise")
    app.include_router(matching_router, prefix="/api/match")
    app.include_router(system_router)
    return app


app = create_app()
