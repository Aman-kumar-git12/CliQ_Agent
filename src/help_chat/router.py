"""HTTP endpoints of the help chat (mounted under /api)."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..core.security import require_internal_key
from .dependencies import get_help_chat_service
from .errors import FALLBACK_ANSWER
from .schemas import ChatRequest, ChatResponse, FeedbackRequest, HistoryAppendRequest, KbVersionResponse, OkResponse
from .service import HelpChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["help chat"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, service: HelpChatService = Depends(get_help_chat_service)) -> ChatResponse:
    try:
        result = await service.ask(request)
    except Exception:
        # Never surface a 500 to the chat UI; log the cause and reply gracefully
        logger.exception("POST /chat failed")
        return ChatResponse(question=request.message, answer=FALLBACK_ANSWER)
    return ChatResponse(question=request.message, answer=result.answer, suggestions=result.suggestions)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest, service: HelpChatService = Depends(get_help_chat_service)
) -> StreamingResponse:
    """Server-Sent Events: status, token, suggestions, error, done.

    When the client disconnects, Starlette cancels this generator, which cancels the
    in-flight Groq stream as well, so a Stop in the UI stops spending tokens.
    """

    async def events() -> AsyncIterator[str]:
        try:
            async for event, data in service.stream(request):
                yield _sse(event, data)
        except Exception:
            # The pipeline already turns its own failures into events; this only catches bugs around it
            logger.exception("POST /chat/stream failed")
            yield _sse("token", {"text": FALLBACK_ANSWER})
            yield _sse("error", {"code": "internal_error"})
            yield _sse("done", {"cacheable": False})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/kb-version", response_model=KbVersionResponse)
async def kb_version(service: HelpChatService = Depends(get_help_chat_service)) -> KbVersionResponse:
    return KbVersionResponse(kbVersion=await service.kb_version())


@router.post("/feedback", response_model=OkResponse, dependencies=[Depends(require_internal_key)])
async def chat_feedback(
    request: FeedbackRequest, service: HelpChatService = Depends(get_help_chat_service)
) -> OkResponse:
    """Votes change answers for every user, so only the backend (with the internal key) may send them."""
    await service.record_feedback(request)
    return OkResponse()


@router.post("/history/append", response_model=OkResponse)
async def append_history(
    request: HistoryAppendRequest, service: HelpChatService = Depends(get_help_chat_service)
) -> OkResponse:
    await service.append_history(request.sessionId, request.question, request.answer)
    return OkResponse()
