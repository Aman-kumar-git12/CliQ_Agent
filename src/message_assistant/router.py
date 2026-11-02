"""HTTP endpoints for AI reply suggestions and Ask AI answers (mounted at /api/message-ai)."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..core.security import require_internal_key
from . import canned_responses
from .dependencies import get_message_assistant_service
from .schemas import MessageAssistantRequest
from .service import MessageAssistantService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_internal_key)])


@router.post("/respond", response_model=None)
def message_assistant(
    request: MessageAssistantRequest,
    service: MessageAssistantService = Depends(get_message_assistant_service),
) -> dict[str, Any]:
    """Ask AI answer when mode is 'ask', otherwise reply suggestions."""
    conversation = [turn.model_dump() for turn in request.conversation]
    if request.mode == "ask":
        return service.answer_about_conversation(
            conversation=conversation,
            assistant_history=[turn.model_dump() for turn in request.assistant_history],
            question=request.question,
            other_name=request.other_name,
            older_context=request.older_context,
            tone=request.tone,
        )
    return service.generate_response(
        conversation=conversation,
        draft=request.draft,
        other_name=request.other_name,
        older_context=request.older_context,
        tone=request.tone,
        max_suggestions=request.max_suggestions,
        emoji_preference=request.emojiPreference,
    )


async def _ended_cleanly(chunks: AsyncIterator[str], question: str) -> AsyncIterator[str]:
    """Stream boundary: the status line is already sent, so a failure must end the text, not the connection."""
    try:
        async for chunk in chunks:
            yield chunk
    except Exception:
        logger.exception("POST /respond/stream failed")
        yield canned_responses.fallback_answer(question)["answer"]


@router.post("/respond/stream")
async def message_assistant_stream(
    request: MessageAssistantRequest,
    service: MessageAssistantService = Depends(get_message_assistant_service),
) -> StreamingResponse:
    """Ask AI answer streamed as plain text."""
    chunks: AsyncIterator[str] = service.stream_answer(
        conversation=[turn.model_dump() for turn in request.conversation],
        assistant_history=[turn.model_dump() for turn in request.assistant_history],
        question=request.question,
        other_name=request.other_name,
        older_context=request.older_context,
        tone=request.tone,
    )
    return StreamingResponse(
        _ended_cleanly(chunks, request.question),
        media_type="text/plain",
        # Proxies must pass each chunk on at once instead of buffering the answer
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
