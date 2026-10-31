"""HTTP endpoints for AI reply suggestions and Ask AI answers (mounted at /api/message-ai)."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from .dependencies import get_message_assistant_service
from .schemas import MessageAssistantRequest
from .service import MessageAssistantService

router = APIRouter()


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
    return StreamingResponse(chunks, media_type="text/plain")
