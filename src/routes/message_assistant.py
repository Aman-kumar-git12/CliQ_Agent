from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from services.message_assistant import (
    answer_about_conversation,
    generate_message_assistant_response,
    stream_answer_about_conversation,
)

router = APIRouter()


class MessageAssistantTurn(BaseModel):
    role: str
    text: str | None = ""
    kind: str | None = "text"
    timestamp: str | None = None


class AssistantHistoryTurn(BaseModel):
    role: str
    text: str | None = ""


class MessageAssistantRequest(BaseModel):
    conversation: list[MessageAssistantTurn] = []
    assistant_history: list[AssistantHistoryTurn] = []
    draft: str = ""
    mode: str = "replies"
    question: str = ""
    older_context: str = ""
    tone: str = "polite"
    other_name: str = "the other person"
    max_suggestions: int = 5


@router.post("/respond")
def message_assistant(request: MessageAssistantRequest):
    conversation = [turn.model_dump() for turn in request.conversation]
    if request.mode == "ask":
        payload = answer_about_conversation(
            conversation=conversation,
            assistant_history=[turn.model_dump() for turn in request.assistant_history],
            question=request.question,
            other_name=request.other_name,
            older_context=request.older_context,
            tone=request.tone,
        )
    else:
        payload = generate_message_assistant_response(
            conversation=conversation,
            draft=request.draft,
            other_name=request.other_name,
            older_context=request.older_context,
            tone=request.tone,
            max_suggestions=request.max_suggestions,
        )
    return payload


@router.post("/respond/stream")
async def message_assistant_stream(request: MessageAssistantRequest):
    async def generate():
        async for chunk in stream_answer_about_conversation(
            conversation=[turn.model_dump() for turn in request.conversation],
            assistant_history=[turn.model_dump() for turn in request.assistant_history],
            question=request.question,
            other_name=request.other_name,
            older_context=request.older_context,
            tone=request.tone,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
