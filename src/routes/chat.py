import asyncio
import hmac
import json
import os
import traceback
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.agent import CliQAgent
from ..chat.pipeline import FALLBACK_ANSWER

router = APIRouter()

# Global agent instance for the application
_agent = None

def get_agent():
    """Provides a singleton instance of the CliQAgent."""
    global _agent
    if _agent is None:
        _agent = CliQAgent()
    return _agent


class UserContext(BaseModel):
    # Only these fields reach the prompt; anything else the caller sends (emails, IDs) is dropped here
    firstname: Optional[str] = Field(default=None, max_length=60)
    pendingRequests: Optional[int] = Field(default=None, ge=0)
    unreadNotifications: Optional[int] = Field(default=None, ge=0)
    expertiseFilled: Optional[bool] = None


class ChatRequest(BaseModel):
    message: str = Field(max_length=1000)
    sessionId: str
    currentPath: Optional[str] = Field(default=None, max_length=200)
    userContext: Optional[UserContext] = None
    regenerate: bool = False


class HistoryAppendRequest(BaseModel):
    sessionId: str
    question: str
    answer: str


class FeedbackRequest(BaseModel):
    messageId: str = Field(max_length=64)
    userHash: str = Field(max_length=64)
    rating: Optional[Literal[1, -1]] = None  # None clears the vote
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=8000)
    language: Optional[str] = Field(default=None, max_length=20)
    sources: list[str] = Field(default_factory=list, max_length=20)
    cacheable: bool = False


def _require_internal_key(key):
    """Feedback changes answers for every user, so only the backend may send it."""
    expected = os.getenv("AGENT_INTERNAL_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Feedback learning is disabled: AGENT_INTERNAL_KEY is not set")
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid internal key")


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent()
    user_context = request.userContext.model_dump(exclude_none=True) if request.userContext else None
    try:
        result = await agent.ask(request.message, request.sessionId, request.currentPath, user_context)
    except Exception:
        # Never surface a 500 to the chat UI; log the cause and reply gracefully
        traceback.print_exc()
        result = {"answer": FALLBACK_ANSWER, "suggestions": []}
    return {
        "question": request.message,
        "answer": result["answer"],
        "suggestions": result.get("suggestions", []),
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Server-Sent Events: status, token, suggestions, error, done.

    When the client disconnects, Starlette cancels this generator, which cancels the
    in-flight Groq stream as well, so a Stop in the UI stops spending tokens.
    """
    agent = get_agent()
    user_context = request.userContext.model_dump(exclude_none=True) if request.userContext else None

    async def generate():
        try:
            async for event, data in agent.stream(
                request.message,
                request.sessionId,
                current_path=request.currentPath,
                user_context=user_context,
                regenerate=request.regenerate,
            ):
                yield _sse(event, data)
        except asyncio.CancelledError:
            raise
        except Exception:
            traceback.print_exc()
            yield _sse("token", {"text": FALLBACK_ANSWER})
            yield _sse("error", {"code": "internal_error"})
            yield _sse("done", {"cacheable": False})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/kb-version")
async def kb_version():
    """Fingerprint of the live knowledge base; the backend puts it in its answer-cache keys."""
    return {"kbVersion": await asyncio.to_thread(get_agent().pipeline.feedback.current_kb_version)}


@router.post("/chat/feedback")
async def chat_feedback(request: FeedbackRequest, x_internal_key: Optional[str] = Header(default=None)):
    _require_internal_key(x_internal_key)
    await asyncio.to_thread(
        get_agent().pipeline.feedback.record,
        request.messageId,
        request.userHash,
        request.rating,
        request.question,
        request.answer,
        request.language,
        request.sources,
        request.cacheable,
    )
    return {"ok": True}


@router.post("/chat/history/append")
async def append_history(request: HistoryAppendRequest):
    await asyncio.to_thread(get_agent().append_history, request.sessionId, request.question, request.answer)
    return {"ok": True}
