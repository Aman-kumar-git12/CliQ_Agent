import os
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from core.agent import CliQAgent

router = APIRouter()

# Global agent instance for the application
_agent = None

def get_agent():
    """Provides a singleton instance of the CliQAgent."""
    global _agent
    if _agent is None:
        _agent = CliQAgent()
    return _agent

class ChatRequest(BaseModel):
    message: str
    sessionId: str

@router.post("/chat")
def chat(request: ChatRequest):
    agent = get_agent()
    result = agent.ask(request.message, request.sessionId)
    return {
        "question": request.message,
        "answer": result["answer"],
    }

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    agent = get_agent()

    async def generate():
        print(f"Starting stream for session: {request.sessionId}")
        try:
            async for event in agent.ask_stream(request.message, request.sessionId):
                kind = event["event"]
                tags = event.get("tags", [])
                
                # Only stream tokens from the final document chain
                if kind == "on_chat_model_stream" and "final_response" in tags:
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
        except Exception as e:
            print(f"Error in stream generation: {e}")
            yield f"Error: {str(e)}"
        print(f"Finished stream for session: {request.sessionId}")

    return StreamingResponse(generate(), media_type="text/plain")