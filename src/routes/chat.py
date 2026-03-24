import os
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.chain import get_rag_chain

load_dotenv()

router = APIRouter()

_rag_chain = None
_conversational_rag_chain = None


def get_session_history(session_id: str):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=os.getenv("DATABASE_URL"),
        database_name="website_db",
        collection_name="chat_histories",
    )


def get_conversational_rag_chain():
    global _rag_chain, _conversational_rag_chain

    if _conversational_rag_chain is None:
        _rag_chain = get_rag_chain()
        _conversational_rag_chain = RunnableWithMessageHistory(
            _rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    return _conversational_rag_chain


class ChatRequest(BaseModel):
    message: str
    sessionId: str


@router.post("/chat")
def chat(request: ChatRequest):
    conversational_rag_chain = get_conversational_rag_chain()

    result = conversational_rag_chain.invoke(
        {"input": request.message},
        config={"configurable": {"session_id": request.sessionId}},
    )

    return {
        "question": request.message,
        "answer": result["answer"],
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    conversational_rag_chain = get_conversational_rag_chain()

    async def generate():
        async for event in conversational_rag_chain.astream_events(
            {"input": request.message},
            config={"configurable": {"session_id": request.sessionId}},
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

    return StreamingResponse(generate(), media_type="text/plain")