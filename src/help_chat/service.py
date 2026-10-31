"""The help chat's public operations, and the factory that wires its parts together."""

import asyncio
from collections.abc import AsyncIterator

from ..core.config import Settings
from ..core.database import get_mongo_client
from ..core.embeddings import get_embeddings
from ..core.llm import get_help_answer_llm, get_help_router_llm
from .answering import AnswerWriter
from .feedback import COLLECTION_NAME as FEEDBACK_COLLECTION
from .feedback import FeedbackMemory
from .history import ChatHistoryStore
from .knowledge.loader import kb_fingerprint, list_sections, load_documents
from .knowledge.vectorstore import COLLECTION_NAME as VECTOR_COLLECTION
from .knowledge.vectorstore import KnowledgeStore
from .pipeline import ChatPipeline, Event
from .retrieval import Retriever
from .routing import QuestionRouter
from .schemas import ChatRequest, ChatResult, FeedbackRequest
from .suggestions import FollowUpSuggester


class HelpChatService:
    """What the API can do with the help chat. Knows nothing about HTTP."""

    def __init__(self, pipeline: ChatPipeline) -> None:
        self.pipeline = pipeline

    def stream(self, request: ChatRequest) -> AsyncIterator[Event]:
        return self.pipeline.stream(
            request.message,
            request.sessionId,
            current_path=request.currentPath,
            user_context=request.user_context_dict(),
            regenerate=request.regenerate,
        )

    async def ask(self, request: ChatRequest) -> ChatResult:
        return await self.pipeline.ask(
            request.message, request.sessionId, request.currentPath, request.user_context_dict()
        )

    async def append_history(self, session_id: str, question: str, answer: str) -> None:
        """Records a turn answered elsewhere (e.g. the backend's Redis cache) so follow-ups keep context."""
        await asyncio.to_thread(self.pipeline.history.append_exchange, session_id, question, answer)

    async def kb_version(self) -> str:
        """Fingerprint of the live knowledge base; the backend puts it in its answer-cache keys."""
        return await asyncio.to_thread(self.pipeline.feedback.current_kb_version)

    async def record_feedback(self, feedback: FeedbackRequest) -> None:
        await asyncio.to_thread(
            self.pipeline.feedback.record,
            feedback.messageId,
            feedback.userHash,
            feedback.rating,
            feedback.question,
            feedback.answer,
            feedback.language,
            feedback.sources,
            feedback.cacheable,
        )


def build_help_chat_service(settings: Settings) -> HelpChatService:
    """Creates every part of the help chat. Called once at startup."""
    client = get_mongo_client()
    database = client[settings.database_name]
    embeddings = get_embeddings()
    answer_llm = get_help_answer_llm()

    pipeline = ChatPipeline(
        router=QuestionRouter(get_help_router_llm(), list_sections()),
        retriever=Retriever(KnowledgeStore(database[VECTOR_COLLECTION], embeddings)),
        writer=AnswerWriter(answer_llm),
        history=ChatHistoryStore(client, settings.database_name),
        feedback=FeedbackMemory(
            database[FEEDBACK_COLLECTION],
            embeddings,
            kb_fingerprint,
            similarity_threshold=settings.feedback_similarity_threshold,
            min_approvals=settings.feedback_min_approvals,
        ),
        suggester=FollowUpSuggester(load_documents()),
        model_name=answer_llm.model_name or "unknown",
    )
    return HelpChatService(pipeline)
