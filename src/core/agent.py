from ..rag.vectorstore import CliQVectorStore
from ..rag.service import CliQRAGService
from ..chat.pipeline import ChatPipeline


class CliQAgent:
    """The central agent class: owns the vector store, the RAG service and the streaming chat pipeline."""

    def __init__(self):
        self.vectorstore = CliQVectorStore()
        self.rag_service = CliQRAGService(self.vectorstore)

        # Chat history shares the vector store's MongoClient (a new client per request leaks connections)
        self.pipeline = ChatPipeline(self.rag_service, self.vectorstore.client)
        print("CliQAgent initialized with stepwise chat pipeline.")

    def stream(self, message, session_id, current_path=None, user_context=None, regenerate=False):
        """Async iterator of (event, data) pairs: status, token, suggestions, error, done."""
        return self.pipeline.stream(message, session_id, current_path, user_context, regenerate)

    async def ask(self, message, session_id, current_path=None, user_context=None):
        """Runs a full turn and returns {answer, suggestions, statuses, meta, error}."""
        return await self.pipeline.ask(message, session_id, current_path, user_context)

    def append_history(self, session_id, question, answer):
        """Records a turn answered elsewhere (e.g. the backend's Redis cache) so follow-ups keep context."""
        self.pipeline.append_exchange(session_id, question, answer)
