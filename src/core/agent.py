from ..rag.vectorstore import CliQVectorStore
from ..rag.service import CliQRAGService
import certifi
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient
import os

class CliQAgent:
    """The central agent class that orchestrates platform intelligence and session history."""
    
    def __init__(self):
        # Initialize dependencies
        self.vectorstore = CliQVectorStore()
        self.rag_service = CliQRAGService(self.vectorstore)
        
        # Build the core RAG chain
        self._raw_chain = self.rag_service.get_chain()
        
        # Add history management
        self.conversational_chain = RunnableWithMessageHistory(
            self._raw_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        print("CliQAgent initialized with OOP architecture.")

    def _get_session_history(self, session_id: str):
        """Internal method to fetch session history from MongoDB."""
        client = MongoClient(
            os.getenv("DATABASE_URL"),
            tlsCAFile=certifi.where(),
        )
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=None,
            database_name="website_db",
            collection_name="chat_histories",
            client=client,
        )

    def ask(self, input_text: str, session_id: str):
        """Invokes the agent with the given input and session ID."""
        return self.conversational_chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}},
        )

    def ask_stream(self, input_text: str, session_id: str):
        """Returns an async generator for streaming the agent's response."""
        return self.conversational_chain.astream_events(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}},
            version="v2",
        )
