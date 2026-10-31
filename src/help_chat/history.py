"""Per-session chat history in MongoDB, so follow-up questions keep their context."""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient

COLLECTION_NAME = "chat_histories"
# Only the most recent turns are sent to the LLM, so long chats don't overflow the context window
MAX_HISTORY_MESSAGES = 10


class ChatHistoryStore:
    def __init__(self, client: MongoClient, database_name: str, collection_name: str = COLLECTION_NAME) -> None:
        self.client = client
        self.database_name = database_name
        self.collection_name = collection_name
        self.collection = client[database_name][collection_name]

    def _history(self, session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=None,
            database_name=self.database_name,
            collection_name=self.collection_name,
            client=self.client,
            history_size=MAX_HISTORY_MESSAGES,
        )

    def messages(self, session_id: str) -> list[BaseMessage]:
        return self._history(session_id).messages

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        self._history(session_id).add_messages([HumanMessage(content=question), AIMessage(content=answer)])

    def drop_last_exchange(self, session_id: str) -> None:
        """Removes the last question/answer pair so Regenerate doesn't see its own previous answer."""
        docs = list(self.collection.find({"SessionId": session_id}).sort("_id", -1).limit(2))
        types = [json.loads(d["History"]).get("type") for d in docs]
        if types == ["ai", "human"]:
            self.collection.delete_many({"_id": {"$in": [d["_id"] for d in docs]}})

    def delete_session(self, session_id: str) -> None:
        self.collection.delete_many({"SessionId": session_id})
