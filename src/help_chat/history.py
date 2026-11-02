"""Per-session chat history in MongoDB, so follow-up questions keep their context."""

import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, message_to_dict, messages_from_dict
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chat_histories"
# Only the most recent turns are sent to the LLM, so long chats don't overflow the context window
MAX_HISTORY_MESSAGES = 10
# Same document shape as langchain-mongodb's MongoDBChatMessageHistory, so existing histories keep working
SESSION_KEY = "SessionId"
HISTORY_KEY = "History"


class ChatHistoryStore:
    """Reads and writes the history directly: MongoDBChatMessageHistory issued a create_index on every use
    and a count plus a skip-scan on every read, three round trips to Atlas where one is enough."""

    def __init__(self, client: MongoClient, database_name: str, collection_name: str = COLLECTION_NAME) -> None:
        self.collection = client[database_name][collection_name]
        try:
            self.collection.create_index(SESSION_KEY)
        except PyMongoError as exc:
            # Startup must not fail (or hang) because Atlas is briefly unreachable; the index usually exists
            logger.warning("Could not ensure the chat history index: %s", exc)

    def messages(self, session_id: str) -> list[BaseMessage]:
        """The last MAX_HISTORY_MESSAGES messages, oldest first."""
        docs = list(
            self.collection.find({SESSION_KEY: session_id}, {HISTORY_KEY: 1})
            .sort("_id", -1)
            .limit(MAX_HISTORY_MESSAGES)
        )
        return messages_from_dict([json.loads(doc[HISTORY_KEY]) for doc in reversed(docs)])

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        messages = [HumanMessage(content=question), AIMessage(content=answer)]
        self.collection.insert_many(
            [{SESSION_KEY: session_id, HISTORY_KEY: json.dumps(message_to_dict(m))} for m in messages]
        )

    def drop_last_exchange(self, session_id: str) -> None:
        """Removes the last question/answer pair so Regenerate doesn't see its own previous answer."""
        docs = list(self.collection.find({SESSION_KEY: session_id}).sort("_id", -1).limit(2))
        types = [json.loads(d[HISTORY_KEY]).get("type") for d in docs]
        if types == ["ai", "human"]:
            self.collection.delete_many({"_id": {"$in": [d["_id"] for d in docs]}})

    def delete_session(self, session_id: str) -> None:
        self.collection.delete_many({SESSION_KEY: session_id})
