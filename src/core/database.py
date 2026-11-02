"""The single MongoDB client shared by every feature (one client per process avoids leaking connections)."""

from functools import lru_cache

import certifi
from pymongo import MongoClient
from pymongo.database import Database

from .config import get_settings

# pymongo waits 30 s to find a server and forever for a reply by default, so an unreachable Atlas would hang
# a chat turn (and startup). With these limits every caller gets an error quickly and can degrade instead.
SERVER_SELECTION_TIMEOUT_MS = 5_000
CONNECT_TIMEOUT_MS = 5_000
SOCKET_TIMEOUT_MS = 20_000


@lru_cache
def get_mongo_client() -> MongoClient:
    return MongoClient(
        get_settings().database_url,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=CONNECT_TIMEOUT_MS,
        socketTimeoutMS=SOCKET_TIMEOUT_MS,
    )


def get_database() -> Database:
    return get_mongo_client()[get_settings().database_name]


def close_mongo_client() -> None:
    if get_mongo_client.cache_info().currsize:
        get_mongo_client().close()
        get_mongo_client.cache_clear()
