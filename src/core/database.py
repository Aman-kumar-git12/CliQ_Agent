"""The single MongoDB client shared by every feature (one client per process avoids leaking connections)."""

from functools import lru_cache

import certifi
from pymongo import MongoClient
from pymongo.database import Database

from .config import get_settings


@lru_cache
def get_mongo_client() -> MongoClient:
    return MongoClient(get_settings().database_url, tlsCAFile=certifi.where())


def get_database() -> Database:
    return get_mongo_client()[get_settings().database_name]


def close_mongo_client() -> None:
    if get_mongo_client.cache_info().currsize:
        get_mongo_client().close()
        get_mongo_client.cache_clear()
