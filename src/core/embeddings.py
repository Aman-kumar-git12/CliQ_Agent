"""The local embedding model, loaded once and shared by the help chat and matching."""

from functools import lru_cache

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge-small-en-v1.5 produces 384-dimensional vectors
EMBEDDING_DIMENSIONS = 384


@lru_cache
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
