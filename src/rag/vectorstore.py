from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore():
    # Use fastembed (CPU-optimized, ~30MB RAM overhead) instead of local transformers (~350MB+ RAM)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    client = MongoClient(os.getenv("DATABASE_URL"))
    collection = client["website_db"]["cliq_vectors"]

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="default"  # This is the name of the Atlas Search Index we'll configure
    )

    return vectorstore