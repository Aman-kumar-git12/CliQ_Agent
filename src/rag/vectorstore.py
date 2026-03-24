from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = MongoClient(os.getenv("DATABASE_URL"))
    collection = client["website_db"]["cliq_vectors"]

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="default"  # This is the name of the Atlas Search Index we'll configure
    )

    return vectorstore