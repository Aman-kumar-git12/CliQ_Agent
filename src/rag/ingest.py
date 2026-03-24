from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from loader import load_documents

# Load environment variables (makes sure standalone python script reads .env correctly)
load_dotenv(dotenv_path="../.env")
load_dotenv() # Fallback

def build_vectorstore():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Connecting to MongoDB to insert Vectors...")
    client = MongoClient(os.getenv("DATABASE_URL"))
    collection = client["website_db"]["cliq_vectors"]

    # Clear old vectors before re-ingesting
    collection.delete_many({})

    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection=collection,
        index_name="default"
    )

    print("MongoDB Atlas Vector Search ingestion complete!")

if __name__ == "__main__":
    build_vectorstore()