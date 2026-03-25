import os
import sys
from dotenv import load_dotenv

# Add the src directory to sys.path to allow imports from subdirectories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loader import CliQDocumentLoader
from rag.vectorstore import CliQVectorStore

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

def build_vectorstore():
    # Instantiate OOP components
    loader = CliQDocumentLoader()
    documents = loader.load_documents()

    if not documents:
        print("No documents found to ingest.")
        return

    print("Connecting to MongoDB to insert Vectors...")
    vectorstore = CliQVectorStore()
    
    # Clear old vectors before re-ingesting
    vectorstore.clear_collection()

    # Ingest new documents
    vectorstore.ingest_documents(documents)

    print("MongoDB Atlas Vector Search ingestion complete!")

if __name__ == "__main__":
    build_vectorstore()