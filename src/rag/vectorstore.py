import os
import certifi
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class CliQVectorStore:
    """Manages the connection and operations for the MongoDB Atlas Vector Search."""
    
    def __init__(self, connection_string: str = None, database_name: str = "website_db", collection_name: str = "cliq_vectors"):
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.database_name = database_name
        self.collection_name = collection_name
        self.index_name = "default"
        
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.client = MongoClient(
            self.connection_string,
            tlsCAFile=certifi.where(),
        )
        self.collection = self.client[self.database_name][self.collection_name]
        
        # Initialize the vector search instance
        self.vector_search = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=self.index_name,
            text_key="text",
            embedding_key="embedding"
        )

    def similarity_search(self, query: str, k: int = 3, pre_filter: dict = None):
        """Performs a similarity search with optional metadata filtering."""
        print(f"Searching for: '{query}' with filter: {pre_filter}")
        
        try:
            results = self.vector_search.similarity_search(
                query,
                k=k,
                pre_filter=pre_filter
            )
            if results:
                return results
        except Exception as e:
            print(f"Atlas Vector Search failed/missing: {e}")

        # Fallback: Local In-Memory Search (for small collections)
        print("Using local in-memory fallback search...")
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Fetch all documents (limited to 1000 for safety)
            cursor = self.collection.find(pre_filter or {}).limit(1000)
            all_docs = list(cursor)
            
            if not all_docs:
                return []

            import numpy as np
            
            scored_docs = []
            for doc in all_docs:
                if 'embedding' not in doc: continue
                
                # Simple dot product for cosine similarity (assuming normalized vectors)
                sim = np.dot(query_vector, doc['embedding'])
                scored_docs.append((sim, doc))
            
            # Sort by similarity descending
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_k = scored_docs[:k]
            
            from langchain_core.documents import Document
            return [
                Document(
                    page_content=d['text'],
                    metadata={k: v for k, v in d.items() if k not in ['_id', 'embedding', 'text']}
                ) for _, d in top_k
            ]
        except Exception as e:
            print(f"Local fallback search failed: {e}")
            return []

    def clear_collection(self):
        """Deletes all documents from the vector collection."""
        print(f"Clearing collection: {self.collection_name}")
        self.collection.delete_many({})

    def ingest_documents(self, documents):
        """Splits and ingests documents into the vector store."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        split_docs = splitter.split_documents(documents)
        
        print(f"Ingesting {len(split_docs)} chunks into {self.collection_name}...")
        # Use add_documents to preserve the instance configuration and metadata
        return self.vector_search.add_documents(split_docs)
