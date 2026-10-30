import os
import time
import certifi
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# BAAI/bge-small-en-v1.5 produces 384-dimensional vectors
EMBEDDING_DIMENSIONS = 384

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

    def similarity_search(self, query: str, k: int = 4, pre_filter: dict = None):
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

    def _index_status(self):
        for idx in self.collection.list_search_indexes(self.index_name):
            return idx.get("status"), bool(idx.get("queryable"))
        return None, False

    def ensure_vector_index(self, wait_seconds: float = 300):
        """Creates the Atlas Vector Search index if needed and waits until it is READY.

        Index definition (same JSON can be pasted into the Atlas UI JSON editor):
        {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
            {"type": "filter", "path": "section"}
        ]}
        Returns True once the index is queryable.
        """
        status, queryable = self._index_status()
        if status is None:
            print(f"Creating vector search index '{self.index_name}'...")
            try:
                self.vector_search.create_vector_search_index(
                    dimensions=EMBEDDING_DIMENSIONS,
                    filters=["section"],
                )
            except Exception as e:
                print(
                    f"Could not create vector search index automatically: {e}\n"
                    "Create it in the Atlas UI: Database > Browse Collections > "
                    f"{self.database_name}.{self.collection_name} > Search Indexes > Create Search Index > "
                    "Atlas Vector Search > JSON Editor, name it "
                    f"'{self.index_name}', and paste the definition from CliQVectorStore.ensure_vector_index."
                )
                return False
        else:
            print(f"Vector search index '{self.index_name}' already exists (status: {status}).")

        deadline = time.time() + wait_seconds
        while True:
            status, queryable = self._index_status()
            if status == "READY" and queryable:
                print(f"Vector search index '{self.index_name}' is READY.")
                return True
            if status == "FAILED" or time.time() > deadline:
                print(f"Vector search index '{self.index_name}' is not ready (status: {status}).")
                return False
            print(f"Waiting for index '{self.index_name}' (status: {status})...")
            time.sleep(5)

    def clear_collection(self):
        """Deletes all documents from the vector collection."""
        print(f"Clearing collection: {self.collection_name}")
        self.collection.delete_many({})

    def ingest_documents(self, documents):
        """Embeds and stores documents. CliQDocumentLoader already chunks by heading,
        so no character-based splitting happens here."""
        print(f"Ingesting {len(documents)} chunks into {self.collection_name}...")
        return self.vector_search.add_documents(documents)
