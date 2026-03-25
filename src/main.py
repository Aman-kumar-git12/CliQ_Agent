import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the current directory to sys.path to allow imports from subdirectories
# when running from the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from routes.chat import router as chat_router, get_conversational_rag_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the RAG chain on startup (downloads/loads embeddings and vectorstore)
    print("Initializing AI RAG chain...")
    get_conversational_rag_chain()
    print("AI RAG chain ready.")
    yield

# Allow requests from the Vite frontend
frontend_url = os.getenv("FRONTEND_URL", "*")
origins = [frontend_url] if frontend_url != "*" else ["*"]

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI LLM Service is running"}