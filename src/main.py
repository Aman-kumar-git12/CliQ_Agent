import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.chat import router as chat_router, get_agent
from routes.message_assistant import router as message_assistant_router
from routes.expertise import router as expertise_router
from routes.matching import router as matching_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the CliQ Agent on startup
    print("🚀 Initializing CliQ AI Agent (OOP Mode)...")
    get_agent()
    print("✅ CliQ AI Agent ready.")
    yield

# Configure CORS
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
app.include_router(message_assistant_router, prefix="/api/message-ai")
app.include_router(expertise_router, prefix="/api/expertise")
app.include_router(matching_router, prefix="/api/match")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CliQ AI Professional Agent Service is online"}
