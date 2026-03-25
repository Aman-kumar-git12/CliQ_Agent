import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the current directory to sys.path to allow imports from subdirectories
# when running from the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.chat import router as chat_router

app = FastAPI()

# Allow requests from the Vite frontend
frontend_url = os.getenv("FRONTEND_URL", "*")
origins = [frontend_url] if frontend_url != "*" else ["*"]

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