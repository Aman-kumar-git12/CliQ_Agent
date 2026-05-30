import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting CliQ AI Agent server on {host}:{port}...")
    uvicorn.run("src.main:app", host=host, port=port, reload=True)
