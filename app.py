"""Starts the agent locally with auto-reload: python app.py"""

import uvicorn

from src.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=True)
