"""FastAPI dependency that hands routes the help chat service built at startup."""

from fastapi import Request

from .service import HelpChatService


def get_help_chat_service(request: Request) -> HelpChatService:
    return request.app.state.help_chat
