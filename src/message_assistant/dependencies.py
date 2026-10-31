"""FastAPI dependency that hands routes the service instance built at startup."""

from fastapi import Request

from .service import MessageAssistantService


def get_message_assistant_service(request: Request) -> MessageAssistantService:
    """The service created in the app lifespan and stored on app.state."""
    return request.app.state.message_assistant
