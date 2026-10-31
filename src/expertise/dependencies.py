"""FastAPI dependency that hands routes the ExpertiseService built in the app lifespan."""

from fastapi import Request

from .service import ExpertiseService


def get_expertise_service(request: Request) -> ExpertiseService:
    return request.app.state.expertise
