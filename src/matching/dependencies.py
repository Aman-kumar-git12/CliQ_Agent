"""FastAPI dependency that hands routes the MatchingService built in the app lifespan."""

from fastapi import Request

from .service import MatchingService


def get_matching_service(request: Request) -> MatchingService:
    return request.app.state.matching
