"""HTTP route for the Magic Builder (mounted at /api/expertise)."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.security import require_internal_key
from .dependencies import get_expertise_service
from .schemas import ExpertiseGenerateRequest, ExpertiseGenerateResponse
from .service import ExpertiseService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_internal_key)])


@router.post("/generate", response_model=ExpertiseGenerateResponse)
async def generate_expertise(
    request: ExpertiseGenerateRequest,
    service: ExpertiseService = Depends(get_expertise_service),
) -> dict[str, Any]:
    existing = request.existing_expertise
    existing_expertise = existing.model_dump() if isinstance(existing, BaseModel) else (existing or {})
    try:
        result = await service.generate_expertise(
            request.user_info.model_dump(), request.answers.model_dump(), existing_expertise, request.mode
        )
    except Exception as exc:
        logger.exception("Expertise generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate expertise JSON")
    return result
