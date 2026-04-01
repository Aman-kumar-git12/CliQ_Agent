import traceback

from fastapi import APIRouter, HTTPException

from ..models.expertise import ExpertiseGenerateRequest, ExpertiseGenerateResponse
from ..services.expertise import expertise_service

router = APIRouter()

@router.post("/generate", response_model=ExpertiseGenerateResponse)
async def generate_expertise(request: ExpertiseGenerateRequest):
    try:
        result = await expertise_service.generate_expertise(
            request.user_info.model_dump(),
            request.answers.model_dump(),
            request.existing_expertise.model_dump() if hasattr(request.existing_expertise, "model_dump") else (request.existing_expertise or {}),
            request.mode,
        )
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate expertise JSON")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
