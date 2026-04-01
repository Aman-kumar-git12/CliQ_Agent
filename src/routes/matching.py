from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from ..services.matching import matching_service

router = APIRouter()

class MatchRequest(BaseModel):
    userProfile: Dict[str, Any]
    historyProfile: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]]
    topK: Optional[int] = None
    retrievalK: Optional[int] = None


class PrecomputeRequest(BaseModel):
    profileId: str
    profile: Dict[str, Any]

@router.post("/match")
async def match_users(request: MatchRequest):
    try:
        ranked_results = await matching_service.rank_candidates(
            request.userProfile,
            request.historyProfile,
            request.candidates,
            top_k=request.topK or 60,
            retrieval_k=request.retrievalK or 180,
        )
        return {"rankedResults": ranked_results}
    except Exception as e:
        print(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/precompute")
async def precompute_profile(request: PrecomputeRequest):
    try:
        return await matching_service.precompute_profile(request.profileId, request.profile)
    except Exception as e:
        print(f"Profile precompute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
