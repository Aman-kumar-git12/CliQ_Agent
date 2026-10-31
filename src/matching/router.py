"""HTTP routes for smart connections and semantic people search (mounted at /api/match)."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..core.security import require_internal_key
from .dependencies import get_matching_service
from .schemas import MatchRequest, PrecomputeRequest, SemanticSearchRequest
from .service import DEFAULT_RETRIEVAL_K, DEFAULT_SEARCH_TOP_K, DEFAULT_TOP_K, MatchingService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_internal_key)])


@router.post("/match", response_model=None)
async def match_users(
    request: MatchRequest, service: MatchingService = Depends(get_matching_service)
) -> dict[str, Any]:
    try:
        ranked_results = await service.rank_candidates(
            request.userProfile,
            request.historyProfile,
            request.candidates,
            # `or` (not a default) so an explicit 0 from the backend also means "use the default"
            top_k=request.topK or DEFAULT_TOP_K,
            retrieval_k=request.retrievalK or DEFAULT_RETRIEVAL_K,
        )
    except Exception as exc:
        logger.exception("Matching error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"rankedResults": ranked_results}


@router.post("/precompute", response_model=None)
async def precompute_profile(
    request: PrecomputeRequest, service: MatchingService = Depends(get_matching_service)
) -> dict[str, Any]:
    try:
        return await service.precompute_profile(request.profileId, request.profile)
    except Exception as exc:
        logger.exception("Profile precompute error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/search", response_model=None)
async def semantic_search(
    request: SemanticSearchRequest, service: MatchingService = Depends(get_matching_service)
) -> dict[str, Any]:
    try:
        results = await service.semantic_search(
            request.query,
            request.candidates,
            top_k=request.topK or DEFAULT_SEARCH_TOP_K,
            min_score=request.minScore,
        )
    except Exception as exc:
        logger.exception("Semantic search error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"results": results}
