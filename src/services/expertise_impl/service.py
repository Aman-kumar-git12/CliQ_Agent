import json
import logging

from langchain_core.output_parsers import StrOutputParser

from ...core.llm import response_llm
from ...models.expertise import ExpertiseGenerateResponse, ExpertisePayload
from ...prompts.expertise import GENERATION_PROMPT
from .utils import (
    _build_minimal_fallback,
    _extract_protected_facts,
    _parse_json_object,
    _validate_and_finalize,
)

logger = logging.getLogger(__name__)

class ExpertiseService:
    def __init__(self):
        self.chain = GENERATION_PROMPT | response_llm | StrOutputParser()

    async def generate_expertise(self, user_info: dict, answers: dict, existing_expertise: dict, mode: str = "fresh") -> dict:
        try:
            use_existing = mode == "refine_existing"
            existing_expertise = existing_expertise if use_existing else {}
            protected_facts = _extract_protected_facts(user_info, answers or {}, existing_expertise)
            response = await self.chain.ainvoke({
                "protected_facts": json.dumps(protected_facts, ensure_ascii=True),
                "existing_expertise": json.dumps(existing_expertise, ensure_ascii=True),
            })
            generated = _parse_json_object(response)
            return ExpertiseGenerateResponse.model_validate({
                "expertise": ExpertisePayload.model_validate(
                    _validate_and_finalize(generated, existing_expertise, answers or {}, protected_facts)
                ).model_dump(),
                "generation_source": "llm",
            }).model_dump()
        except Exception as exc:
            logger.exception("Expertise generation fell back to deterministic draft: %s", exc)
            return ExpertiseGenerateResponse.model_validate({
                "expertise": ExpertisePayload.model_validate(
                    _build_minimal_fallback(user_info, answers or {}, existing_expertise or {})
                ).model_dump(),
                "generation_source": "fallback",
            }).model_dump()


expertise_service = ExpertiseService()
