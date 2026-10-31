"""Generates an expertise profile with the LLM, falling back to a deterministic draft when that fails."""

import json
import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .drafting import build_minimal_fallback, validate_and_finalize
from .facts import extract_protected_facts
from .prompts import GENERATION_PROMPT
from .schemas import ExpertiseGenerateResponse, ExpertisePayload

logger = logging.getLogger(__name__)

REFINE_MODE = "refine_existing"


def parse_json_object(response: str) -> Any:
    """Parses the outermost {...} in a model reply, ignoring any prose or code fences around it."""
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in expertise response")
    return json.loads(response[start:end])


def _response(expertise: dict[str, Any], source: str) -> dict[str, Any]:
    return ExpertiseGenerateResponse.model_validate(
        {"expertise": ExpertisePayload.model_validate(expertise).model_dump(), "generation_source": source}
    ).model_dump()


class ExpertiseService:
    """Builds Magic Builder profiles. The LLM is injected so tests can pass a fake."""

    def __init__(self, llm: Runnable) -> None:
        self._chain = GENERATION_PROMPT | llm | StrOutputParser()

    async def generate_expertise(
        self,
        user_info: dict[str, Any],
        answers: dict[str, Any],
        existing_expertise: dict[str, Any],
        mode: str = "fresh",
    ) -> dict[str, Any]:
        """Returns {"expertise": ..., "generation_source": "llm" | "fallback"}."""
        answers = answers or {}
        # A fresh build must not leak the old profile into the prompt or the fallback
        existing_expertise = existing_expertise if mode == REFINE_MODE else {}
        try:
            protected_facts = extract_protected_facts(user_info, answers, existing_expertise)
            reply = await self._chain.ainvoke(
                {
                    "protected_facts": json.dumps(protected_facts, ensure_ascii=True),
                    "existing_expertise": json.dumps(existing_expertise, ensure_ascii=True),
                }
            )
            finalized = validate_and_finalize(parse_json_object(reply), existing_expertise, answers, protected_facts)
            return _response(finalized, "llm")
        except Exception:
            # Deliberately broad: any provider, parsing or validation failure must still give the user a draft
            logger.warning("Expertise generation fell back to deterministic draft", exc_info=True)
            return _response(build_minimal_fallback(user_info, answers, existing_expertise or {}), "fallback")
