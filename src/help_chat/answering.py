"""Step 3 of a help chat turn: streams one grounded answer that covers every part."""

from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .prompts import (
    ANSWER_STYLES,
    DIRECT_INSTRUCTIONS,
    DIRECT_RESPONSE_PROMPT,
    LANGUAGE_INSTRUCTIONS,
    NO_INFO_ANSWER,
    RAG_PROMPT,
)
from .routing import DIRECT_INTENTS
from .schemas import RoutedPart


class AnswerWriter:
    def __init__(self, llm: Runnable) -> None:
        # Tagged so log/trace tooling can tell the user-facing answer apart from the router call
        self.direct_chain = (DIRECT_RESPONSE_PROMPT | llm | StrOutputParser()).with_config(tags=["final_response"])
        self.rag_chain = (RAG_PROMPT.partial(no_info=NO_INFO_ANSWER) | llm | StrOutputParser()).with_config(
            tags=["final_response"]
        )

    async def stream(
        self,
        message: str,
        language: str,
        parts: list[RoutedPart],
        docs_by_part: list[list[Document]],
        extra_system: str = "",
    ) -> AsyncIterator[str]:
        language = language if language in LANGUAGE_INSTRUCTIONS else "other"
        inputs = {
            "input": message,
            "language_instruction": LANGUAGE_INSTRUCTIONS[language],
            "extra_system": extra_system,
        }

        if all(p.intent in DIRECT_INTENTS for p in parts):
            instruction = " ".join(dict.fromkeys(DIRECT_INSTRUCTIONS[p.intent] for p in parts))
            chain, inputs = self.direct_chain, {**inputs, "direct_instruction": instruction}
        else:
            chain, inputs = self.rag_chain, {**inputs, "parts_block": build_parts_block(parts, docs_by_part)}

        async for token in chain.astream(inputs):
            yield token


def build_parts_block(parts: list[RoutedPart], docs_by_part: list[list[Document]]) -> str:
    """One block per part: its question, answer style and reference chunks."""
    blocks = []
    for i, (part, docs) in enumerate(zip(parts, docs_by_part, strict=False), start=1):
        # Each chunk already starts with "Feature: <Name> — <heading>" and "Route: <path>"
        reference = "\n---\n".join(doc.page_content for doc in docs) or "(none)"
        blocks.append(
            f"Part {i}\n"
            f"Question: {part.question}\n"
            f"Answer style: {ANSWER_STYLES[part.intent]}\n"
            f"Reference text:\n{reference}"
        )
    return "\n\n".join(blocks)
