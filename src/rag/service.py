import re

import groq
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ValidationError

from ..core.llm import help_answer_llm, help_router_llm
from .loader import CliQDocumentLoader
from .prompt import (
    DIRECT_RESPONSE_PROMPT,
    LANGUAGE_INSTRUCTIONS,
    NO_INFO_ANSWER,
    RAG_PROMPT,
    ROUTER_PROMPT,
    SECTION_HINTS,
)
from .vectorstore import CliQVectorStore

# gpt-oss models on Groq occasionally emit a stray tool call ("Tool choice is none, but model called a tool")
LLM_ATTEMPTS = 3
# Only transient failures are worth retrying. A 400 (e.g. JSON mode's json_validate_failed on a
# prompt-injection message) fails the same way every time, so retrying only adds seconds.
# Rate limits are left to the Groq SDK's own retry and the pipeline's limit breaker, so a daily limit fails fast.
TRANSIENT_ERRORS = (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)

RETRIEVAL_K = 4

MAX_PARTS = 3
INTENTS = {"steps", "information", "navigation", "greeting", "general_assistance", "out_of_scope"}
DIRECT_INTENTS = {"greeting", "general_assistance", "out_of_scope"}

GREETINGS = {
    "hi", "hii", "hello", "hey", "hey there", "hello there", "namaste",
    "how are you", "how r u", "who are you",
}

# Per-intent answer style, added only for the intents actually present so the model
# never sees (and echoes) rules for intents that aren't in this question.
ANSWER_STYLES = {
    "steps": "numbered steps in the order the user clicks, with UI labels in **bold** exactly as written in the reference text",
    "navigation": "the path in backticks, like `Settings > Edit Profile`, then one short sentence",
    "information": "up to 3 short bullet points, only if they add something",
    "greeting": "one short friendly line",
    "general_assistance": "one polite line saying you only help with using CliQ",
    "out_of_scope": "one polite line saying you only help with CliQ",
}

DIRECT_INSTRUCTIONS = {
    "greeting": "Greet the user briefly and offer help with CliQ.",
    "general_assistance": (
        "Politely explain that you focus on helping with CliQ features and can't help with general writing tasks."
    ),
    "out_of_scope": "Politely explain that you can only help with CliQ platform features and guidance.",
}


class RoutedPart(BaseModel):
    question: str = ""
    intent: str = "information"
    section: str = "cliq_details"


class RouterOutput(BaseModel):
    language: str = "other"
    parts: list[RoutedPart] = Field(default_factory=list)


def _normalize(text):
    """Lowercases and strips punctuation/emoji so "Hello :)" matches "hello"."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _guess_language(text):
    return "hindi" if re.search(r"[ऀ-ॿ]", text) else "other"


class CliQRAGService:
    """RAG pipeline: one router call, per-part retrieval, one grounded answer call."""

    def __init__(self, vectorstore: CliQVectorStore):
        self.vectorstore = vectorstore

        # Valid sections are the knowledge base files, so new files are picked up automatically
        self.sections = sorted(p.stem for p in CliQDocumentLoader().data_dir.glob("*.txt")) or ["cliq_details"]
        self.sections_text = "\n".join(
            f"- {s}: {SECTION_HINTS[s]}" if s in SECTION_HINTS else f"- {s}" for s in self.sections
        )

        # 1. Router: rewrite + split + classify in one JSON call
        self.router_chain = (ROUTER_PROMPT | help_router_llm | StrOutputParser()).with_retry(
            retry_if_exception_type=TRANSIENT_ERRORS,
            stop_after_attempt=LLM_ATTEMPTS,
        )

        # 2. Answer chains; only these are tagged so the stream route forwards their tokens
        # {extra_system} is an optional block (user context, link rules) a caller can inject.
        direct_chain = DIRECT_RESPONSE_PROMPT.partial(extra_system="") | help_answer_llm | StrOutputParser()
        rag_chain = RAG_PROMPT.partial(no_info=NO_INFO_ANSWER, extra_system="") | help_answer_llm | StrOutputParser()
        self.direct_chain, self.rag_chain = direct_chain, rag_chain
        self.final_answer_chain = RunnableBranch(
            (
                lambda x: x["route"]["mode"] == "direct",
                direct_chain.with_config(tags=["final_response"]),
            ),
            rag_chain.with_config(tags=["final_response"]),
        ).with_retry(retry_if_exception_type=TRANSIENT_ERRORS, stop_after_attempt=LLM_ATTEMPTS)

    def _parse_route(self, raw, message):
        """Validates router JSON, tolerating code fences; never falls back to out_of_scope."""
        start, end = raw.find("{"), raw.rfind("}")
        try:
            routed = RouterOutput.model_validate_json(raw[start:end + 1]) if start != -1 else None
        except (ValidationError, ValueError):
            routed = None

        if not routed or not routed.parts:
            # Unclear routing: still try the knowledge base with the raw message
            return _guess_language(message), [RoutedPart(question=message)]

        parts = []
        for part in routed.parts[:MAX_PARTS]:
            parts.append(RoutedPart(
                question=part.question.strip() or message,
                intent=part.intent if part.intent in INTENTS else "information",
                section=part.section if part.section in self.sections else "cliq_details",
            ))
        language = routed.language if routed.language in LANGUAGE_INSTRUCTIONS else "other"
        return language, parts

    def route(self, message, chat_history=None):
        """Public step 1: returns {"language": str, "parts": [{"question", "intent", "section"}]}."""
        language, parts = self._classify(message, chat_history or [])
        return {"language": language, "parts": [p.model_dump() for p in parts]}

    def _classify(self, message, chat_history):

        if _normalize(message) in GREETINGS:
            language, parts = _guess_language(message), [RoutedPart(question=message, intent="greeting")]
        else:
            try:
                raw = self.router_chain.invoke({
                    "input": message,
                    "chat_history": chat_history,
                    "sections": self.sections_text,
                })
                language, parts = self._parse_route(raw, message)
            except groq.BadRequestError as e:
                # JSON mode rejects output it can't validate; in practice that is an attempt
                # to make the router break format (prompt injection), so refuse politely.
                print(f"Router rejected the message, treating it as out of scope: {e}")
                language, parts = _guess_language(message), [RoutedPart(question=message, intent="out_of_scope")]
            except Exception as e:
                print(f"Routing failed, answering from the raw message: {e}")
                language, parts = self._parse_route("", message)

        print(f"Flow: language='{language}', parts={[(p.intent, p.section, p.question) for p in parts]}")
        return language, parts

    def _plan(self, language, parts, docs_by_part=None):
        """Builds the answer-prompt inputs. docs_by_part: one doc list per part (retrieved if None)."""
        route = {
            "mode": "rag",
            "language_instruction": LANGUAGE_INSTRUCTIONS[language],
            "direct_instruction": "",
            "parts_block": "",
        }

        if all(p.intent in DIRECT_INTENTS for p in parts):
            route["mode"] = "direct"
            route["direct_instruction"] = " ".join(dict.fromkeys(DIRECT_INSTRUCTIONS[p.intent] for p in parts))
            return route

        if docs_by_part is None:
            docs_by_part = self.retrieve(parts)
        blocks = []
        for i, (part, docs) in enumerate(zip(parts, docs_by_part), start=1):
            # Each chunk starts with "Feature: <Name> — <heading>" and "Route: <path>".
            reference = "\n---\n".join(doc.page_content for doc in docs) or "(none)"
            blocks.append(
                f"Part {i}\n"
                f"Question: {part.question}\n"
                f"Answer style: {ANSWER_STYLES[part.intent]}\n"
                f"Reference text:\n{reference}"
            )
        route["parts_block"] = "\n\n".join(blocks)
        return route

    def _route(self, info):
        language, parts = self._classify(info["input"], info.get("chat_history", []))
        return self._plan(language, parts)

    def retrieve(self, parts):
        """Public step 2: one list of Documents per part (empty for greeting/out-of-scope parts).
        Each Document has metadata section, heading, route, feature and source."""
        parts = [p if isinstance(p, RoutedPart) else RoutedPart(**p) for p in parts]
        return [[] if p.intent in DIRECT_INTENTS else self._retrieve(p.question, p.section) for p in parts]

    async def astream_answer(self, message, language, parts, docs_by_part, extra_system=""):
        """Public step 3: async iterator of answer text tokens."""
        parts = [p if isinstance(p, RoutedPart) else RoutedPart(**p) for p in parts]
        language = language if language in LANGUAGE_INSTRUCTIONS else "other"
        plan = self._plan(language, parts, docs_by_part)
        inputs = {"input": message, "extra_system": extra_system, **plan}
        chain = self.direct_chain if plan["mode"] == "direct" else self.rag_chain
        async for token in chain.with_config(tags=["final_response"]).astream(inputs):
            yield token

    def _retrieve(self, question, section, k=RETRIEVAL_K):
        """k chunks from the routed section plus k from all sections, de-duplicated.

        The router can pick a section that does not hold the answer (e.g. sending a
        request is documented under "find", not "requests"), so results from the whole
        collection are always merged in.
        """
        section_docs = self.vectorstore.similarity_search(
            question, k=k, pre_filter={"section": {"$eq": section}}
        )
        global_docs = self.vectorstore.similarity_search(question, k=k)

        docs, seen = [], set()
        for doc in section_docs + global_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
        print("Retrieved: " + "; ".join(
            f"{d.metadata.get('section')} › {d.metadata.get('heading')}" for d in docs
        ))
        return docs

    def get_chain(self):
        """Main RAG pipeline"""

        return (
            RunnablePassthrough.assign(route=RunnableLambda(self._route))
            | RunnablePassthrough.assign(
                language_instruction=lambda x: x["route"]["language_instruction"],
                direct_instruction=lambda x: x["route"]["direct_instruction"],
                parts_block=lambda x: x["route"]["parts_block"],
            )
            | RunnablePassthrough.assign(answer=self.final_answer_chain)
        )
