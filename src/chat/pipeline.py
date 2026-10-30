import asyncio
import json
import math
import re
import time
import traceback

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from ..core.llm import help_answer_llm
from .errors import LimitBreaker, classify_error, resolve_language, user_message
from .feedback import FeedbackMemory, source_docs, source_key
from .pages import build_extra_system, describe_page, is_personal, refers_to_page, _clean_name
from .suggestions import FollowUpSuggester
from .telemetry import error_code, log_chat

# Only the most recent turns are sent to the LLM, so long chats don't overflow the context window
MAX_HISTORY_MESSAGES = 10
DIRECT_INTENTS = {"greeting", "general_assistance", "out_of_scope"}
FALLBACK_ANSWER = "Sorry, something went wrong on my side. Please try asking that again in a moment."

STATUS_UNDERSTANDING = "Understanding your question…"
STATUS_SEARCHING = "Searching CliQ docs…"
STATUS_WRITING = "Writing answer…"


class ChatPipeline:
    """Runs one help-chat turn step by step and yields (event, data) pairs for SSE.

    Events: status {text}, token {text}, suggestions {items}, error {code},
    done {cacheable, language, parts, sources, fromFeedback}.
    """

    def __init__(self, rag_service, mongo_client, database_name="website_db", collection_name="chat_histories"):
        self.rag = rag_service
        self.mongo_client = mongo_client
        self.database_name = database_name
        self.collection_name = collection_name
        self.collection = mongo_client[database_name][collection_name]
        self.suggester = FollowUpSuggester()
        self.feedback = FeedbackMemory(mongo_client, rag_service.vectorstore.embeddings, database_name)
        self.breaker = LimitBreaker()
        self.model = getattr(help_answer_llm, "model_name", None) or "unknown"

    def history(self, session_id):
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=None,
            database_name=self.database_name,
            collection_name=self.collection_name,
            client=self.mongo_client,
            history_size=MAX_HISTORY_MESSAGES,
        )

    def append_exchange(self, session_id, question, answer):
        self.history(session_id).add_messages([HumanMessage(content=question), AIMessage(content=answer)])

    def drop_last_exchange(self, session_id):
        """Removes the last question/answer pair so Regenerate doesn't see its own previous answer."""
        docs = list(self.collection.find({"SessionId": session_id}).sort("_id", -1).limit(2))
        types = [json.loads(d["History"]).get("type") for d in docs]
        if types == ["ai", "human"]:
            self.collection.delete_many({"_id": {"$in": [d["_id"] for d in docs]}})

    async def stream(self, message, session_id, current_path=None, user_context=None, regenerate=False):
        started_at = time.perf_counter()
        language, parts, docs_by_part, answer = "unknown", [], [], ""
        page, _ = describe_page(current_path)
        error, cacheable = None, False

        try:
            yield "status", {"text": STATUS_UNDERSTANDING}
            if regenerate:
                await asyncio.to_thread(self.drop_last_exchange, session_id)
            chat_history = await asyncio.to_thread(lambda: self.history(session_id).messages)

            # The router has no page parameter, so page-relative questions carry the page in the text
            page_relative = refers_to_page(message)

            # Feedback only applies to standalone, non-personal questions (same rule as the shared cache)
            memory = {"approved": None, "avoid": []}
            if not chat_history and not page_relative and not is_personal(message):
                memory = await asyncio.to_thread(self.feedback.lookup, message)

            approved = memory["approved"]
            if approved and not regenerate:
                # Users already approved an answer to this question: reuse it without calling the LLM
                answer, language = approved["answer"], approved.get("language") or language
                sources = approved.get("sources") or []
                for token in re.findall(r"\S+\s*|\s+", answer):
                    yield "token", {"text": token}
                await asyncio.to_thread(self.append_exchange, session_id, message, answer)
                yield "suggestions", {"items": self.suggester.suggest(message, [source_docs(sources)])}
                cacheable = True
                yield "done", {
                    "cacheable": True, "language": language, "parts": parts,
                    "sources": sources, "fromFeedback": True,
                }
                return

            # While the answer model is rate limited, reply at once instead of waiting on retries that will fail
            blocked = self.breaker.check(self.model)
            if blocked:
                kind, seconds_left = blocked
                error = "rate_limited"
                answer = user_message(kind, resolve_language(language, message), seconds_left)
                for event in self._error_events(error, kind, answer, seconds_left, language, parts):
                    yield event
                return

            routed_message = f"{message} (I am on the {page} page)" if page_relative and page else message
            routed = await asyncio.to_thread(self.rag.route, routed_message, chat_history)
            language, parts = routed["language"], routed["parts"]

            if any(p["intent"] not in DIRECT_INTENTS for p in parts):
                yield "status", {"text": STATUS_SEARCHING}
                docs_by_part = await asyncio.to_thread(self.rag.retrieve, parts)
                scores = await asyncio.to_thread(self.feedback.chunk_scores)
                docs_by_part = [self.feedback.rank_docs(docs, scores) for docs in docs_by_part]
            else:
                docs_by_part = [[] for _ in parts]

            yield "status", {"text": STATUS_WRITING}
            extra_system = build_extra_system(user_context, current_path)
            if memory["avoid"]:
                rejected = "\n".join(f'- "{excerpt}"' for excerpt in memory["avoid"])
                extra_system += (
                    "\n\nEarlier answers to this same question were rated unhelpful by users. "
                    "Do not repeat them; answer more accurately using only the reference text:\n" + rejected
                )
            async for token in self.rag.astream_answer(
                routed_message, language, parts, docs_by_part, extra_system=extra_system
            ):
                if token:
                    answer += token
                    yield "token", {"text": token}

            await asyncio.to_thread(self.append_exchange, session_id, message, answer)

            question = parts[0]["question"] if len(parts) == 1 else message
            yield "suggestions", {"items": self.suggester.suggest(question, docs_by_part)}

            # Answers that depend on this user (their page, counts or name) must not be shared via the cache
            name = _clean_name((user_context or {}).get("firstname"))
            cacheable = bool(answer) and not page_relative and not is_personal(message) and not (
                name and name.lower() in answer.lower()
            )
            sources = list(dict.fromkeys(source_key(d) for docs in docs_by_part for d in docs))
            yield "done", {
                "cacheable": cacheable, "language": language, "parts": parts,
                "sources": sources, "fromFeedback": False,
            }
        except asyncio.CancelledError:
            # The client pressed Stop or disconnected; the upstream LLM stream is cancelled with us
            error = "client_cancelled"
            raise
        except Exception as exc:
            error = error_code(exc)
            if error.startswith("llm_") or error == "rate_limited":
                # Provider errors are expected (quota, timeouts); one line is enough
                print(f"Help chat LLM error ({error}): {str(exc)[:300]}")
            else:
                traceback.print_exc()
            info = classify_error(exc)
            self.breaker.trip(info["model"] or self.model, info["kind"], info["retry_after"])
            text = user_message(info["kind"], resolve_language(language, message), info["retry_after"])
            # A half-streamed answer is kept, with the reason it stopped underneath
            shown = f"\n\n_{text}_" if answer else text
            answer += shown
            for event in self._error_events(error, info["kind"], shown, info["retry_after"], language, parts):
                yield event
        finally:
            log_chat(
                session_id=session_id,
                language=language,
                parts=parts,
                chunks=sum(len(d) for d in docs_by_part),
                model=self.model,
                started_at=started_at,
                error=error,
                current_page=page,
                cacheable=cacheable,
            )

    @staticmethod
    def _error_events(code, kind, text, retry_after, language, parts):
        """The message is sent as a token (so it is shown and saved) and repeated in the error event for the UI."""
        return [
            ("token", {"text": text}),
            ("error", {
                "code": code,
                "kind": kind,
                "retryAfter": math.ceil(retry_after) if retry_after else None,
                "message": text.strip().strip("_"),
            }),
            ("done", {"cacheable": False, "language": language, "parts": parts}),
        ]

    async def ask(self, message, session_id, current_path=None, user_context=None):
        """Collects a full turn; used by /chat and the eval runner."""
        result = {"answer": "", "suggestions": [], "statuses": [], "meta": {}, "error": None}
        async for event, data in self.stream(message, session_id, current_path, user_context):
            if event == "token":
                result["answer"] += data["text"]
            elif event == "status":
                result["statuses"].append(data["text"])
            elif event == "suggestions":
                result["suggestions"] = data["items"]
            elif event == "error":
                result["error"] = data["code"]
            elif event == "done":
                result["meta"] = data
        return result
