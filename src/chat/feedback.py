import hashlib
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
from langchain_core.documents import Document

from ..rag.loader import CliQDocumentLoader

# A new question reuses an earlier rated answer only when it means the same thing
SIMILARITY_THRESHOLD = float(os.getenv("FEEDBACK_SIMILARITY_THRESHOLD", "0.92"))
# Distinct users who must 👍 an answer (with no 👎) before it is served to everyone
MIN_APPROVALS = max(1, int(os.getenv("FEEDBACK_MIN_APPROVALS", "2")))
# Chunk-level votes needed before retrieval order is changed
MIN_CHUNK_VOTES = 2
MAX_AVOID_ANSWERS = 2
AVOID_EXCERPT_CHARS = 300
CACHE_SECONDS = 60
MAX_RECORDS = 5000


def source_key(doc):
    return f"{doc.metadata.get('section')}::{doc.metadata.get('heading')}"


def source_docs(keys):
    """Rebuilds metadata-only Documents from source keys (enough for follow-up suggestions)."""
    docs = []
    for key in keys or []:
        section, _, heading = key.partition("::")
        docs.append(Document(page_content="", metadata={"section": section, "heading": heading}))
    return docs


def _kb_fingerprint():
    """Changes whenever a knowledge file changes, so votes on outdated answers stop counting."""
    digest = hashlib.sha1()
    for path in sorted(CliQDocumentLoader().data_dir.glob("*.txt")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class FeedbackMemory:
    """Learns from 👍/👎 without retraining the model (no extra LLM calls).

    - Approved answers (MIN_APPROVALS distinct 👍, no 👎) are served again for the same question.
    - Rejected answers are passed to the next attempt as "don't repeat this".
    - Chunks that keep appearing in 👎 answers are pushed down or dropped from retrieval.
    One document per rated AI message, keyed by messageId, so changing a vote replaces it.
    """

    def __init__(self, mongo_client, embeddings, database_name="website_db", collection_name="chat_feedback"):
        self.collection = mongo_client[database_name][collection_name]
        self.collection.create_index("messageId", unique=True)
        self.embeddings = embeddings
        self.kb_version = _kb_fingerprint()
        self._lock = threading.Lock()
        self._records, self._loaded_at = [], 0.0

    # ---------- writes ----------

    def record(self, message_id, user_hash, rating, question, answer, language=None, sources=None, cacheable=False):
        self.current_kb_version()  # stamp the vote with the knowledge base that is live now
        if rating not in (1, -1):
            self.collection.delete_one({"messageId": message_id})
        else:
            self.collection.update_one(
                {"messageId": message_id},
                {"$set": {
                    "messageId": message_id,
                    "userHash": user_hash,
                    "rating": rating,
                    "question": question,
                    "embedding": _unit(self.embeddings.embed_query(question)).tolist(),
                    "answer": answer,
                    "language": language,
                    "sources": list(sources or []),
                    "cacheable": bool(cacheable),
                    "kbVersion": self.kb_version,
                    "updatedAt": datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        with self._lock:
            self._loaded_at = 0.0  # next read reloads

    # ---------- reads ----------

    def current_kb_version(self):
        """The knowledge-base fingerprint, re-read at most every CACHE_SECONDS (docs can change without a restart)."""
        self._current_records()
        return self.kb_version

    def _current_records(self):
        with self._lock:
            if time.monotonic() - self._loaded_at > CACHE_SECONDS:
                self.kb_version = _kb_fingerprint()
                records = list(
                    self.collection.find({"kbVersion": self.kb_version}, {"_id": 0})
                    .sort("updatedAt", -1)
                    .limit(MAX_RECORDS)
                )
                for r in records:
                    r["embedding"] = np.asarray(r["embedding"], dtype=np.float32)
                self._records, self._loaded_at = records, time.monotonic()
            return self._records

    def _similar(self, question):
        records = self._current_records()
        if not records:
            return []
        query = _unit(self.embeddings.embed_query(question))
        return [r for r in records if float(np.dot(query, r["embedding"])) >= SIMILARITY_THRESHOLD]

    @staticmethod
    def _votes_by_answer(records):
        votes = {}
        for r in records:
            entry = votes.setdefault(" ".join(r["answer"].split()), {"record": r, "up": set(), "down": set()})
            (entry["up"] if r["rating"] == 1 else entry["down"]).add(r["userHash"])
        return votes

    def lookup(self, question):
        """Returns {"approved": record | None, "avoid": [answer excerpts]} for this question."""
        votes = self._votes_by_answer(self._similar(question))

        approved = [
            v for v in votes.values()
            if len(v["up"]) >= MIN_APPROVALS and not v["down"] and v["record"]["cacheable"]
        ]
        approved.sort(key=lambda v: len(v["up"]), reverse=True)

        rejected = [v for v in votes.values() if v["down"] and len(v["down"]) >= len(v["up"])]
        rejected.sort(key=lambda v: len(v["down"]), reverse=True)

        return {
            "approved": approved[0]["record"] if approved else None,
            "avoid": [v["record"]["answer"][:AVOID_EXCERPT_CHARS] for v in rejected[:MAX_AVOID_ANSWERS]],
        }

    def chunk_scores(self):
        """{source_key: (ups, downs)} counted once per user."""
        ups, downs = {}, {}
        for r in self._current_records():
            bucket = ups if r["rating"] == 1 else downs
            for key in set(r.get("sources") or []):
                bucket.setdefault(key, set()).add(r["userHash"])
        return {k: (len(ups.get(k, ())), len(downs.get(k, ()))) for k in set(ups) | set(downs)}

    def rank_docs(self, docs, scores):
        """Keeps retrieval order, but lifts chunks users liked and demotes or drops ones they disliked."""
        if not docs or not scores:
            return docs

        def bucket(doc):
            up, down = scores.get(source_key(doc), (0, 0))
            if up + down < MIN_CHUNK_VOTES:
                return 1
            score = (up + 1) / (up + down + 2)  # smoothed share of 👍
            if down >= 3 and score < 0.25:
                return 3  # strong evidence it misleads: drop
            return 0 if score >= 0.6 else (2 if score < 0.4 else 1)

        ranked = sorted(docs, key=bucket)  # stable: same order inside a bucket
        kept = [d for d in ranked if bucket(d) < 3]
        return kept or ranked[:1]
