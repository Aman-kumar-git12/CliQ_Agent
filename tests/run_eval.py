"""Runs the CliQ help-chat eval set against the real agent pipeline.

Usage (from WebsiteAgent/):
    .venv/bin/python tests/run_eval.py                 # all cases
    .venv/bin/python tests/run_eval.py -k hinglish     # only cases whose id contains "hinglish"
    .venv/bin/python tests/run_eval.py --local-kb      # search src/rag/data in memory instead of Atlas
    .venv/bin/python tests/run_eval.py --verbose       # print every answer
    .venv/bin/python tests/run_eval.py --pause 10      # wait between cases (Groq free tier has a tokens/minute limit)

A case that fails only because of a Groq rate limit is retried after a wait (--retries, default 2),
so the score measures answer quality rather than quota.

Each case uses a fresh session id; its chat history is deleted afterwards.
Exit code is 1 when any case fails, so it can gate CI.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.chat.pages import LINKABLE_PAGES  # noqa: E402

ALLOWED_ROUTES = {route for route, _, _ in LINKABLE_PAGES}
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
HINGLISH_WORDS = {
    "kaise", "karo", "karu", "karein", "kare", "aap", "apna", "apni", "apne", "hai", "hain", "ho", "ka", "ki",
    "ke", "ko", "se", "par", "pe", "mein", "me", "aur", "bhi", "nahi", "toh", "phir", "yeh", "ye", "wo", "jo",
    "dabao", "kholo", "khol", "chuno", "milega", "dikhega", "jayega", "sakte", "skte", "bas", "liye", "wala",
}


def normalize(text):
    """Folds the typographic characters gpt-oss likes (non-breaking hyphens, curly quotes) to ASCII."""
    return (
        text.replace("‑", "-").replace("‐", "-").replace("–", "-").replace("—", "-")
        .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        .replace(" ", " ").replace(" ", " ").lower()
    )


def detect_language(text):
    if DEVANAGARI_RE.search(text):
        return "hindi"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for w in words if w in HINGLISH_WORDS)
    return "hinglish" if hits >= 3 else "english"


def check_answer(case, answer, defaults):
    failures = []
    text = normalize(answer)

    for item in case.get("must_include", []):
        options = item if isinstance(item, list) else [item]
        if not any(normalize(o) in text for o in options):
            failures.append(f"missing {options if len(options) > 1 else options[0]!r}")

    banned = list(case.get("must_not_include", []))
    if not case.get("allow_no_info"):
        banned += defaults.get("must_not_include", [])
    for item in banned:
        if normalize(item) in text:
            failures.append(f"contains forbidden {item!r}")

    for label, href in LINK_RE.findall(answer):
        if href not in ALLOWED_ROUTES:
            failures.append(f"link outside allowlist: [{label}]({href})")

    expected = case.get("language")
    if expected:
        actual = detect_language(answer)
        if actual != expected:
            failures.append(f"language {actual}, expected {expected}")

    if not answer.strip():
        failures.append("empty answer")
    return failures


def use_local_kb(vectorstore):
    """Replaces Atlas search with an in-memory search over the local knowledge files."""
    import numpy as np
    from src.rag.loader import CliQDocumentLoader

    with redirect_stdout(StringIO()):
        chunks = CliQDocumentLoader().load_documents()
    vectors = np.array(vectorstore.embeddings.embed_documents([c.page_content for c in chunks]))

    def search(query, k=4, pre_filter=None):
        sims = vectors @ np.array(vectorstore.embeddings.embed_query(query))
        section = pre_filter["section"]["$eq"] if pre_filter else None
        order = [i for i in np.argsort(-sims) if section is None or chunks[i].metadata["section"] == section]
        return [chunks[i] for i in order[:k]]

    vectorstore.similarity_search = search
    return len(chunks)


async def run_case(agent, case, defaults, verbose, retries=2, retry_wait=30):
    for attempt in range(retries + 1):
        result = await _run_case_once(agent, case, defaults, verbose)
        if "pipeline error rate_limited" not in result["failures"] or attempt == retries:
            result["attempts"] = attempt + 1
            return result
        print(f"      (rate limited, retrying {case['id']} in {retry_wait}s)", flush=True)
        await asyncio.sleep(retry_wait)


async def _run_case_once(agent, case, defaults, verbose):
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    turns = case.get("turns") or [case["question"]]
    user_context = case.get("user_context", defaults.get("user_context"))
    current_path = case.get("current_path", defaults.get("current_path"))

    result, latency_ms = None, 0
    log = StringIO()
    try:
        for turn in turns:
            started = time.perf_counter()
            # The pipeline prints routing/retrieval lines; keep them out of the report unless verbose
            with redirect_stdout(log if not verbose else sys.stdout):
                result = await agent.ask(turn, session_id, current_path, user_context)
            latency_ms = (time.perf_counter() - started) * 1000  # latency of the checked (last) turn
    finally:
        agent.pipeline.collection.delete_many({"SessionId": session_id})

    answer = result["answer"]
    failures = check_answer(case, answer, defaults)
    if result.get("error"):
        failures.append(f"pipeline error {result['error']}")

    telemetry = [line for line in log.getvalue().splitlines() if line.startswith('{"event": "aichat"')]
    return {
        "id": case["id"],
        "question": turns[-1],
        "answer": answer,
        "suggestions": result.get("suggestions", []),
        "statuses": result.get("statuses", []),
        "latency_ms": latency_ms,
        "failures": failures,
        "telemetry": telemetry[-1] if telemetry else None,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="only run cases whose id contains this text")
    parser.add_argument("--local-kb", action="store_true", help="search local knowledge files instead of Atlas")
    parser.add_argument("--verbose", action="store_true", help="print every answer")
    parser.add_argument("--json", help="write full results to this file")
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait between cases")
    parser.add_argument("--retries", type=int, default=2, help="retries for a case that hit a rate limit")
    args = parser.parse_args()

    spec = json.loads((Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8"))
    defaults, cases = spec.get("defaults", {}), spec["cases"]
    if args.k:
        cases = [c for c in cases if args.k in c["id"]]

    with redirect_stdout(StringIO()):
        from src.core.agent import CliQAgent
        agent = CliQAgent()
    if args.local_kb:
        print(f"Using local knowledge base ({use_local_kb(agent.vectorstore)} chunks in memory)")
    print(f"Running {len(cases)} cases\n")

    results = []
    for case in cases:
        r = await run_case(agent, case, defaults, args.verbose, retries=args.retries)
        results.append(r)
        mark = "PASS" if not r["failures"] else "FAIL"
        print(f"{mark}  {r['id']:<26} {r['latency_ms']:>7.0f} ms", flush=True)
        for failure in r["failures"]:
            print(f"      - {failure}")
        if r["failures"] or args.verbose:
            print("      answer: " + r["answer"].replace("\n", "\n              ")[:1200])
            print(f"      chips:  {r['suggestions']}", flush=True)
        if args.pause and case is not cases[-1]:
            await asyncio.sleep(args.pause)

    passed = sum(1 for r in results if not r["failures"])
    avg = sum(r["latency_ms"] for r in results) / max(1, len(results))
    print(f"\nScore: {passed}/{len(results)} passed ({passed / max(1, len(results)):.0%})")
    print(f"Average latency: {avg:.0f} ms (last turn of each case)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
