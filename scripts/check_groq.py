"""Checks that GROQ_API_KEY works with both configured models (run after changing the key).

Usage (from WebsiteAgent/):
    .venv/bin/python scripts/check_groq.py
"""

import sys
from pathlib import Path

import groq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings  # noqa: E402
from src.core.llm import get_classifier_llm, get_response_llm  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env")
        return 1

    ok = True
    for name, llm in (("classifier", get_classifier_llm()), ("response", get_response_llm())):
        try:
            reply = llm.invoke("Reply with the single word: ready")
            print(f"OK    {name:<10} {llm.model_name}: {reply.content.strip()[:40]}")
        except groq.APIError as exc:
            ok = False
            print(f"FAIL  {name:<10} {llm.model_name}: {type(exc).__name__}: {str(exc)[:200]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
