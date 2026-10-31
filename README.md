# CliQ AI Agent

FastAPI service behind CliQ's AI features. The Node backend (`WebsiteBackend`) calls it; the browser never does.

| Feature | Folder | Endpoints |
|---|---|---|
| Help chatbot ("CliQ AI") | `src/help_chat/` | `POST /api/chat`, `POST /api/chat/stream`, `POST /api/chat/feedback`, `GET /api/chat/kb-version`, `POST /api/chat/history/append` |
| Reply suggestions in chats | `src/message_assistant/` | `POST /api/message-ai/respond`, `POST /api/message-ai/respond/stream` |
| Expertise builder (Magic Builder) | `src/expertise/` | `POST /api/expertise/generate` |
| Smart connections and semantic search | `src/matching/` | `POST /api/match/match`, `POST /api/match/precompute`, `POST /api/match/search` |
| Status | `src/system/` | `GET /` (status page), `GET /api/health` |

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env          # then fill in DATABASE_URL and GROQ_API_KEY
.venv/bin/python scripts/check_groq.py   # confirms the key works with both models
```

## Run

```bash
.venv/bin/python app.py       # http://localhost:8000, reloads on code changes
```

`.env` is read once at startup, so restart after changing it.

## Help chat knowledge base

The chatbot answers only from `src/help_chat/knowledge/data/*.txt`. After editing any of them, re-embed:

```bash
.venv/bin/python -m src.help_chat.knowledge.ingest
```

Every file follows the same template, and each `## ` section becomes one searchable chunk:

```
# Feature: <Name>
Route: <page path>
## Summary
## How to <task>        numbered steps, exact UI labels in **bold**
## Limits
## Common problems      "- Problem: ... → Fix: ..."
```

## Project layout

```
src/
├── main.py            creates the app: logging, startup/shutdown, CORS, routers
├── core/              shared infrastructure, never imports from a feature
│   ├── config.py      Settings: the only place that reads environment variables
│   ├── logging.py     log format
│   ├── database.py    the one MongoDB client
│   ├── embeddings.py  the local embedding model (bge-small-en-v1.5)
│   ├── llm.py         Groq model factories
│   └── security.py    X-Internal-Key check for backend-only endpoints
├── help_chat/         router → service → pipeline (routing → retrieval → answering)
│   └── knowledge/     knowledge base files, chunking, vector store, ingest script
├── message_assistant/
├── expertise/
├── matching/
└── system/            status page, health check, keep-alive pinger
tests/
├── unit/              fast, offline: pytest
└── eval/              live answer-quality check against Groq: run_eval.py
scripts/               one-off command-line tools
```

Every feature folder has the same shape:

| File | Holds |
|---|---|
| `router.py` | HTTP endpoints. Validates input, calls the service, nothing else |
| `schemas.py` | Every Pydantic model of the feature |
| `service.py` | The feature's logic; knows nothing about HTTP |
| `dependencies.py` | Hands routes the service instance built at startup |
| `prompts.py` | Every prompt sent to the LLM |

Services are created once in `main.py`'s `lifespan` and stored on `app.state`; routes get them through `Depends(...)`, which also lets tests swap in fakes.

## Coding standards

1. **Settings** come from `get_settings()`; never call `os.getenv` in a feature.
2. **Layers:** router (HTTP) → service (logic) → schemas (data shapes).
3. **Nothing heavy at import time:** models, clients and services are built at startup.
4. **Logging, not print:** `logger = logging.getLogger(__name__)`.
5. **Type hints** on every function, including the return type.
6. **Names:** `snake_case` files and functions, `PascalCase` classes, `UPPER_CASE` constants; files are named after what they do (no `utils.py`).
7. **Small units:** about 300 lines per file, 40 per function.
8. **Prompts** live in the feature's `prompts.py`.
9. **Errors:** catch specific exceptions; a broad `except Exception` only at an HTTP or stream boundary, with `logger.exception`.
10. **Comments** explain why, not what; every module starts with a one-line docstring.

Before every commit:

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/python -m pytest
```

Live answer-quality check (uses Groq quota):

```bash
.venv/bin/python tests/eval/run_eval.py --pause 10
```
