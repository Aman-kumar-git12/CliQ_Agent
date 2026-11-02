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
.venv/bin/python scripts/download_model.py   # puts the embedding model in .cache/ (once per build)
.venv/bin/python scripts/check_groq.py   # confirms the key works with both models
```

## Run

```bash
.venv/bin/python app.py       # http://localhost:8000, reloads on code changes
```

`.env` is read once at startup, so restart after changing it.

## Operations

**CI:** `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check` and the unit tests on every push and pull request, and checks that the Docker image builds (nothing is pushed). The tests need no real secrets.

**Rotating `AGENT_INTERNAL_KEY` without downtime** (the agent accepts either key while both are set):

1. On the agent, set `AGENT_INTERNAL_KEY=<new>` and `AGENT_INTERNAL_KEY_PREVIOUS=<old>`, then restart it.
2. On the backend, set its `AGENT_INTERNAL_KEY=<new>` and restart it.
3. On the agent, remove `AGENT_INTERNAL_KEY_PREVIOUS` and restart it.

**Request ids:** every response carries `X-Request-Id`, taken from the backend's request or generated, and every log line shows it (`[<id>]`, or `-` outside a request). `LOG_FORMAT=json` prints one JSON object per line instead.

**Error monitoring (optional):** set `SENTRY_DSN` to enable Sentry; `SENTRY_ENVIRONMENT` (default `production`) and `SENTRY_TRACES_SAMPLE_RATE` (default `0.1`) tune it. No PII, request bodies or the `X-Internal-Key` header are sent. Empty `SENTRY_DSN` (the default, and always in tests) disables it.

**Blocking work:** embedding, matching and LLM-classifier calls are synchronous, so `async` code runs them with `asyncio.to_thread`; the event loop stays free for health checks while a precompute or ranking runs.

**Response time and failures:**

- Every Groq call has a timeout (`LLM_TIMEOUT_SECONDS`, 20; `LLM_CLASSIFIER_TIMEOUT_SECONDS`, 8 for the help chat router and the message assistant's classifiers). The SDK's own retries are off (`LLM_MAX_RETRIES=0`), because they also wait out 429s for up to a minute; the code retries only fast failures (5xx, dropped connection, gpt-oss's stray tool call) once, and only before the first token. A rate limit reaches the user at once as a clear message.
- gpt-oss models get `reasoning_effort=low` (`LLM_REASONING_EFFORT`), which makes answers start much sooner.
- The message assistant asks the small classifier model (never the large response model) whether a request is safe and how to route it, and runs both checks at the same time.
- MongoDB calls give up after 5 s (`src/core/database.py`). Without the database the help chat still answers (without history or feedback); only a knowledge base search failure becomes an error, and such answers are never cacheable.
- `scripts/download_model.py` stores the embedding model in `.cache/fastembed` at build time (the Dockerfile runs it). On Render's native Python runtime, use the Build Command `pip install -r requirements.txt && python scripts/download_model.py`; otherwise every cold start downloads the model again.
- The keep-alive pings localhost every 12 minutes, which does not stop Render's free tier from putting the
  agent to sleep when idle (the first AI reply after a quiet period then waits for it to wake). To keep it
  awake, set `KEEPALIVE_URL` to the agent's public URL; only one free Render service can stay awake all month.

## Stored profile embeddings

Matching saves every profile embedding in the `profile_embeddings` collection (one document per user:
the profile-content hash, the model name and the vector as packed float32, ~1.5 KB). They are loaded into
memory at startup, so a restart or a Render instance waking up re-embeds nobody; a profile is embedded
again only when its content changes. If MongoDB is unreachable, matching still works and just embeds again.

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
│   ├── logging.py     log format (text or JSON), with the request id on every line
│   ├── request_id.py  X-Request-Id middleware and the log filter
│   ├── monitoring.py  optional Sentry setup
│   ├── database.py    the one MongoDB client
│   ├── embeddings.py  the local embedding model (bge-small-en-v1.5)
│   ├── llm.py         Groq model factories
│   └── security.py    X-Internal-Key check, required on every feature endpoint
├── help_chat/         router → service → pipeline (routing → retrieval → answering)
│   └── knowledge/     knowledge base files, chunking, vector store, ingest script
├── message_assistant/
├── expertise/
├── matching/          ranking + people search; vector_store.py keeps profile embeddings in MongoDB
└── system/            status page, health check, keep-alive pinger
tests/
├── unit/              fast, offline: pytest
└── eval/              live answer-quality check against Groq: run_eval.py
scripts/               one-off command-line tools (check_groq.py, download_model.py)
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
