#!/usr/bin/env bash
# This script starts the Website AI Agent using Astral's `uv` for blazing fast performance.

set -e

if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
    if uv run uvicorn src.main:app --reload --port 8000; then
        exit 0
    fi
fi

python3 -m uvicorn src.main:app --reload --port 8000
