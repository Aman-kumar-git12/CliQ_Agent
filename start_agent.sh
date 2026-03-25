#!/usr/bin/env bash
# This script starts the Website AI Agent using Astral's `uv` for blazing fast performance.

uv run uvicorn src.main:app --reload --port 8000
