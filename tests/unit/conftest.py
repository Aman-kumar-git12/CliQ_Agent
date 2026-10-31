"""Shared fixtures: unit tests never touch the network, MongoDB or Groq."""

import os

import pytest

# Settings require a database URL; unit tests never connect to it
os.environ.setdefault("DATABASE_URL", "mongodb://localhost:27017")


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch):
    """Sets env variables for one test and re-reads Settings."""
    from src.core.config import get_settings

    def apply(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield apply
    get_settings.cache_clear()
