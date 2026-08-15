"""Shared pytest fixtures.

Default CI runs offline and deterministic. Any test needing a live provider must
be marked ``llm`` and is excluded from the default selection (spec section 24).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the whole default suite off the network and off the billing meter.

    Autouse rather than opt-in: a test that accidentally reaches a paid provider
    is a cost bug that is easy to introduce and hard to notice.
    """
    monkeypatch.setenv("EVIDENCE_ROUTE_OFFLINE", "true")
    yield


@pytest.fixture
def clean_settings_cache() -> Iterator[None]:
    """Reset the cached settings singleton around a test that changes the env."""
    from evidence_route.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def no_azure_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every Azure variable so tests do not depend on the developer's .env."""
    for key in list(os.environ):
        if key.startswith("EVIDENCE_ROUTE_AZURE_"):
            monkeypatch.delenv(key, raising=False)
    yield
