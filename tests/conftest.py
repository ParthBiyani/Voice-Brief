from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# Keep unit tests hermetic: never let a developer's real .env or credentials leak
# into a test run.
os.environ.setdefault("VB_LLM_PROVIDER", "echo")
os.environ.setdefault("VB_TTS_ENGINE", "null")


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 2, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def yesterday(now: datetime) -> datetime:
    return now - timedelta(days=1)
