from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

# Keep unit tests hermetic: never let a developer's real .env or credentials leak
# into a test run.
os.environ.setdefault("VB_LLM_PROVIDER", "echo")
os.environ.setdefault("VB_TTS_ENGINE", "null")
# A developer's real delivery settings must never make a test message a real inbox.
for _name in ("SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID", "VB_SMTP_HOST", "VB_EMAIL_TO", "VB_MAIL_FROM"):
    os.environ[_name] = ""


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 2, 6, 0, tzinfo=UTC)


@pytest.fixture
def yesterday(now: datetime) -> datetime:
    return now - timedelta(days=1)


@pytest.fixture(autouse=True)
def _no_real_backups(monkeypatch, request):
    """`daily` backs up after a brief; a test run must not dump a developer's database."""
    if "test_backup" in request.node.nodeid:  # these test the hook itself
        return
    monkeypatch.setattr(
        "voicebrief.backup.backup_after_generation", lambda: {"ok": True, "skipped": "test"}
    )
