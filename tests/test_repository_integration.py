"""Integration tests against the docker-compose Postgres.

Marked `integration` so `make test` stays fast and hermetic; `make test-all` runs
these. They exist because the interesting behaviour here is Postgres behaviour —
ON CONFLICT semantics and advisory locks are not worth faking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from voicebrief.db import session_scope
from voicebrief.db.models import Source
from voicebrief.db.repository import (
    bulk_insert_items,
    due_sources,
    upsert_source,
)
from voicebrief.db.session import advisory_lock
from voicebrief.sources.base import RawItem

pytestmark = pytest.mark.integration


@pytest.fixture
def spec() -> dict:
    return {
        "slug": f"test-{uuid.uuid4().hex[:8]}",
        "name": "Test Source",
        "kind": "rss",
        "endpoint": "https://example.com/feed.xml",
        "trust_weight": 0.5,
        "default_topics": ["testing"],
        "config": {"adapter": "rss"},
    }


@pytest.fixture
def session():
    with session_scope() as s:
        yield s


def _cleanup(session, slug: str) -> None:
    session.execute(text("DELETE FROM source WHERE slug = :slug"), {"slug": slug})


class TestUpsertSource:
    def test_insert_then_update_is_idempotent(self, session, spec):
        try:
            first = upsert_source(session, spec)
            session.flush()
            spec["name"] = "Renamed"
            spec["trust_weight"] = 0.9
            second = upsert_source(session, spec)
            session.flush()

            assert first.id == second.id, "upsert must not create a second row"
            assert second.name == "Renamed"
            assert second.trust_weight == 0.9
        finally:
            _cleanup(session, spec["slug"])

    def test_reseed_preserves_operational_state(self, session, spec):
        """Re-seeding must not wipe failure counters or poll history."""
        try:
            source = upsert_source(session, spec)
            source.consecutive_failures = 3
            source.last_polled_at = datetime.now(timezone.utc)
            session.flush()

            again = upsert_source(session, spec)
            session.flush()
            assert again.consecutive_failures == 3
            assert again.last_polled_at is not None
        finally:
            _cleanup(session, spec["slug"])


class TestDueSources:
    def test_never_polled_source_is_due(self, session, spec):
        try:
            upsert_source(session, spec)
            session.flush()
            assert spec["slug"] in {s.slug for s in due_sources(session)}
        finally:
            _cleanup(session, spec["slug"])

    def test_recently_polled_source_is_not_due(self, session, spec):
        try:
            source = upsert_source(session, spec)
            source.last_polled_at = datetime.now(timezone.utc)
            session.flush()
            assert spec["slug"] not in {s.slug for s in due_sources(session)}
        finally:
            _cleanup(session, spec["slug"])

    def test_failures_back_off_exponentially(self, session, spec):
        """Two failures => 4x the normal interval before we try again."""
        try:
            source = upsert_source(session, spec)
            source.poll_interval_minutes = 60
            source.consecutive_failures = 2
            source.last_polled_at = datetime.now(timezone.utc) - timedelta(minutes=180)
            session.flush()
            assert spec["slug"] not in {s.slug for s in due_sources(session)}

            source.last_polled_at = datetime.now(timezone.utc) - timedelta(minutes=300)
            session.flush()
            assert spec["slug"] in {s.slug for s in due_sources(session)}
        finally:
            _cleanup(session, spec["slug"])


class TestBulkInsertItems:
    def test_duplicate_external_ids_are_skipped_not_raised(self, session, spec):
        try:
            source = upsert_source(session, spec)
            session.flush()
            items = [
                RawItem(
                    external_id="e1",
                    url="https://example.com/1",
                    title="One",
                    published_at=datetime.now(timezone.utc),
                )
            ]
            inserted, skipped = bulk_insert_items(session, source.id, items)
            assert (inserted, skipped) == (1, 0)

            inserted, skipped = bulk_insert_items(session, source.id, items)
            assert (inserted, skipped) == (0, 1)
        finally:
            _cleanup(session, spec["slug"])

    def test_empty_input_is_a_noop(self, session, spec):
        try:
            source = upsert_source(session, spec)
            session.flush()
            assert bulk_insert_items(session, source.id, []) == (0, 0)
        finally:
            _cleanup(session, spec["slug"])


class TestAdvisoryLock:
    def test_lock_is_acquired_and_released(self, session):
        with advisory_lock(session, "voicebrief:test") as acquired:
            assert acquired is True
        with advisory_lock(session, "voicebrief:test") as reacquired:
            assert reacquired is True, "lock must be released on scope exit"
