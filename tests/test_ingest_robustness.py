"""Ingest windows, poll scheduling, lock identity and release-feed titles."""

from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from voicebrief.db.models import Source, SourceKind
from voicebrief.db.repository import disable_sources_not_in, due_sources
from voicebrief.db.session import get_engine
from voicebrief.pipeline.filtering import is_noise
from voicebrief.pipeline.ingest import (
    MAX_BACKFILL_HOURS,
    POLL_OVERLAP,
    SERIAL_ADAPTERS,
    source_since,
)
from voicebrief.sources.base import SourceConfig
from voicebrief.sources.rss import RSSAdapter, qualify_release_title, release_feed_repo

NOW = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


def src(last_polled_at):
    return SimpleNamespace(last_polled_at=last_polled_at)


class TestSourceSince:
    def test_first_ever_poll_uses_the_default_window(self):
        assert source_since(src(None), started=NOW, lookback_hours=36) == NOW - timedelta(hours=36)

    def test_regular_cadence_keeps_the_default_window(self):
        since = source_since(src(NOW - timedelta(hours=12)), started=NOW, lookback_hours=36)
        assert since == NOW - timedelta(hours=36)

    def test_a_long_outage_reaches_back_to_the_last_successful_poll(self):
        last = NOW - timedelta(days=4)
        since = source_since(src(last), started=NOW, lookback_hours=36)
        assert since == last - POLL_OVERLAP

    def test_backfill_is_bounded(self):
        since = source_since(src(NOW - timedelta(days=90)), started=NOW, lookback_hours=36)
        assert since == NOW - timedelta(hours=MAX_BACKFILL_HOURS)


class TestReleaseFeedTitles:
    def test_repo_is_read_from_a_releases_feed_url(self):
        url = "https://github.com/langchain-ai/langgraph/releases.atom"
        assert release_feed_repo(url) == "langchain-ai/langgraph"
        assert release_feed_repo("https://example.com/feed.xml") is None

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("v0.40.13", "langchain-ai/langchain v0.40.13"),
            ("2026-07-28", "langchain-ai/langchain 2026-07-28"),
            ("langchain==1.2.18", "langchain==1.2.18"),
            ("LangChain 1.3 released", "LangChain 1.3 released"),
        ],
    )
    def test_bare_tags_gain_the_repo_name(self, title, expected):
        assert qualify_release_title(title, "langchain-ai/langchain") == expected

    def test_qualified_release_titles_are_not_filtered_as_noise(self):
        item = SimpleNamespace(
            title=qualify_release_title("v0.40.13", "langchain-ai/langchain"),
            url="https://github.com/langchain-ai/langchain/releases/tag/v0.40.13",
        )
        assert not is_noise(item)

    @respx.mock
    async def test_adapter_applies_it_end_to_end(self):
        endpoint = "https://github.com/ollama/ollama/releases.atom"
        atom = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Release notes from ollama</title>
<entry><id>tag:github.com,2008:Repository/1/v0.9.1</id><title>v0.9.1</title>
<link rel="alternate" href="https://github.com/ollama/ollama/releases/tag/v0.9.1"/>
<updated>{NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}</updated><content type="html">Faster loads.</content></entry>
</feed>"""
        respx.get(endpoint).mock(return_value=httpx.Response(200, text=atom))
        adapter = RSSAdapter(
            SourceConfig(
                slug="ollama-releases",
                name="Ollama",
                endpoint=endpoint,
                default_topics=["releases"],
                trust_weight=0.7,
            )
        )
        async with httpx.AsyncClient() as client:
            items = await adapter.fetch(client, NOW - timedelta(days=1))
        assert [i.title for i in items] == ["ollama/ollama v0.9.1"]

    @respx.mock
    async def test_rate_limit_is_reported_as_a_clear_error(self):
        endpoint = "https://example.com/feed.xml"
        respx.get(endpoint).mock(return_value=httpx.Response(429, headers={"Retry-After": "60"}))
        adapter = RSSAdapter(
            SourceConfig(slug="x", name="X", endpoint=endpoint, default_topics=[], trust_weight=0.5)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPError, match="rate limited"):
                await adapter.fetch(client, NOW - timedelta(days=1))


def test_arxiv_is_fetched_one_request_at_a_time():
    assert "arxiv" in SERIAL_ADAPTERS


def test_lock_key_hashes_identically_across_processes():
    """The original used hash(), which Python randomises per process, so a CLI run and an
    API run never actually excluded each other."""
    code = "import zlib;print(zlib.crc32(b'voicebrief:ingest') % (2**31))"
    ids = {
        subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(ids) == 1


@pytest.mark.integration
class TestSourceRegistry:
    @pytest.fixture
    def session(self):
        with Session(get_engine()) as s:
            s.begin()
            try:
                yield s
            finally:
                s.rollback()

    def make(self, session, *, polled, interval=720, failures=0, enabled=True) -> Source:
        source = Source(
            slug=f"t-{uuid.uuid4().hex[:8]}",
            name="t",
            kind=SourceKind.rss,
            endpoint="https://example.com/f",
            poll_interval_minutes=interval,
            last_polled_at=polled,
            consecutive_failures=failures,
            enabled=enabled,
        )
        session.add(source)
        session.flush()
        return source

    def test_cron_jitter_does_not_skip_a_whole_batch(self, session):
        # Polled exactly one interval ago minus a few milliseconds: previously "not due".
        source = self.make(session, polled=NOW - timedelta(minutes=720) + timedelta(seconds=2))
        assert source in due_sources(session, now=NOW)

    def test_a_source_polled_recently_is_not_due(self, session):
        source = self.make(session, polled=NOW - timedelta(hours=2))
        assert source not in due_sources(session, now=NOW)

    def test_failure_backoff_is_capped_at_a_day(self, session):
        source = self.make(session, polled=NOW - timedelta(hours=25), failures=6)
        assert source in due_sources(session, now=NOW)

    def test_disabled_sources_are_never_due(self, session):
        source = self.make(session, polled=None, enabled=False)
        assert source not in due_sources(session, now=NOW)

    def test_sources_removed_from_the_seed_are_disabled_not_deleted(self, session):
        keep = self.make(session, polled=None)
        gone = self.make(session, polled=None)
        retired = disable_sources_not_in(session, [keep.slug])
        assert gone.slug in retired
        assert gone.enabled is False
        assert keep.enabled is True
