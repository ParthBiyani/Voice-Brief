"""The orchestrator's contract is about failure, not success: one broken source must
never take down a run. These tests exercise that boundary directly."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from voicebrief.db.models import SourceKind
from voicebrief.pipeline.ingest import _apply_topics, _fetch_one
from voicebrief.sources import registry
from voicebrief.sources.base import RawItem, SourceAdapter


class _FakeSource:
    """Just enough of the Source row for the fetch boundary."""

    def __init__(self, slug: str, adapter: str, topics: list[str] | None = None):
        self.slug = slug
        self.kind = SourceKind.api
        self.name = slug
        self.endpoint = "https://example.com"
        self.config = {"adapter": adapter}
        self.default_topics = topics or []
        self.trust_weight = 0.5


@pytest.fixture(scope="module", autouse=True)
def _adapters():
    @registry.register
    class Good(SourceAdapter):
        kind = "test_good"

        async def fetch(self, client, since):
            return [
                RawItem(
                    external_id="1",
                    url="https://example.com/1",
                    title="Fine",
                    published_at=datetime.now(timezone.utc),
                )
            ]

    @registry.register
    class Boom(SourceAdapter):
        kind = "test_boom"

        async def fetch(self, client, since):
            raise ValueError("upstream changed its schema")

    @registry.register
    class Http500(SourceAdapter):
        kind = "test_http"

        async def fetch(self, client, since):
            raise httpx.ConnectError("connection refused")

    @registry.register
    class Slow(SourceAdapter):
        kind = "test_slow"

        async def fetch(self, client, since):
            await asyncio.sleep(60)
            return []


async def _run(source):
    async with httpx.AsyncClient() as client:
        return await _fetch_one(
            client, source, datetime.now(timezone.utc), asyncio.Semaphore(1)
        )


class TestFetchBoundary:
    async def test_healthy_source_returns_items(self):
        stats, items = await _run(_FakeSource("good", "test_good"))
        assert stats.ok and stats.fetched == 1 and len(items) == 1

    async def test_adapter_exception_is_captured_not_raised(self):
        stats, items = await _run(_FakeSource("boom", "test_boom"))
        assert not stats.ok
        assert "upstream changed its schema" in stats.error
        assert items == []

    async def test_network_error_is_captured(self):
        stats, items = await _run(_FakeSource("net", "test_http"))
        assert not stats.ok and "ConnectError" in stats.error and items == []

    async def test_unknown_adapter_is_captured(self):
        stats, _ = await _run(_FakeSource("nope", "does_not_exist"))
        assert not stats.ok and "No adapter registered" in stats.error

    async def test_timeout_is_bounded(self, monkeypatch):
        monkeypatch.setattr("voicebrief.pipeline.ingest.PER_SOURCE_TIMEOUT", 0.05)
        stats, _ = await _run(_FakeSource("slow", "test_slow"))
        assert not stats.ok and "timeout" in stats.error

    async def test_one_bad_source_does_not_stop_the_others(self):
        sources = [
            _FakeSource("good", "test_good"),
            _FakeSource("boom", "test_boom"),
            _FakeSource("good2", "test_good"),
        ]
        async with httpx.AsyncClient() as client:
            sem = asyncio.Semaphore(3)
            results = await asyncio.gather(
                *(_fetch_one(client, s, datetime.now(timezone.utc), sem) for s in sources)
            )
        assert [r[0].ok for r in results] == [True, False, True]


class TestTopicFallback:
    def test_source_topics_fill_in_when_adapter_infers_none(self):
        item = RawItem(
            external_id="1",
            url="https://e.com",
            title="t",
            published_at=datetime.now(timezone.utc),
        )
        _apply_topics([item], _FakeSource("s", "test_good", ["agentic-ai"]))
        assert item.topics == ["agentic-ai"]

    def test_adapter_topics_win_over_source_defaults(self):
        item = RawItem(
            external_id="1",
            url="https://e.com",
            title="t",
            published_at=datetime.now(timezone.utc),
            topics=["specific"],
        )
        _apply_topics([item], _FakeSource("s", "test_good", ["generic"]))
        assert item.topics == ["specific"]
