from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from voicebrief.sources.arxiv import ArxivAdapter
from voicebrief.sources.base import SourceConfig

ENDPOINT = "http://export.arxiv.org/api/query"
SAMPLE = (Path(__file__).parent / "fixtures" / "arxiv_sample.xml").read_text(encoding="utf-8")


@pytest.fixture
def adapter() -> ArxivAdapter:
    return ArxivAdapter(
        SourceConfig(
            slug="arxiv-test",
            name="arXiv",
            endpoint=ENDPOINT,
            config={"categories": ["cs.AI"], "max_results": 10},
        )
    )


@pytest.fixture
def since() -> datetime:
    return datetime(2026, 6, 1, tzinfo=timezone.utc)


@respx.mock
async def test_parses_entries_into_raw_items(adapter, since):
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, text=SAMPLE))
    async with httpx.AsyncClient() as client:
        items = await adapter.fetch(client, since)

    assert len(items) == 1, "the May paper is outside the since window"
    item = items[0]
    assert item.external_id == "2606.01234v1"
    assert item.title == "Durable Execution for Long-Running Agent Graphs"
    assert item.summary == "We present a checkpointing scheme that survives process restarts."
    assert item.author == "A. Researcher, B. Coauthor"
    assert item.url.endswith("2606.01234v1")


@respx.mock
async def test_maps_categories_to_topics(adapter, since):
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, text=SAMPLE))
    async with httpx.AsyncClient() as client:
        items = await adapter.fetch(client, since)
    assert items[0].topics == ["agentic-ai", "machine-learning", "research"]


@respx.mock
async def test_cross_listed_papers_are_deduplicated(since):
    """The same id returned for two categories must yield one item, not two —
    otherwise the semantic dedup stage gets free wins and its precision is inflated."""
    adapter = ArxivAdapter(
        SourceConfig(
            slug="arxiv-test",
            name="arXiv",
            endpoint=ENDPOINT,
            config={"categories": ["cs.AI", "cs.LG"], "max_results": 10},
        )
    )
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, text=SAMPLE))
    async with httpx.AsyncClient() as client:
        items = await adapter.fetch(client, since)
    assert len(items) == 1


@respx.mock
async def test_http_error_propagates_to_the_orchestrator(adapter, since):
    """Adapters do not swallow transport errors; the orchestrator owns that boundary."""
    respx.get(ENDPOINT).mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.fetch(client, since)


@respx.mock
async def test_unparseable_response_raises(adapter, since):
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, text="not xml at all"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="unparseable Atom"):
            await adapter.fetch(client, since)


@respx.mock
async def test_max_results_is_capped(since):
    adapter = ArxivAdapter(
        SourceConfig(
            slug="a", name="a", endpoint=ENDPOINT, config={"max_results": 100_000}
        )
    )
    route = respx.get(ENDPOINT).mock(return_value=httpx.Response(200, text=SAMPLE))
    async with httpx.AsyncClient() as client:
        await adapter.fetch(client, since)
    assert route.calls[0].request.url.params["max_results"] == "300"
