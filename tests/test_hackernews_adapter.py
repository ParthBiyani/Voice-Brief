from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from voicebrief.sources.base import SourceConfig
from voicebrief.sources.hackernews import HackerNewsAdapter

BASE = "https://hacker-news.firebaseio.com/v0"


@pytest.fixture
def adapter() -> HackerNewsAdapter:
    return HackerNewsAdapter(
        SourceConfig(
            slug="hn",
            name="HN",
            endpoint=BASE,
            default_topics=["engineering"],
            config={"feed": "topstories", "limit": 10, "min_score": 40},
        )
    )


def story(sid: int, **over) -> dict:
    base = {
        "id": sid,
        "type": "story",
        "title": f"Story {sid}",
        "by": "someone",
        "score": 100,
        "descendants": 42,
        "time": int(datetime.now(timezone.utc).timestamp()),
        "url": f"https://example.com/{sid}",
    }
    return {**base, **over}


def mock_hn(stories: list[dict]) -> None:
    respx.get(f"{BASE}/topstories.json").mock(
        return_value=httpx.Response(200, json=[s["id"] for s in stories])
    )
    for s in stories:
        respx.get(f"{BASE}/item/{s['id']}.json").mock(return_value=httpx.Response(200, json=s))


async def run(adapter, since=None):
    since = since or datetime.now(timezone.utc) - timedelta(days=2)
    async with httpx.AsyncClient() as client:
        return await adapter.fetch(client, since)


@respx.mock
async def test_maps_story_fields(adapter):
    mock_hn([story(1)])
    items = await run(adapter)
    assert len(items) == 1
    item = items[0]
    assert item.external_id == "1"
    assert item.url == "https://example.com/1"
    assert item.raw["comments"] == 42
    assert item.raw["discussion_url"] == "https://news.ycombinator.com/item?id=1"


@respx.mock
async def test_score_below_threshold_is_dropped(adapter):
    mock_hn([story(1, score=10), story(2, score=90)])
    items = await run(adapter)
    assert [i.external_id for i in items] == ["2"]


@respx.mock
async def test_engagement_is_normalized_and_saturates(adapter):
    mock_hn([story(1, score=250), story(2, score=5000)])
    items = {i.external_id: i.engagement for i in await run(adapter)}
    assert items["1"] == pytest.approx(0.5)
    assert items["2"] == 1.0, "engagement must stay comparable across sources"


@respx.mock
async def test_ask_hn_without_url_points_at_the_thread(adapter):
    s = story(1)
    del s["url"]
    mock_hn([s])
    items = await run(adapter)
    assert items[0].url == "https://news.ycombinator.com/item?id=1"


@respx.mock
@pytest.mark.parametrize("bad", [{"dead": True}, {"deleted": True}, {"type": "job"}])
async def test_dead_deleted_and_non_stories_are_skipped(adapter, bad):
    mock_hn([story(1, **bad), story(2)])
    items = await run(adapter)
    assert [i.external_id for i in items] == ["2"]


@respx.mock
async def test_old_story_outside_window_is_dropped(adapter):
    old = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
    mock_hn([story(1, time=old), story(2)])
    items = await run(adapter)
    assert [i.external_id for i in items] == ["2"]


@respx.mock
async def test_one_failing_item_does_not_lose_the_rest(adapter):
    """A single 500 on one story id must not cost us the whole front page."""
    respx.get(f"{BASE}/topstories.json").mock(return_value=httpx.Response(200, json=[1, 2]))
    respx.get(f"{BASE}/item/1.json").mock(return_value=httpx.Response(500))
    respx.get(f"{BASE}/item/2.json").mock(return_value=httpx.Response(200, json=story(2)))
    items = await run(adapter)
    assert [i.external_id for i in items] == ["2"]
