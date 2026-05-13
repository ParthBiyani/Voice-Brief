"""Hacker News adapter — official Firebase API.

HN's API is one request per item, so a 120-story front page is 120 round trips. That
is fine at our cadence but only because it is bounded and concurrency-limited here;
the comment count and score are worth the calls because they are the only real
engagement signal in the Tier 1 set.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx

from voicebrief.sources.base import RawItem, SourceAdapter, registry

_MAX_CONCURRENT = 10
_SCORE_SATURATION = 500.0  # points at which engagement is treated as 1.0


@registry.register
class HackerNewsAdapter(SourceAdapter):
    kind = "hackernews"

    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        feed = self.config.config.get("feed", "topstories")
        limit = int(self.config.config.get("limit", 100))
        min_score = int(self.config.config.get("min_score", 0))

        response = await client.get(f"{self.config.endpoint}/{feed}.json", headers=self.headers())
        response.raise_for_status()
        story_ids = (response.json() or [])[:limit]

        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
        stories = await asyncio.gather(
            *(self._fetch_story(client, sid, semaphore) for sid in story_ids),
            return_exceptions=True,
        )

        items: list[RawItem] = []
        for story in stories:
            # A single dead item id must not cost us the other 119.
            if isinstance(story, BaseException) or not story:
                continue
            item = self._to_item(story, since, min_score)
            if item:
                items.append(item)
        return items

    async def _fetch_story(
        self, client: httpx.AsyncClient, story_id: int, semaphore: asyncio.Semaphore
    ) -> dict | None:
        async with semaphore:
            response = await client.get(
                f"{self.config.endpoint}/item/{story_id}.json", headers=self.headers()
            )
            response.raise_for_status()
            return response.json()

    def _to_item(self, story: dict, since: datetime, min_score: int) -> RawItem | None:
        if story.get("type") != "story" or story.get("dead") or story.get("deleted"):
            return None

        score = int(story.get("score", 0))
        if score < min_score:
            return None

        published = datetime.fromtimestamp(int(story.get("time", 0)), tz=timezone.utc)
        if published < since:
            return None

        story_id = story["id"]
        hn_url = f"https://news.ycombinator.com/item?id={story_id}"
        # Ask HNs (no external url) point at the thread itself.
        url = story.get("url") or hn_url

        return RawItem(
            external_id=str(story_id),
            url=url,
            title=story.get("title", ""),
            summary=story.get("text"),
            author=story.get("by"),
            published_at=published,
            topics=self.config.default_topics,
            engagement=min(score / _SCORE_SATURATION, 1.0),
            raw={
                "score": score,
                "comments": int(story.get("descendants", 0)),
                "discussion_url": hn_url,
                "hn_id": story_id,
            },
        )
