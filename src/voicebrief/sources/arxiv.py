"""arXiv adapter — official Atom API.

arXiv asks for a 3-second gap between requests and no parallel hammering. We honour
that with an explicit delay between category pages rather than relying on politeness
by accident; the whole point of the source strategy (PRD §3) is that every integration
is one we could describe out loud to the people running it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone

import feedparser
import httpx

from voicebrief.sources.base import RawItem, SourceAdapter, registry

# arXiv's stated rate limit. Do not lower this.
_REQUEST_GAP_SECONDS = 3.0
_MAX_RESULTS_CAP = 300


@registry.register
class ArxivAdapter(SourceAdapter):
    kind = "arxiv"

    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        categories: list[str] = self.config.config.get("categories", ["cs.AI"])
        max_results = min(int(self.config.config.get("max_results", 100)), _MAX_RESULTS_CAP)

        items: list[RawItem] = []
        for index, category in enumerate(categories):
            if index:
                await asyncio.sleep(_REQUEST_GAP_SECONDS)
            items.extend(await self._fetch_category(client, category, max_results, since))

        # The same paper is routinely cross-listed (cs.CL and cs.LG). Collapse on the
        # arXiv id here so the semantic dedup stage isn't handed free wins that would
        # inflate its measured precision.
        unique: dict[str, RawItem] = {}
        for item in items:
            unique.setdefault(item.external_id, item)
        return list(unique.values())

    async def _fetch_category(
        self, client: httpx.AsyncClient, category: str, max_results: int, since: datetime
    ) -> list[RawItem]:
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": "0",
            "max_results": str(max_results),
        }
        response = await client.get(
            self.config.endpoint, params=params, headers={"User-Agent": self.settings.user_agent}
        )
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        if feed.bozo and not feed.entries:
            raise ValueError(f"arXiv returned unparseable Atom for {category}")

        out: list[RawItem] = []
        for entry in feed.entries:
            published = self._parse_date(entry)
            if published is None or published < since:
                continue

            arxiv_id = entry.get("id", "").rsplit("/", 1)[-1]
            if not arxiv_id:
                continue

            authors = [a.get("name", "") for a in entry.get("authors", [])]
            entry_cats = [t.get("term") for t in entry.get("tags", []) if t.get("term")]

            out.append(
                RawItem(
                    external_id=arxiv_id,
                    url=entry.get("link") or f"https://arxiv.org/abs/{arxiv_id}",
                    title=entry.get("title", "").strip(),
                    summary=entry.get("summary", "").strip() or None,
                    author=", ".join(authors[:4]) or None,
                    published_at=published,
                    topics=self._topics_for(entry_cats),
                    # arXiv exposes no popularity signal at submission time. A flat
                    # prior is honest; ranking leans on the stack profile instead.
                    engagement=0.0,
                    raw={
                        "arxiv_id": arxiv_id,
                        "categories": entry_cats,
                        "primary_category": entry.get("arxiv_primary_category", {}).get("term"),
                        "authors": authors,
                        "comment": entry.get("arxiv_comment"),
                    },
                )
            )
        return out

    @staticmethod
    def _parse_date(entry) -> datetime | None:
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if not parsed:
            return None
        return datetime(*parsed[:6], tzinfo=timezone.utc)

    def _topics_for(self, categories: list[str]) -> list[str]:
        mapping = {
            "cs.AI": "agentic-ai",
            "cs.CL": "nlp",
            "cs.LG": "machine-learning",
            "cs.CV": "computer-vision",
            "cs.SE": "software-engineering",
            "cs.DC": "systems",
            "cs.IR": "retrieval",
        }
        topics = {mapping[c] for c in categories if c in mapping}
        topics.add("research")
        return sorted(topics)
