"""Hugging Face Hub adapter — public JSON API.

Serves both models and datasets; they share a response shape closely enough that one
adapter with a `kind` switch is honest rather than lazy.

The Hub exposes `downloads` and `likes`, which are far better engagement signals than
anything arXiv gives us, but they are heavily skewed — a handful of foundation models
have millions of downloads while a good new model has thousands. A log scale keeps the
long tail visible instead of flattening it to zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx

from voicebrief.sources.base import RawItem, SourceAdapter, registry

# Downloads at which engagement reaches 1.0 on a log scale.
_DOWNLOAD_SATURATION = 1_000_000.0
_MAX_LIMIT = 200


@registry.register
class HuggingFaceAdapter(SourceAdapter):
    kind = "huggingface"

    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        cfg = self.config.config
        entity = cfg.get("kind", "models")
        limit = min(int(cfg.get("limit", 50)), _MAX_LIMIT)

        response = await client.get(
            self.config.endpoint,
            params={
                "sort": cfg.get("sort", "trendingScore"),
                "direction": "-1",
                "limit": str(limit),
                "full": "true",
            },
            headers=self.headers(),
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Hugging Face returned {type(payload).__name__}, expected a list")

        items: list[RawItem] = []
        for record in payload:
            item = self._to_item(record, entity, since)
            if item:
                items.append(item)
        return items

    def _to_item(self, record: dict, entity: str, since: datetime) -> RawItem | None:
        repo_id = record.get("id") or record.get("modelId")
        if not repo_id or record.get("private"):
            return None

        # `lastModified` is the only reliably present timestamp; `createdAt` is often
        # missing on older repos. Recency here means "changed recently", which is the
        # right semantics for a trending feed anyway.
        published = self._parse_ts(record.get("lastModified") or record.get("createdAt"))
        if published is None or published < since:
            return None

        downloads = int(record.get("downloads") or 0)
        likes = int(record.get("likes") or 0)
        path = "datasets/" if entity == "datasets" else ""

        return RawItem(
            external_id=f"{entity}:{repo_id}",
            url=f"https://huggingface.co/{path}{repo_id}",
            title=f"{repo_id} ({'dataset' if entity == 'datasets' else 'model'})",
            summary=self._describe(record, downloads, likes),
            author=record.get("author") or repo_id.split("/")[0],
            published_at=published,
            topics=self._topics(record),
            engagement=self._engagement(downloads, likes),
            raw={
                "repo_id": repo_id,
                "entity": entity,
                "downloads": downloads,
                "likes": likes,
                "pipeline_tag": record.get("pipeline_tag"),
                "library": record.get("library_name"),
                "tags": record.get("tags", []),
            },
        )

    @staticmethod
    def _parse_ts(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _describe(record: dict, downloads: int, likes: int) -> str:
        parts = []
        if task := record.get("pipeline_tag"):
            parts.append(f"Task: {task}.")
        if library := record.get("library_name"):
            parts.append(f"Library: {library}.")
        parts.append(f"{downloads:,} downloads, {likes:,} likes.")
        return " ".join(parts)

    def _topics(self, record: dict) -> list[str]:
        topics = set(self.config.default_topics)
        # Hub tags are a mix of licences, languages, arxiv links and real task tags.
        # Only the task-shaped ones are useful downstream.
        noise_prefixes = ("license:", "arxiv:", "dataset:", "base_model:", "region:", "doi:")
        for tag in record.get("tags", []):
            if isinstance(tag, str) and not tag.startswith(noise_prefixes) and len(tag) < 40:
                topics.add(tag.lower())
        if task := record.get("pipeline_tag"):
            topics.add(task.lower())
        return sorted(topics)[:12]

    @staticmethod
    def _engagement(downloads: int, likes: int) -> float:
        """Log-scaled so a 5k-download newcomer is distinguishable from a dead repo.

        A linear scale against a million-download saturation point would round almost
        every genuinely new model to zero, which is exactly the item we most want the
        ranker to be able to see.
        """
        if downloads <= 0 and likes <= 0:
            return 0.0
        download_score = math.log1p(downloads) / math.log1p(_DOWNLOAD_SATURATION)
        like_score = min(likes / 1000.0, 1.0)
        return min(0.7 * download_score + 0.3 * like_score, 1.0)
