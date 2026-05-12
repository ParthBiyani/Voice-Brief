"""Source adapter contract.

Every upstream — an Atom feed, a Firebase tree, a REST search — is reduced to the
same `RawItem` here. Downstream stages (dedup, clustering, ranking, scripting) are
written against that one shape and never learn where an item came from.

Adapters register themselves by `kind`; the DB row supplies endpoint and config, so
a new feed is an INSERT rather than a deploy (PRD §3).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, Field, HttpUrl, field_validator

from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)


class RawItem(BaseModel):
    """Normalized content unit, pre-persistence."""

    external_id: str
    url: str
    title: str
    summary: str | None = None
    body: str | None = None
    author: str | None = None
    published_at: datetime
    topics: list[str] = Field(default_factory=list)

    # 0..1, comparable across sources. Each adapter is responsible for squashing its
    # own native scale (HN points, GitHub stars, HF downloads) into this range.
    engagement: float = 0.0
    raw: dict = Field(default_factory=dict)

    @field_validator("published_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v

    @field_validator("title", "summary", "body")
    @classmethod
    def _collapse_whitespace(cls, v: str | None) -> str | None:
        return " ".join(v.split()) if v else v

    @field_validator("engagement")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def fingerprint(self) -> str:
        """Content hash used for cheap exact-duplicate detection across sources."""
        basis = f"{self.title.lower().strip()}|{self.url.split('?')[0].rstrip('/')}"
        return hashlib.sha256(basis.encode()).hexdigest()[:32]


class SourceConfig(BaseModel):
    """The subset of a `source` row an adapter is allowed to see."""

    slug: str
    name: str
    endpoint: str
    config: dict = Field(default_factory=dict)
    default_topics: list[str] = Field(default_factory=list)
    trust_weight: float = 0.5


class SourceAdapter(ABC):
    """One upstream integration.

    Adapters must not raise on upstream failure paths they can anticipate; the
    orchestrator records a failed `IngestRun` and the episode proceeds without them
    (PRD §10 — a dead source must never block the brief).
    """

    kind: str

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.settings = get_settings()

    @abstractmethod
    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        """Return items published after `since`. Ordering is not guaranteed."""

    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.settings.user_agent, "Accept": "application/json"}

    @property
    def log(self):
        return log.bind(source=self.config.slug, kind=self.kind)


class _Registry:
    """Maps `source.kind` to an adapter class."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[SourceAdapter]] = {}

    def register(self, cls: type[SourceAdapter]) -> type[SourceAdapter]:
        if not getattr(cls, "kind", None):
            raise ValueError(f"{cls.__name__} must define a `kind`")
        self._adapters[cls.kind] = cls
        return cls

    def build(self, kind: str, config: SourceConfig) -> SourceAdapter:
        try:
            return self._adapters[kind](config)
        except KeyError:
            raise LookupError(
                f"No adapter registered for kind={kind!r}. Known: {sorted(self._adapters)}"
            ) from None

    def known_kinds(self) -> list[str]:
        return sorted(self._adapters)


registry = _Registry()
