"""Ingestion orchestrator.

Two invariants, both from PRD §10:

1. **A failing source degrades, it never blocks.** Every adapter runs inside its own
   error boundary; a failure is written to `ingest_run`, the source's failure counter
   is bumped (which backs off its next poll), and the pass continues.
2. **The global cap is enforced here, not hoped for.** `VB_MAX_ITEMS_PER_RUN` is the
   budget guardrail that keeps LLM cost bounded three stages downstream.

Concurrency control is a Postgres advisory lock: overlapping cron hits are a no-op
rather than a duplicate crawl. That is the whole scheduler for v1.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from voicebrief.config import get_settings
from voicebrief.db import session_scope
from voicebrief.db.models import IngestRun, Source
from voicebrief.db.repository import bulk_insert_items, due_sources, to_source_config
from voicebrief.db.session import advisory_lock
from voicebrief.logging import get_logger
from voicebrief.sources import registry

log = get_logger(__name__)

INGEST_LOCK = "voicebrief:ingest"
PER_SOURCE_TIMEOUT = 90.0
MAX_CONCURRENT_SOURCES = 6


@dataclass(slots=True)
class SourceStats:
    slug: str
    fetched: int = 0
    inserted: int = 0
    skipped: int = 0
    ok: bool = True
    error: str | None = None


async def _fetch_one(
    client: httpx.AsyncClient,
    source: Source,
    since: datetime,
    semaphore: asyncio.Semaphore,
) -> tuple[SourceStats, list]:
    """Fetch a single source inside its own error boundary."""
    stats = SourceStats(slug=source.slug)
    bound = log.bind(source=source.slug, kind=source.kind.value)

    async with semaphore:
        try:
            adapter_kind = (source.config or {}).get("adapter", source.kind.value)
            adapter = registry.build(adapter_kind, to_source_config(source))
            items = await asyncio.wait_for(
                adapter.fetch(client, since), timeout=PER_SOURCE_TIMEOUT
            )
            stats.fetched = len(items)
            bound.info("source.fetched", count=len(items))
            return stats, list(items)

        except TimeoutError:
            stats.ok, stats.error = False, f"timeout after {PER_SOURCE_TIMEOUT}s"
        except httpx.HTTPError as exc:
            stats.ok, stats.error = False, f"{type(exc).__name__}: {exc}"
        except LookupError as exc:
            stats.ok, stats.error = False, str(exc)
        except Exception as exc:  # noqa: BLE001 — the boundary is the point
            stats.ok, stats.error = False, f"{type(exc).__name__}: {exc}"

        bound.warning("source.failed", error=stats.error)
        return stats, []


def _apply_topics(items: list, source: Source) -> list:
    """Fall back to the source's declared topics when an adapter can't infer any."""
    for item in items:
        if not item.topics:
            item.topics = list(source.default_topics or [])
    return items


async def run_ingest(
    *, only_slug: str | None = None, force: bool = False, lookback_hours: int = 36
) -> list[SourceStats]:
    """Run one ingestion pass across every due source."""
    settings = get_settings()
    started = datetime.now(timezone.utc)
    since = started - timedelta(hours=lookback_hours)

    with session_scope() as session:
        with advisory_lock(session, INGEST_LOCK) as acquired:
            if not acquired:
                log.warning("ingest.skipped", reason="another run holds the lock")
                return []

            sources = list(session.query(Source).all()) if force else list(due_sources(session))
            if only_slug:
                sources = [s for s in sources if s.slug == only_slug]
            if not sources:
                log.info("ingest.nothing_due")
                return []

            log.info("ingest.start", sources=len(sources), since=since.isoformat())

            semaphore = asyncio.Semaphore(MAX_CONCURRENT_SOURCES)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0), follow_redirects=True
            ) as client:
                results = await asyncio.gather(
                    *(_fetch_one(client, s, since, semaphore) for s in sources)
                )

            # Persist under the global cap. Higher-trust sources are drained first so
            # that when the cap bites it truncates the least valuable tail.
            budget = settings.max_items_per_run
            ordered = sorted(
                zip(sources, results, strict=True),
                key=lambda pair: pair[0].trust_weight,
                reverse=True,
            )

            all_stats: list[SourceStats] = []
            for source, (stats, items) in ordered:
                if items and budget > 0:
                    items = _apply_topics(items[:budget], source)
                    stats.inserted, stats.skipped = bulk_insert_items(session, source.id, items)
                    budget -= stats.inserted
                elif items:
                    stats.skipped += len(items)
                    log.warning("ingest.cap_reached", source=source.slug, dropped=len(items))

                source.last_polled_at = started
                source.last_status = "ok" if stats.ok else "error"
                source.consecutive_failures = 0 if stats.ok else source.consecutive_failures + 1

                session.add(
                    IngestRun(
                        source_id=source.id,
                        started_at=started,
                        finished_at=datetime.now(timezone.utc),
                        fetched=stats.fetched,
                        inserted=stats.inserted,
                        skipped=stats.skipped,
                        ok=stats.ok,
                        error=stats.error,
                    )
                )
                all_stats.append(stats)

            log.info(
                "ingest.done",
                inserted=sum(s.inserted for s in all_stats),
                failed=sum(1 for s in all_stats if not s.ok),
                budget_left=budget,
            )
            return all_stats
