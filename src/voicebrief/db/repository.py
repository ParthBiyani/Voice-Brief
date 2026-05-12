"""Data access.

Kept deliberately thin — these are the few queries that are either hot, subtle, or
repeated. Everything else uses the session directly.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from voicebrief.db.models import Item, Source, SourceKind
from voicebrief.sources.base import RawItem, SourceConfig


def upsert_source(session: Session, spec: dict) -> Source:
    """Idempotent by slug.

    Runtime state (last_polled_at, failure counters) is intentionally never
    overwritten — re-seeding must not erase operational history.
    """
    stmt = (
        insert(Source)
        .values(
            slug=spec["slug"],
            name=spec["name"],
            kind=SourceKind(spec["kind"]),
            endpoint=spec["endpoint"],
            auth_ref=spec.get("auth_ref"),
            poll_interval_minutes=spec.get("poll_interval_minutes", 720),
            default_topics=spec.get("default_topics", []),
            trust_weight=spec.get("trust_weight", 0.5),
            enabled=spec.get("enabled", True),
            config=spec.get("config", {}),
        )
        .on_conflict_do_update(
            index_elements=[Source.slug],
            set_={
                "name": spec["name"],
                "kind": SourceKind(spec["kind"]),
                "endpoint": spec["endpoint"],
                "poll_interval_minutes": spec.get("poll_interval_minutes", 720),
                "default_topics": spec.get("default_topics", []),
                "trust_weight": spec.get("trust_weight", 0.5),
                "enabled": spec.get("enabled", True),
                "config": spec.get("config", {}),
                "updated_at": func.now(),
            },
        )
        .returning(Source)
    )
    # populate_existing: without it the identity map hands back the pre-update
    # object on a re-upsert, so callers silently observe stale values.
    return session.execute(
        stmt, execution_options={"populate_existing": True}
    ).scalar_one()


def due_sources(session: Session, *, now: datetime | None = None) -> Sequence[Source]:
    """Enabled sources whose poll interval has elapsed.

    A source that has failed repeatedly is backed off rather than disabled, so it
    recovers on its own once upstream is healthy again.
    """
    now = now or datetime.now(timezone.utc)
    rows = session.execute(select(Source).where(Source.enabled.is_(True))).scalars().all()

    due = []
    for source in rows:
        if source.last_polled_at is None:
            due.append(source)
            continue
        backoff = min(2**source.consecutive_failures, 16)
        interval = timedelta(minutes=source.poll_interval_minutes * backoff)
        if now - source.last_polled_at >= interval:
            due.append(source)
    return due


def to_source_config(source: Source) -> SourceConfig:
    return SourceConfig(
        slug=source.slug,
        name=source.name,
        endpoint=source.endpoint,
        config=source.config or {},
        default_topics=list(source.default_topics or []),
        trust_weight=source.trust_weight,
    )


def bulk_insert_items(
    session: Session, source_id: uuid.UUID, items: Iterable[RawItem]
) -> tuple[int, int]:
    """Insert items, skipping ones already seen from this source.

    Returns (inserted, skipped). Conflict handling is done by Postgres rather than a
    read-then-write, so concurrent ingest runs can't double-insert.
    """
    payload = [
        {
            "source_id": source_id,
            "external_id": item.external_id,
            "url": item.url,
            "title": item.title,
            "summary": item.summary,
            "body": item.body,
            "author": item.author,
            "published_at": item.published_at,
            "topics": item.topics,
            "engagement": item.engagement,
            "raw": item.raw,
        }
        for item in items
    ]
    if not payload:
        return 0, 0

    stmt = (
        insert(Item)
        .values(payload)
        .on_conflict_do_nothing(index_elements=[Item.source_id, Item.external_id])
        .returning(Item.id)
    )
    inserted = len(session.execute(stmt).scalars().all())
    return inserted, len(payload) - inserted


def recent_items(
    session: Session, *, since: datetime, limit: int, exclude_duplicates: bool = True
) -> Sequence[Item]:
    stmt = select(Item).where(Item.published_at >= since)
    if exclude_duplicates:
        stmt = stmt.where(Item.duplicate_of_id.is_(None))
    stmt = stmt.order_by(Item.published_at.desc()).limit(limit)
    return session.execute(stmt).scalars().all()


def items_needing_embedding(session: Session, *, limit: int = 500) -> Sequence[Item]:
    stmt = (
        select(Item)
        .where(Item.embedded.is_(False))
        .order_by(Item.published_at.desc())
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()
