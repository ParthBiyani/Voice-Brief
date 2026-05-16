"""Enrichment stage: embed → dedup → cluster.

Runs after ingest and before ranking. Everything here is local computation — no LLM
calls — which is why it can afford to touch all ~300 surviving items rather than the
~40 the ranker sees.

Persists three things:
  * item vectors in Qdrant, for dedup now and episode search later
  * `duplicate_of_id` back-references on collapsed items
  * `cluster` rows with membership, so the ranker scores stories rather than items
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db.models import Cluster as ClusterRow
from voicebrief.db.models import Item, Source
from voicebrief.logging import get_logger
from voicebrief.pipeline.clustering import cluster_items
from voicebrief.pipeline.dedup import DedupCandidate, find_duplicates
from voicebrief.pipeline.embedding import get_embedding_service, item_text
from voicebrief.pipeline.filtering import prefilter
from voicebrief.pipeline.vectorstore import get_vector_store

log = get_logger(__name__)


@dataclass(slots=True)
class EnrichResult:
    considered: int
    filtered: int
    embedded: int
    duplicate_groups: int
    collapsed: int
    clusters: int
    multi_item_clusters: int


def enrich(
    session: Session,
    *,
    user_topics: set[str] | None = None,
    lookback_hours: int = 36,
    keep: int | None = None,
    dedup_threshold: float = 0.92,
    now: datetime | None = None,
) -> EnrichResult:
    settings = get_settings()
    now = now or datetime.now(timezone.utc)
    keep = keep or 300
    since = now - timedelta(hours=lookback_hours)

    sources = list(session.execute(select(Source)).scalars())
    trust_by_source = {s.id: s.trust_weight for s in sources}

    items = list(
        session.execute(
            select(Item)
            .where(Item.published_at >= since)
            .order_by(Item.published_at.desc())
            .limit(settings.max_items_per_run)
        ).scalars()
    )
    if not items:
        log.warning("enrich.no_items", since=since.isoformat())
        return EnrichResult(0, 0, 0, 0, 0, 0, 0)

    # 1 — cheap filter, no vectors involved
    survivors = prefilter(
        items, trust_by_source=trust_by_source, user_topics=user_topics, keep=keep, now=now
    )
    kept_items = [s.item for s in survivors]

    # 2 — embed
    service = get_embedding_service()
    texts = [item_text(i.title, i.summary) for i in kept_items]
    vectors = service.encode(texts)

    store = get_vector_store()
    store.ensure_collections()
    slug_by_source = {s.id: s.slug for s in sources}
    store.upsert_items(
        [i.id for i in kept_items],
        vectors,
        [
            {
                "title": i.title,
                "url": i.url,
                "source_slug": slug_by_source.get(i.source_id, "unknown"),
                "published_at": i.published_at.timestamp(),
                "topics": list(i.topics or []),
            }
            for i in kept_items
        ],
    )
    for item in kept_items:
        item.embedded = True

    # 3 — dedup
    candidates = [
        DedupCandidate(
            id=i.id,
            title=i.title,
            url=i.url,
            trust_weight=trust_by_source.get(i.source_id, 0.5),
            engagement=float(i.engagement or 0.0),
            published_ts=i.published_at.timestamp(),
        )
        for i in kept_items
    ]
    groups = find_duplicates(candidates, vectors, threshold=dedup_threshold)

    duplicate_ids: set[uuid.UUID] = set()
    by_id = {i.id: i for i in kept_items}
    for group in groups:
        for member_id in group.member_ids:
            if member := by_id.get(member_id):
                member.duplicate_of_id = group.canonical_id
                duplicate_ids.add(member_id)

    # 4 — cluster the survivors only. Clustering duplicates would inflate cluster
    # sizes with copies of one story and skew the ranker toward well-syndicated news.
    unique_items = [i for i in kept_items if i.id not in duplicate_ids]
    index_by_id = {item.id: idx for idx, item in enumerate(kept_items)}
    unique_vectors = np.stack([vectors[index_by_id[i.id]] for i in unique_items])

    clusters = cluster_items(
        [i.id for i in unique_items],
        unique_vectors,
        topics_by_id={i.id: list(i.topics or []) for i in unique_items},
    )

    # 5 — persist clusters
    session.query(ClusterRow).filter(ClusterRow.run_date >= now.date()).delete(
        synchronize_session=False
    )
    for cluster in clusters:
        row = ClusterRow(
            run_date=now,
            topics=cluster.topics,
            size=cluster.size,
            centroid_item_id=cluster.centroid_id,
        )
        session.add(row)
        session.flush()
        for member_id in cluster.member_ids:
            if member := by_id.get(member_id):
                member.cluster_id = row.id

    result = EnrichResult(
        considered=len(items),
        filtered=len(kept_items),
        embedded=len(kept_items),
        duplicate_groups=len(groups),
        collapsed=len(duplicate_ids),
        clusters=len(clusters),
        multi_item_clusters=sum(1 for c in clusters if c.size > 1),
    )
    log.info("enrich.done", **asdict(result))
    return result
