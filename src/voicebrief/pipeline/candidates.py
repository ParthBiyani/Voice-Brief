"""Turn persisted clusters into ranking candidates.

Extracted so the CLI, the API and the eval harness all build candidates the same way
— a second, slightly different implementation is how the harness ends up measuring
something the product does not actually do.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import Cluster, Item, Source
from voicebrief.pipeline.ranking import RankCandidate
from voicebrief.pipeline.summarize import SourceRef

MAX_SOURCES_PER_CLUSTER = 4
MAX_BODY_CHARS = 1200


def build_candidates(session: Session, *, limit: int = 60) -> list[RankCandidate]:
    sources = {s.id: s for s in session.execute(select(Source)).scalars()}
    clusters = list(
        session.execute(
            select(Cluster).order_by(Cluster.size.desc(), Cluster.run_date.desc()).limit(limit)
        ).scalars()
    )

    candidates: list[RankCandidate] = []
    for cluster in clusters:
        items = list(
            session.execute(
                select(Item)
                .where(Item.cluster_id == cluster.id)
                .order_by(Item.engagement.desc())
                .limit(MAX_SOURCES_PER_CLUSTER)
            ).scalars()
        )
        if not items:
            continue

        head = items[0]
        source = sources.get(head.source_id)
        candidates.append(
            RankCandidate(
                cluster_id=cluster.id,
                title=head.title,
                summary=(head.summary or "")[:600],
                url=head.url,
                topics=list(cluster.topics or head.topics or []),
                engagement=float(head.engagement or 0.0),
                trust_weight=float(source.trust_weight) if source else 0.5,
                cluster_size=cluster.size,
                source_slug=source.slug if source else "",
                sources=[
                    SourceRef(
                        item_id=i.id,
                        title=i.title,
                        url=i.url,
                        source_slug=sources[i.source_id].slug if i.source_id in sources else "",
                    )
                    for i in items
                ],
                bodies={i.id: (i.summary or i.body or "")[:MAX_BODY_CHARS] for i in items},
            )
        )
    return candidates
