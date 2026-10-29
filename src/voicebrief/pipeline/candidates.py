"""Turn persisted clusters into ranking candidates.

Extracted so the CLI, the API and the eval harness all build candidates the same way
— a second, slightly different implementation is how the harness ends up measuring
something the product does not actually do.

What counts as a candidate is the heart of "a brief you can listen to every day":

  * **Fresh** — the story has at least one item first seen inside the window. A big
    cluster from last Tuesday is not today's news however many outlets covered it.
  * **Not already aired** — a story the listener heard in a recent episode is skipped,
    matched by source URL so it still holds when a later enrich run re-clusters the
    same items under a new cluster id.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import Cluster, Episode, EpisodeStatus, Item, Segment, Source
from voicebrief.logging import get_logger
from voicebrief.pipeline.filtering import item_freshness
from voicebrief.pipeline.ranking import RankCandidate
from voicebrief.pipeline.summarize import SourceRef

log = get_logger(__name__)

MAX_SOURCES_PER_CLUSTER = 4
MAX_BODY_CHARS = 1200

# The default look-back, and the most a long gap between episodes may widen it to.
DEFAULT_FRESH_HOURS = 36
MAX_FRESH_HOURS = 24 * 7
# Margin so a story first seen just before the previous episode still gets a hearing if
# that episode skipped it.
GAP_MARGIN_HOURS = 6
AIRED_LOOKBACK_DAYS = 30
# A cluster is "aired" once at least this share of its items were already cited. One
# aired item inside a bigger, newer cluster is a developing story, not a repeat.
AIRED_SHARE = 0.5


@dataclass(slots=True)
class CandidateReport:
    """What was considered and what was set aside — kept for the history log."""

    window_hours: float
    fresh_clusters: int = 0
    excluded_aired: int = 0
    candidates: int = 0


def normalize_url(url: str) -> str:
    """Compare URLs without caring about scheme, `www.`, fragments or a trailing slash."""
    value = (url or "").strip().lower().split("#", 1)[0]
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    if value.startswith("www."):
        value = value[4:]
    return value.rstrip("/")


def freshness_window_hours(last_episode_at: datetime | None, *, now: datetime) -> float:
    """Look back far enough to cover the gap since the last episode, never less than
    the default and never more than a week. Skip a few days and the next brief should
    still catch up on what you missed; the aired filter stops anything being repeated."""
    window = float(DEFAULT_FRESH_HOURS)
    if last_episode_at is not None:
        if last_episode_at.tzinfo is None:
            last_episode_at = last_episode_at.replace(tzinfo=UTC)
        gap = (now - last_episode_at).total_seconds() / 3600.0 + GAP_MARGIN_HOURS
        window = max(window, gap)
    return min(window, float(MAX_FRESH_HOURS))


def is_aired(item_urls: list[str], aired: set[str]) -> bool:
    if not item_urls or not aired:
        return False
    hits = sum(1 for url in item_urls if normalize_url(url) in aired)
    return hits / len(item_urls) >= AIRED_SHARE


def _recent_episodes(session: Session, user_id: uuid.UUID | None, since: datetime):
    query = select(Episode).where(
        Episode.status == EpisodeStatus.ready, Episode.created_at >= since
    )
    if user_id is not None:
        query = query.where(Episode.user_id == user_id)
    return list(session.execute(query.order_by(Episode.created_at.desc())).scalars())


def aired_urls(session: Session, episodes: list[Episode]) -> set[str]:
    """Every source URL cited by the given episodes' story segments."""
    if not episodes:
        return set()
    rows = session.execute(
        select(Segment.citations).where(
            Segment.episode_id.in_([e.id for e in episodes]), Segment.kind == "story"
        )
    ).scalars()
    urls: set[str] = set()
    for citations in rows:
        for citation in citations or []:
            if isinstance(citation, dict) and citation.get("url"):
                urls.add(normalize_url(citation["url"]))
    return urls


def build_candidates(
    session: Session,
    *,
    limit: int = 200,
    user_id: uuid.UUID | None = None,
    now: datetime | None = None,
    report: CandidateReport | None = None,
) -> list[RankCandidate]:
    """Fresh, not-yet-aired stories as ranking candidates.

    `limit` bounds how many clusters are loaded, newest and largest first. It is
    generous on purpose: the heuristic pre-rank is free, and only the top slice of its
    output ever reaches a paid model call.
    """
    now = now or datetime.now(UTC)

    last = _recent_episodes(session, user_id, now - timedelta(days=AIRED_LOOKBACK_DAYS))
    window = freshness_window_hours(last[0].created_at if last else None, now=now)
    since = now - timedelta(hours=window)
    aired = aired_urls(session, last)

    fresh_ids = (
        select(Item.cluster_id)
        .where(
            Item.cluster_id.is_not(None),
            Item.duplicate_of_id.is_(None),
            Item.created_at >= since,
        )
        .distinct()
    )
    clusters = list(
        session.execute(
            select(Cluster)
            .where(Cluster.id.in_(fresh_ids))
            .order_by(Cluster.size.desc(), Cluster.run_date.desc())
            .limit(limit)
        ).scalars()
    )

    items_by_cluster: dict[uuid.UUID, list[Item]] = defaultdict(list)
    if clusters:
        rows = session.execute(
            select(Item)
            .where(Item.cluster_id.in_([c.id for c in clusters]), Item.duplicate_of_id.is_(None))
            .order_by(Item.engagement.desc())
        ).scalars()
        for item in rows:
            items_by_cluster[item.cluster_id].append(item)

    sources = {s.id: s for s in session.execute(select(Source)).scalars()}

    stats = report or CandidateReport(window_hours=window)
    stats.window_hours = window
    stats.fresh_clusters = len(clusters)

    candidates: list[RankCandidate] = []
    for cluster in clusters:
        members = items_by_cluster.get(cluster.id, [])
        if not members:
            continue
        urls = [m.url for m in members]
        if is_aired(urls, aired):
            stats.excluded_aired += 1
            continue

        items = members[:MAX_SOURCES_PER_CLUSTER]
        head = items[0]
        source = sources.get(head.source_id)
        newest = max(item_freshness(m.published_at, m.created_at) for m in members)
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
                freshness=newest,
                item_urls=urls,
            )
        )

    stats.candidates = len(candidates)
    log.info(
        "candidates.built",
        window_hours=round(window, 1),
        fresh_clusters=stats.fresh_clusters,
        excluded_aired=stats.excluded_aired,
        candidates=stats.candidates,
    )
    return candidates
