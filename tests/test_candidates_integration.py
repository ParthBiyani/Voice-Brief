"""build_candidates / enrich against the docker-compose Postgres.

Every test runs inside a transaction that is rolled back, so the shared development
database is never polluted and tests cannot see each other's rows.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import (
    Cluster,
    Episode,
    EpisodeMode,
    EpisodeStatus,
    Item,
    Segment,
    Source,
    SourceKind,
    User,
)
from voicebrief.db.session import get_engine
from voicebrief.pipeline.candidates import CandidateReport, build_candidates
from voicebrief.pipeline.enrich import _drop_orphan_clusters

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


@pytest.fixture
def session():
    with Session(get_engine()) as s:
        s.begin()
        try:
            yield s
        finally:
            s.rollback()


@pytest.fixture
def source(session) -> Source:
    src = Source(
        slug=f"t-{uuid.uuid4().hex[:8]}",
        name="Test",
        kind=SourceKind.rss,
        endpoint="https://example.com/feed",
        trust_weight=0.7,
    )
    session.add(src)
    session.flush()
    return src


@pytest.fixture
def user(session) -> User:
    u = User(email=f"{uuid.uuid4().hex[:8]}@example.com")
    session.add(u)
    session.flush()
    return u


def make_story(session, source, *, title, age_hours, url=None, extra_urls=(), size=None) -> Cluster:
    """A cluster of one or more items, all first seen `age_hours` ago."""
    urls = [url or f"https://example.com/{uuid.uuid4().hex}", *extra_urls]
    seen = NOW - timedelta(hours=age_hours)
    cluster = Cluster(run_date=seen, topics=["agentic-ai"], size=size or len(urls))
    session.add(cluster)
    session.flush()
    for i, u in enumerate(urls):
        session.add(
            Item(
                source_id=source.id,
                external_id=uuid.uuid4().hex,
                url=u,
                title=title if i == 0 else f"{title} (again)",
                published_at=seen,
                created_at=seen,
                cluster_id=cluster.id,
                engagement=0.5,
            )
        )
    session.flush()
    return cluster


def air(session, user, cluster_urls: list[str], *, cluster: Cluster | None = None) -> Episode:
    episode = Episode(
        user_id=user.id,
        mode=EpisodeMode.brief,
        status=EpisodeStatus.ready,
        created_at=NOW - timedelta(hours=20),
    )
    session.add(episode)
    session.flush()
    session.add(
        Segment(
            episode_id=episode.id,
            position=1,
            kind="story",
            script="x",
            cluster_id=cluster.id if cluster else None,
            citations=[{"url": u, "title": "t"} for u in cluster_urls],
        )
    )
    session.flush()
    return episode


class TestFreshness:
    def test_only_recently_seen_stories_are_candidates(self, session, source, user):
        fresh = make_story(session, source, title="Fresh story today", age_hours=3)
        make_story(session, source, title="Stale story last week", age_hours=24 * 7, size=9)
        got = build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        assert {c.cluster_id for c in got} >= {fresh.id}
        assert all("Stale" not in c.title for c in got)

    def test_a_huge_old_cluster_no_longer_crowds_out_todays_news(self, session, source, user):
        make_story(session, source, title="Old blockbuster story", age_hours=24 * 10, size=50)
        today = make_story(session, source, title="Small release today", age_hours=2)
        ids = [
            c.cluster_id for c in build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        ]
        assert today.id in ids

    def test_candidates_carry_freshness_for_the_recency_term(self, session, source, user):
        make_story(session, source, title="Release notes today", age_hours=4)
        got = build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        assert all(c.freshness is not None for c in got)


class TestNeverRepeat:
    def test_a_story_already_aired_is_excluded(self, session, source, user):
        url = f"https://example.com/{uuid.uuid4().hex}"
        story = make_story(session, source, title="Big launch yesterday", age_hours=10, url=url)
        air(session, user, [url], cluster=story)
        report = CandidateReport(window_hours=0)
        got = build_candidates(session, user_id=user.id, now=NOW, report=report, limit=100_000)
        assert story.id not in {c.cluster_id for c in got}
        assert report.excluded_aired >= 1

    def test_exclusion_survives_the_cluster_being_rebuilt(self, session, source, user):
        # A later enrich run re-clusters the same item under a brand-new cluster id
        # (and the old segment link is nulled). The URL match must still hold.
        url = f"https://example.com/{uuid.uuid4().hex}"
        old = make_story(session, source, title="Launch covered earlier", age_hours=10, url=url)
        air(session, user, [url], cluster=None)
        rebuilt = Cluster(run_date=NOW, topics=[], size=1)
        session.add(rebuilt)
        session.flush()
        for item in session.execute(select(Item).where(Item.cluster_id == old.id)).scalars():
            item.cluster_id = rebuilt.id
        session.flush()
        got = build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        assert rebuilt.id not in {c.cluster_id for c in got}

    def test_another_users_episode_does_not_hide_a_story(self, session, source, user):
        other = User(email=f"{uuid.uuid4().hex[:8]}@example.com")
        session.add(other)
        session.flush()
        url = f"https://example.com/{uuid.uuid4().hex}"
        story = make_story(session, source, title="Shared story", age_hours=5, url=url)
        air(session, other, [url])
        got = build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        assert story.id in {c.cluster_id for c in got}

    def test_a_developing_story_with_new_coverage_is_kept(self, session, source, user):
        old_url = f"https://example.com/{uuid.uuid4().hex}"
        story = make_story(
            session,
            source,
            title="Model release",
            age_hours=5,
            url=old_url,
            extra_urls=[f"https://b.com/{uuid.uuid4().hex}", f"https://c.com/{uuid.uuid4().hex}"],
        )
        air(session, user, [old_url])
        got = build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        assert story.id in {c.cluster_id for c in got}


class TestWindowWidensAfterAGap:
    def test_stories_from_a_skipped_weekend_are_picked_up(self, session, source, user):
        # Last episode three days ago: yesterday-and-before news is still unheard.
        episode = Episode(
            user_id=user.id,
            mode=EpisodeMode.brief,
            status=EpisodeStatus.ready,
            created_at=NOW - timedelta(days=3),
        )
        session.add(episode)
        session.flush()
        story = make_story(session, source, title="Saturday release", age_hours=50)
        got = build_candidates(session, user_id=user.id, now=NOW, limit=100_000)
        assert story.id in {c.cluster_id for c in got}


class TestOrphanClusterCleanup:
    def test_empty_unreferenced_clusters_are_removed_but_aired_ones_kept(
        self, session, source, user
    ):
        empty = Cluster(run_date=NOW, topics=[], size=0)
        aired_empty = Cluster(run_date=NOW, topics=[], size=0)
        live = make_story(session, source, title="Still has items", age_hours=1)
        session.add_all([empty, aired_empty])
        session.flush()
        air(session, user, ["https://example.com/x"], cluster=aired_empty)

        _drop_orphan_clusters(session)
        session.flush()

        remaining = {c.id for c in session.execute(select(Cluster)).scalars()}
        assert empty.id not in remaining
        assert aired_empty.id in remaining
        assert live.id in remaining
