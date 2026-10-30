"""What the listener's thumbs say about what they want to hear.

A thumb is on one segment. Its lesson is wider: a story from a source or on a topic that
keeps getting thumbs-down should rank lower tomorrow, and one that keeps getting thumbs-up
should rank higher. This reduces the votes to a per-source and a per-topic affinity, and
keeps the most recent liked and disliked headlines as examples for the re-rank prompt.

Affinities are smoothed, so one stray thumb moves a source a little and a pattern moves it
a lot, and they stay in (-1, 1) so the signal can never outweigh evidence about the story.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import Feedback, Item, Segment, Source
from voicebrief.logging import get_logger

log = get_logger(__name__)

# Pseudo-votes pulling every affinity towards zero: one thumb-up gives 1/3, three give 3/5.
SMOOTHING = 2
EXAMPLES_EACH = 5
MAX_VOTES = 400


@dataclass(slots=True)
class Affinity:
    source: dict[str, float] = field(default_factory=dict)
    topic: dict[str, float] = field(default_factory=dict)
    liked: list[str] = field(default_factory=list)
    disliked: list[str] = field(default_factory=list)
    votes: int = 0

    @property
    def is_empty(self) -> bool:
        return self.votes == 0

    def value(self, source_slug: str, topics: list[str]) -> float:
        """The candidate's affinity in (-1, 1); 0 when nothing is known about it."""
        known: list[float] = []
        if source_slug in self.source:
            known.append(self.source[source_slug])
        topic_values = [self.topic[t.lower()] for t in topics if t.lower() in self.topic]
        if topic_values:
            known.append(sum(topic_values) / len(topic_values))
        return sum(known) / len(known) if known else 0.0


def _smoothed(tally: dict[str, list[int]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, (ups, downs) in tally.items():
        value = (ups - downs) / (ups + downs + SMOOTHING)
        if value:
            result[key] = round(value, 4)
    return result


def affinity_from_votes(rows: list[tuple[uuid.UUID, int, str | None, str, list[str]]]) -> Affinity:
    """`rows` are (segment id, vote, heading, source slug, topics), newest first.

    A segment is built from several items, so it appears once per item; its vote counts
    once per source and once per topic.
    """
    sources: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    topics: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    counted_sources: set[tuple[uuid.UUID, str]] = set()
    counted_topics: set[tuple[uuid.UUID, str]] = set()
    voted: dict[uuid.UUID, int] = {}
    liked: dict[str, None] = {}
    disliked: dict[str, None] = {}

    for segment_id, vote, heading, slug, item_topics in rows:
        slot = 0 if vote > 0 else 1
        if (segment_id, slug) not in counted_sources:
            counted_sources.add((segment_id, slug))
            sources[slug][slot] += 1
        for topic in item_topics or []:
            key = (segment_id, topic.lower())
            if key not in counted_topics:
                counted_topics.add(key)
                topics[topic.lower()][slot] += 1
        voted[segment_id] = vote
        if heading:
            (liked if vote > 0 else disliked).setdefault(heading, None)

    return Affinity(
        source=_smoothed(sources),
        topic=_smoothed(topics),
        liked=list(liked)[:EXAMPLES_EACH],
        disliked=list(disliked)[:EXAMPLES_EACH],
        votes=len(voted),
    )


def load_affinity(session: Session, user_id: uuid.UUID) -> Affinity:
    """The listener's affinities. Empty (no effect) when there are no votes or on any error:
    feedback sharpens a ranking, it is never a precondition for one."""
    try:
        rows = session.execute(
            select(Feedback.segment_id, Feedback.vote, Segment.heading, Source.slug, Item.topics)
            .join(Segment, Segment.id == Feedback.segment_id)
            .join(Item, Item.cluster_id == Segment.cluster_id)
            .join(Source, Source.id == Item.source_id)
            .where(Feedback.user_id == user_id)
            .order_by(Feedback.created_at.desc())
            .limit(MAX_VOTES * 8)
        ).all()
        affinity = affinity_from_votes([(r[0], r[1], r[2], r[3], r[4]) for r in rows])
    except Exception as exc:
        log.warning("feedback.load_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        return Affinity()
    if not affinity.is_empty:
        log.info(
            "feedback.loaded",
            votes=affinity.votes,
            sources=len(affinity.source),
            topics=len(affinity.topic),
        )
    return affinity


def examples_block(affinity: Affinity | None) -> str:
    """Recent thumbs as examples for the re-rank prompt; empty when there are none."""
    if affinity is None or not (affinity.liked or affinity.disliked):
        return ""
    lines = ["\nStories they gave a thumbs-up or thumbs-down to recently (learn their taste):"]
    lines += [f"- liked: {title[:140]}" for title in affinity.liked]
    lines += [f"- disliked: {title[:140]}" for title in affinity.disliked]
    return "\n".join(lines)
