"""Cheap pre-LLM filter.

This is the stage that makes the cost target achievable. Per PRD §3 the funnel is
1500 → ~300 here, and only ~40 items ever reach the ranking model. Every decision in
this module is arithmetic on data we already have: no embeddings, no network, no
tokens. That is the point — an expensive filter defeats its own purpose.

Scoring is a weighted sum of four signals:

  recency     — exponential decay; a 6-hour-old release beats a 3-day-old one
  trust       — the source's prior, straight from the registry
  engagement  — upstream popularity, already normalized 0..1 by the adapter
  topic match — overlap with what the user declared they care about

Deliberately *not* here: anything requiring the user's stack profile. That is a
ranking concern, and doing it here would mean recomputing the filter per user.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from voicebrief.db.models import Item
from voicebrief.logging import get_logger

log = get_logger(__name__)

# Half-life in hours: an item loses half its recency score every 18 hours. Tuned so a
# twice-daily brief still surfaces yesterday evening's news this morning.
RECENCY_HALF_LIFE_HOURS = 18.0

WEIGHTS = {
    "recency": 0.35,
    "trust": 0.30,
    "engagement": 0.20,
    "topic": 0.15,
}

# Titles that are almost never worth a segment. Kept small and specific on purpose —
# an aggressive blocklist silently destroys recall and nobody notices.
NOISE_PATTERNS = re.compile(
    r"^(ask hn|tell hn|show hn: my|hiring|who is hiring|freelancer\?|"
    r"weekly discussion|daily thread|simple questions|monthly thread)",
    re.IGNORECASE,
)

MIN_TITLE_WORDS = 3
MIN_TITLE_CHARS = 8

# Repo- and model-shaped identifiers ("langchain-ai/langgraph", "org/model v1.2").
# These are legitimate titles that happen to contain almost no whitespace, so the
# word-count heuristic must not be applied to them.
IDENTIFIER_RE = re.compile(r"^[\w.-]+/[\w.-]+")


@dataclass(slots=True)
class ScoredItem:
    item: Item
    score: float
    reasons: dict[str, float]


def _recency_score(published_at: datetime, *, now: datetime) -> float:
    age_hours = max((now - published_at).total_seconds() / 3600.0, 0.0)
    return math.pow(0.5, age_hours / RECENCY_HALF_LIFE_HOURS)


def _topic_score(item_topics: list[str], user_topics: set[str]) -> float:
    """Jaccard-ish overlap, but asymmetric.

    We care what fraction of the *user's* interests an item touches, not what fraction
    of the item's tags the user shares — a densely tagged Hub model shouldn't be
    penalised for carrying tags the user never asked about.
    """
    if not user_topics:
        return 0.5  # no declared preference: stay neutral rather than zero everything
    if not item_topics:
        return 0.0
    overlap = user_topics & {t.lower() for t in item_topics}
    return min(len(overlap) / max(len(user_topics), 1) * 2.0, 1.0)


def is_noise(item: Item) -> bool:
    """Structural junk, decided without a model.

    The word-count rule is skipped for repo/model identifiers. A naive
    `len(title.split()) < 3` looked harmless in unit tests but, on a real crawl,
    silently deleted every GitHub repo, every GitHub release and every Hugging Face
    model — 59 items, three entire source types — because "langchain-ai/langgraph"
    is one word. Cheap heuristics fail loudest on the sources you care most about.
    """
    title = (item.title or "").strip()
    if not title or not item.url:
        return True
    if NOISE_PATTERNS.match(title):
        return True
    if IDENTIFIER_RE.match(title):
        return len(title) < MIN_TITLE_CHARS
    return len(title.split()) < MIN_TITLE_WORDS


def score_item(
    item: Item,
    *,
    trust_weight: float,
    user_topics: set[str],
    now: datetime | None = None,
) -> ScoredItem:
    now = now or datetime.now(timezone.utc)
    published = item.published_at
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    reasons = {
        "recency": _recency_score(published, now=now),
        "trust": trust_weight,
        "engagement": float(item.engagement or 0.0),
        "topic": _topic_score(list(item.topics or []), user_topics),
    }
    score = sum(WEIGHTS[key] * value for key, value in reasons.items())
    return ScoredItem(item=item, score=score, reasons=reasons)


def prefilter(
    items: list[Item],
    *,
    trust_by_source: dict,
    user_topics: set[str] | None = None,
    keep: int = 300,
    now: datetime | None = None,
    min_score: float = 0.0,
    max_per_source_ratio: float = 0.25,
) -> list[ScoredItem]:
    """Reduce a day's crawl to the top `keep` items.

    Returns them sorted best-first. The caller decides what to do with the tail; this
    function's only job is to be cheap and to never let a high-trust release get
    buried under a thousand arXiv preprints.

    `max_per_source_ratio` caps any single source's share of the survivors. See
    `_enforce_diversity` for why that cap is not optional.
    """
    now = now or datetime.now(timezone.utc)
    topics = {t.lower() for t in (user_topics or set())}

    scored: list[ScoredItem] = []
    dropped_noise = 0

    for item in items:
        if is_noise(item):
            dropped_noise += 1
            continue
        trust = float(trust_by_source.get(item.source_id, 0.5))
        candidate = score_item(item, trust_weight=trust, user_topics=topics, now=now)
        if candidate.score >= min_score:
            scored.append(candidate)

    scored.sort(key=lambda s: s.score, reverse=True)
    result = _enforce_diversity(scored, keep=keep, max_ratio=max_per_source_ratio)

    log.info(
        "prefilter.done",
        given=len(items),
        noise=dropped_noise,
        scored=len(scored),
        kept=len(result),
        sources=len({r.item.source_id for r in result}),
        floor=round(result[-1].score, 4) if result else None,
    )
    return result


def _enforce_diversity(
    scored: list[ScoredItem], *, keep: int, max_ratio: float
) -> list[ScoredItem]:
    """Stop one high-volume source from monopolising the survivors.

    Measured on a real 560-item crawl, pure score ordering handed arXiv 235 of 300
    slots and evicted every GitHub release — including sources with a *higher* trust
    weight — simply because arXiv publishes hundreds of papers an hour and they are
    all equally fresh. Volume was beating relevance.

    So each source gets a quota. Items beyond it are held back, and the leftover
    capacity is refilled in score order once every source has had its share, which
    keeps the cap from wasting slots when there genuinely isn't enough diversity.
    """
    if not scored:
        return []

    quota = max(1, int(keep * max_ratio))
    taken: dict = {}
    primary: list[ScoredItem] = []
    overflow: list[ScoredItem] = []

    for candidate in scored:
        source_id = candidate.item.source_id
        if taken.get(source_id, 0) < quota:
            taken[source_id] = taken.get(source_id, 0) + 1
            primary.append(candidate)
        else:
            overflow.append(candidate)

    if len(primary) >= keep:
        return primary[:keep]

    # Backfill in score order — a cap that leaves slots empty is worse than no cap.
    return primary + overflow[: keep - len(primary)]
