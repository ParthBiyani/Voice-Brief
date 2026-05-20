"""Personalized ranking.

Two stages, and the split is the whole design:

1. **Heuristic pre-rank** — free, deterministic, runs over all ~60 clusters. Scores
   stack-profile overlap, declared topics, recency and source trust.
2. **LLM re-rank** — runs over only the top ~15 survivors, in one batched call.

Why not send all 60 to the model: measured against live pricing, summarizing and
scoring every cluster costs about ₹7.5 per episode against a ₹8 total budget. The
pre-rank exists to make the expensive stage affordable, and it is also the *only*
stage that runs when no LLM is configured, so the system degrades to a working
heuristic ranker rather than to nothing.

The stack profile is what makes this personalization rather than popularity sorting.
A dependency the user actually has in a manifest is worth far more than any amount of
upstream engagement, because it is the only signal that can produce:

    "You use LangGraph in ContextPilot and PlacementPilot — this replaces the
     checkpoint workaround in both."
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from voicebrief.llm.base import Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger
from voicebrief.personalization.github_profile import StackProfileData

log = get_logger(__name__)

# Heuristic weights. Dependency hits dominate deliberately — see the module docstring.
W_DEPENDENCY = 3.0
W_LANGUAGE = 0.8
W_TOPIC = 1.2
W_STARRED = 0.4
W_ENGAGEMENT = 0.6
W_TRUST = 1.0
W_CLUSTER_SIZE = 0.5

RERANK_CANDIDATES = 15


@dataclass(slots=True)
class RankCandidate:
    """A story to be ranked. One per cluster, not one per item."""

    cluster_id: uuid.UUID
    title: str
    summary: str
    url: str
    topics: list[str] = field(default_factory=list)
    engagement: float = 0.0
    trust_weight: float = 0.5
    cluster_size: int = 1
    source_slug: str = ""
    # The items behind this cluster, carried through so the summarizer has the raw
    # material without a second database round-trip per story.
    sources: list = field(default_factory=list)
    bodies: dict = field(default_factory=dict)


@dataclass(slots=True)
class RankedStory:
    candidate: RankCandidate
    score: float
    heuristic_score: float
    llm_score: float | None = None
    # Which of the user's dependencies this story touches, and where they use them.
    matched_dependencies: dict[str, list[str]] = field(default_factory=dict)
    rationale: str = ""

    @property
    def is_personal(self) -> bool:
        return bool(self.matched_dependencies)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — heuristic
# ─────────────────────────────────────────────────────────────────────────────
def match_dependencies(
    text: str, profile: StackProfileData | None
) -> dict[str, list[str]]:
    """Find the user's actual dependencies mentioned in a story.

    Matching is on word boundaries against a lowercased haystack. Short names are
    skipped: a two-character dependency would match inside half the words in English
    and turn the strongest signal in the system into noise.
    """
    if profile is None or not profile.dependencies:
        return {}

    haystack = f" {text.lower()} "
    matched: dict[str, list[str]] = {}
    for name, info in profile.dependencies.items():
        if len(name) < 4:
            continue
        # Dependency names appear as words, in slashes, or in "org/name" form.
        for form in (f" {name} ", f"/{name}", f" {name}.", f" {name},", f" {name}:"):
            if form in haystack:
                matched[name] = info.get("repos", [])
                break
    return matched


def heuristic_score(
    candidate: RankCandidate,
    *,
    profile: StackProfileData | None,
    declared_topics: set[str],
) -> tuple[float, dict[str, list[str]]]:
    text = f"{candidate.title} {candidate.summary}"
    matched = match_dependencies(text, profile)

    score = 0.0
    score += W_DEPENDENCY * min(len(matched), 3)

    if profile:
        lowered = text.lower()
        score += W_LANGUAGE * sum(
            1 for language in profile.languages if language in lowered
        )
        score += W_STARRED * len(
            {t.lower() for t in candidate.topics} & set(profile.starred_topics)
        )

    if declared_topics:
        overlap = {t.lower() for t in candidate.topics} & declared_topics
        score += W_TOPIC * min(len(overlap), 3)

    score += W_ENGAGEMENT * candidate.engagement
    score += W_TRUST * candidate.trust_weight
    # A story several sources covered independently is more likely to matter.
    score += W_CLUSTER_SIZE * min(candidate.cluster_size - 1, 3)

    return score, matched


def prerank(
    candidates: list[RankCandidate],
    *,
    profile: StackProfileData | None = None,
    declared_topics: set[str] | None = None,
) -> list[RankedStory]:
    topics = {t.lower() for t in (declared_topics or set())}
    ranked = []
    for candidate in candidates:
        score, matched = heuristic_score(
            candidate, profile=profile, declared_topics=topics
        )
        ranked.append(
            RankedStory(
                candidate=candidate,
                score=score,
                heuristic_score=score,
                matched_dependencies=matched,
            )
        )
    ranked.sort(key=lambda s: s.score, reverse=True)
    log.info(
        "rank.prerank",
        candidates=len(candidates),
        personal=sum(1 for s in ranked if s.is_personal),
    )
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — LLM re-rank
# ─────────────────────────────────────────────────────────────────────────────
RERANK_SYSTEM = """\
You rank technology news for one specific engineer's daily audio brief.

You are given that engineer's stack profile and a numbered list of candidate stories.
Score each story 0-10 for how much it deserves a slot in *their* brief today.

What earns a high score:
- It changes something in a library or tool they actually depend on.
- It is a concrete release, breaking change, or capability they could use this week.
- It is a genuine shift in an area they work in.

What earns a low score:
- Generic industry news with no bearing on their stack.
- Incremental research with no practical implication for a practitioner.
- Marketing, funding rounds, and product launches unrelated to their work.

Judge relevance to this engineer, not general importance. A widely-covered story that
does not touch their stack scores lower than a small release that does.

Return JSON only.\
"""


RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["index", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}


def _profile_brief(profile: StackProfileData | None) -> str:
    if profile is None or profile.is_empty:
        return "No stack profile available. Judge on general practitioner relevance."

    languages = ", ".join(list(profile.languages)[:6]) or "unknown"
    deps = sorted(profile.dependencies, key=lambda d: -len(profile.dependencies[d]["repos"]))
    dependency_lines = [
        f"- {name} (used in {', '.join(r.split('/')[-1] for r in profile.dependencies[name]['repos'][:3])})"
        for name in deps[:25]
    ]
    return (
        f"Languages: {languages}\n"
        f"Repositories: {len(profile.repos)}\n"
        f"Dependencies they actually build on:\n" + "\n".join(dependency_lines)
    )


def rerank(
    stories: list[RankedStory],
    client: LLMClient,
    *,
    profile: StackProfileData | None = None,
    limit: int = RERANK_CANDIDATES,
) -> list[RankedStory]:
    """Re-score the top `limit` stories in one batched call.

    One call for fifteen stories rather than fifteen calls: the system prompt and the
    stack profile are the bulk of the input, and repeating them per story would cost
    roughly ten times as much for the same judgement.

    On any model failure the heuristic order stands. Ranking is a quality improvement,
    not a dependency — an episode with heuristic ordering is still a good episode.
    """
    head = stories[:limit]
    if not head:
        return stories

    listing = "\n\n".join(
        f"[{i}] {s.candidate.title}\n"
        f"    source: {s.candidate.source_slug} | covered by {s.candidate.cluster_size} item(s)\n"
        f"    {s.candidate.summary[:400]}"
        + (
            f"\n    touches their dependencies: {', '.join(s.matched_dependencies)}"
            if s.matched_dependencies
            else ""
        )
        for i, s in enumerate(head)
    )

    prompt = (
        f"Engineer's stack profile:\n{_profile_brief(profile)}\n\n"
        f"Candidate stories:\n{listing}\n\n"
        f"Score all {len(head)} stories. Use the index shown in brackets."
    )

    try:
        completion = client.complete(
            prompt=prompt,
            stage=Stage.ranking,
            tier=Tier.utility,
            system=RERANK_SYSTEM,
            schema=RERANK_SCHEMA,
            max_tokens=4000,
            # The system prompt and profile are identical across runs in a session.
            cache_system=True,
        )
    except Exception as exc:  # noqa: BLE001 — ranking must never fail an episode
        log.warning("rank.rerank_failed", error=str(exc), fallback="heuristic order")
        return stories

    payload = completion.parsed
    if payload is None and completion.text:
        try:
            payload = json.loads(completion.text)
        except json.JSONDecodeError:
            log.warning("rank.rerank_unparseable", fallback="heuristic order")
            return stories

    for entry in (payload or {}).get("rankings", []):
        index = entry.get("index")
        if not isinstance(index, int) or not 0 <= index < len(head):
            continue
        story = head[index]
        story.llm_score = float(entry.get("score", 0))
        story.rationale = str(entry.get("reason", ""))[:400]
        # Blend rather than replace. The heuristic carries the dependency evidence,
        # which is factual; the model contributes judgement about significance.
        story.score = 0.4 * story.heuristic_score + 0.6 * story.llm_score

    head.sort(key=lambda s: s.score, reverse=True)
    scored = sum(1 for s in head if s.llm_score is not None)
    log.info("rank.reranked", scored=scored, of=len(head))
    return head + stories[limit:]


def select_for_episode(
    stories: list[RankedStory], *, max_stories: int = 10, min_personal: int = 2
) -> list[RankedStory]:
    """Pick the final running order.

    Guarantees at least `min_personal` stack-relevant stories when any exist. Without
    this the ranker drifts toward whatever is objectively biggest that day, and the
    brief stops being personal — which is the entire product.
    """
    ordered = sorted(stories, key=lambda s: s.score, reverse=True)
    selected = ordered[:max_stories]

    if min_personal and sum(1 for s in selected if s.is_personal) < min_personal:
        personal = [s for s in ordered if s.is_personal and s not in selected]
        needed = min_personal - sum(1 for s in selected if s.is_personal)
        for story in personal[:needed]:
            # Displace the weakest non-personal story rather than growing the episode.
            for i in range(len(selected) - 1, -1, -1):
                if not selected[i].is_personal:
                    selected[i] = story
                    break
        selected.sort(key=lambda s: s.score, reverse=True)

    log.info(
        "rank.selected",
        stories=len(selected),
        personal=sum(1 for s in selected if s.is_personal),
    )
    return selected
