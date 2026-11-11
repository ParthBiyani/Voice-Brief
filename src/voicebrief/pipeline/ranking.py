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
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from voicebrief.llm.base import BudgetExceeded, Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger
from voicebrief.personalization.feedback import Affinity, examples_block
from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.pipeline.topics import content_topics

log = get_logger(__name__)

# Heuristic weights. Dependency hits dominate deliberately — see the module docstring.
W_DEPENDENCY = 3.0
W_LANGUAGE = 0.8
W_TOPIC = 1.2
# A declared topic the story's source is tagged with but its own words do not mention.
W_SOURCE_TOPIC = 0.4
W_STARRED = 0.4
W_ENGAGEMENT = 0.6
W_TRUST = 1.0
W_CLUSTER_SIZE = 0.5
# Without a recency term the biggest cluster of all time beats today's single-item
# release, which is exactly the story the dependency signal exists to surface.
W_RECENCY = 2.0
RECENCY_HALF_LIFE_HOURS = 24.0
# Personal context from ProjectMind: a tool the listener reaches for is worth a nudge,
# one they moved away from is a smaller one in the other direction.
W_INTEREST = 1.0
W_MOVED_AWAY = 1.5
# A story with almost no source text (a trending repo with a one-line description)
# yields a segment that can only say "the sources give no further detail". It is
# demoted below substantive stories but not removed, so a quiet day still fills.
W_THIN = 2.0
# Thumbs: at most this much either way, so taste can reorder similar stories but never
# bury a release of something the listener depends on (W_DEPENDENCY is 3.0).
W_FEEDBACK = 1.5
THIN_STORY_CHARS = 250

# Package names that are also everyday English words. A dependency called `next` or
# `rich` matches ordinary prose, which would flag unrelated stories as "personal" and
# put a confidently wrong "you use this" claim into the audio. These only count when
# written the way a package is written: `org/name`, `name.js` or `name.py`.
AMBIGUOUS_DEPENDENCIES = frozenset(
    {
        "next", "rich", "click", "black", "requests", "express", "build", "test", "tests",
        "time", "start", "base", "core", "common", "utils", "types", "main", "app",
        "server", "client", "markdown", "json", "yaml", "toml", "html", "css", "logging",
        "typing", "http", "socket", "random", "string", "docs", "data", "model", "agent",
        "agents", "tools", "config", "cache", "queue", "events", "image", "audio",
        "video", "search", "graph", "chart", "table", "form", "router", "layout",
        # ML vocabulary that is also a library name: "looped transformers" is the
        # architecture, not Hugging Face's package.
        "transformers", "transformer", "datasets", "accelerate", "evaluate", "tokenizers",
    }
)  # fmt: skip

# Common English words, taken from the embedding model's own vocabulary (see
# `personalization/common_words.txt`). A dependency called `path`, `http` or `catalogue`
# matches ordinary prose, and a hand-kept stop-list can never keep up with the names a
# frozen requirements file drags in. Anything on this list needs package-style evidence.
_COMMON_WORDS_FILE = Path(__file__).resolve().parent.parent / "personalization" / "common_words.txt"


def _load_common_words() -> frozenset[str]:
    try:
        return frozenset(
            w.strip() for w in _COMMON_WORDS_FILE.read_text(encoding="utf-8").splitlines() if w
        )
    except OSError:
        return frozenset()


COMMON_WORDS = _load_common_words()
# Libraries whose names are words but which tech news uses as proper nouns ("React 19").
KNOWN_LIBRARIES_THAT_ARE_WORDS = frozenset({"react"})


def is_ambiguous_dependency(name: str) -> bool:
    lowered = name.lower()
    if lowered in KNOWN_LIBRARIES_THAT_ARE_WORDS:
        return False
    return lowered in AMBIGUOUS_DEPENDENCIES or lowered in COMMON_WORDS


# Languages whose name is a common word or a single letter; matched only by the
# unambiguous form (empty string = never matched from prose).
AMBIGUOUS_LANGUAGES = {"go": "golang", "c": "", "r": "", "d": "", "v": ""}
_WORD_RE = re.compile(r"[a-z0-9+#]+")

# How many heuristic leaders the model re-scores, in one call. Measured offline against
# independent labels (eval/EXPERIMENTS.md): P@10 0.567 at 15, 0.600 at 25, 0.667 at 30,
# 0.633 at 40 (noise from here on). Anything the heuristic puts below this line can never
# reach the brief, and the model is the better judge, so it sees thirty: about five paise
# more a day in input tokens, no extra call.
RERANK_CANDIDATES = 30


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
    # When the story effectively became news (see filtering.item_freshness). None means
    # unknown and the recency term is skipped: the eval corpus has no ingest times, and
    # inventing one would make the harness measure something fictional.
    freshness: datetime | None = None
    # URLs of every item in the story, used to recognise a story already aired.
    item_urls: list = field(default_factory=list)
    # Total characters of source text behind the story; None when unknown (eval corpus).
    substance_chars: int | None = None


@dataclass(slots=True)
class RankedStory:
    candidate: RankCandidate
    score: float
    heuristic_score: float
    llm_score: float | None = None
    # Which of the user's dependencies this story touches, and where they use them.
    matched_dependencies: dict[str, list[str]] = field(default_factory=dict)
    # Tools from the listener's stated interests (ProjectMind) the story is about.
    matched_interests: list[str] = field(default_factory=list)
    rationale: str = ""
    # How much the whole industry is talking about it today, whoever the listener is
    # (0-10, from the re-rank call). None when no model scored it.
    buzz: float | None = None
    # Where it sits in the running order: "headline" (the day's big news), "spotlight" (a
    # rising repository worth knowing), or "relevant" (chosen for this listener).
    role: str = "relevant"
    # What each heuristic signal contributed, so a decision can be explained afterwards.
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def is_personal(self) -> bool:
        return bool(self.matched_dependencies or self.matched_interests)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — heuristic
# ─────────────────────────────────────────────────────────────────────────────
def match_dependencies(text: str, profile: StackProfileData | None) -> dict[str, list[str]]:
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
        if is_ambiguous_dependency(name):
            forms: tuple[str, ...] = (f"/{name}", f" {name}.js", f" {name}.py", f" {name}-js")
        else:
            # Dependency names appear as words, in slashes, or in "org/name" form.
            forms = (f" {name} ", f"/{name}", f" {name}.", f" {name},", f" {name}:")
        if any(form in haystack for form in forms):
            matched[name] = info.get("repos", [])
    return matched


def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    usable = [t for t in terms if t and len(t) >= 3 and not is_ambiguous_dependency(t)]
    if not usable:
        return None
    alternation = "|".join(sorted((re.escape(t) for t in usable), key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])")


@lru_cache(maxsize=8)
def _interest_pattern(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    return _term_pattern(terms)


def match_interests(text: str, profile: StackProfileData | None) -> tuple[list[str], list[str]]:
    """(interests the story is about, tools-moved-away-from it is about).

    Same discipline as dependency matching: whole words only, and a name that is also an
    ordinary English word does not count on its own.
    """
    if profile is None or not (profile.interests or profile.avoid):
        return [], []
    lowered = text.lower()
    interests: list[str] = []
    avoided: list[str] = []
    if pattern := _interest_pattern(tuple(profile.interests)):
        interests = sorted(set(pattern.findall(lowered)))
    if pattern := _interest_pattern(tuple(profile.avoid)):
        avoided = sorted(set(pattern.findall(lowered)))
    return interests, avoided


def mentions_language(text: str, language: str) -> bool:
    """Whole-word match. A substring test made `c` match nearly every sentence and
    `go` match "google" and "ago"."""
    language = language.lower()
    words = _WORD_RE.findall(text.lower())
    if language in AMBIGUOUS_LANGUAGES:
        unambiguous = AMBIGUOUS_LANGUAGES[language]
        return bool(unambiguous) and unambiguous in words
    return language in words


def recency_score(freshness: datetime | None, *, now: datetime) -> float:
    if freshness is None:
        return 0.0
    if freshness.tzinfo is None:
        freshness = freshness.replace(tzinfo=UTC)
    age_hours = max((now - freshness).total_seconds() / 3600.0, 0.0)
    return math.pow(0.5, age_hours / RECENCY_HALF_LIFE_HOURS)


def score_parts(
    candidate: RankCandidate,
    *,
    profile: StackProfileData | None,
    declared_topics: set[str],
    now: datetime | None = None,
    affinity: Affinity | None = None,
) -> tuple[float, dict[str, list[str]], dict[str, float]]:
    """The heuristic score, the dependencies it matched, and each signal's contribution."""
    text = f"{candidate.title} {candidate.summary}"
    matched = match_dependencies(text, profile)
    about_text = content_topics(text)
    source_tags = {t.lower() for t in candidate.topics}

    def topic_overlap(wanted: set[str], full: float, prior: float) -> float:
        # What the story is about counts fully; the source's blanket tag only as a prior.
        # Scored on source tags alone, every paper from one feed tied (see topics.py).
        about = about_text & wanted
        tagged = (source_tags & wanted) - about
        return full * min(len(about), 3) + prior * min(len(tagged), 3)

    parts: dict[str, float] = {"stack": W_DEPENDENCY * min(len(matched), 3)}

    if profile:
        parts["language"] = W_LANGUAGE * sum(
            1 for language in profile.languages if mentions_language(text, language)
        )
        parts["starred"] = topic_overlap(
            set(profile.starred_topics), W_STARRED, W_STARRED * W_SOURCE_TOPIC / W_TOPIC
        )
        interests, avoided = match_interests(text, profile)
        if interests:
            parts["interests"] = W_INTEREST * min(len(interests), 3)
        if avoided and not interests:
            parts["moved_away"] = -W_MOVED_AWAY

    if declared_topics:
        parts["topics"] = topic_overlap(declared_topics, W_TOPIC, W_SOURCE_TOPIC)

    parts["engagement"] = W_ENGAGEMENT * candidate.engagement
    parts["trust"] = W_TRUST * candidate.trust_weight
    # A story several sources covered independently is more likely to matter.
    parts["coverage"] = W_CLUSTER_SIZE * min(candidate.cluster_size - 1, 3)
    parts["recency"] = W_RECENCY * recency_score(candidate.freshness, now=now or datetime.now(UTC))
    if candidate.substance_chars is not None and candidate.substance_chars < THIN_STORY_CHARS:
        parts["thin_penalty"] = -W_THIN
    if affinity is not None and not affinity.is_empty:
        parts["feedback"] = W_FEEDBACK * affinity.value(candidate.source_slug, candidate.topics)

    rounded = {k: round(v, 4) for k, v in parts.items() if v}
    return sum(parts.values()), matched, rounded


def heuristic_score(
    candidate: RankCandidate,
    *,
    profile: StackProfileData | None,
    declared_topics: set[str],
    now: datetime | None = None,
    affinity: Affinity | None = None,
) -> tuple[float, dict[str, list[str]]]:
    score, matched, _ = score_parts(
        candidate, profile=profile, declared_topics=declared_topics, now=now, affinity=affinity
    )
    return score, matched


def prerank(
    candidates: list[RankCandidate],
    *,
    profile: StackProfileData | None = None,
    declared_topics: set[str] | None = None,
    now: datetime | None = None,
    affinity: Affinity | None = None,
) -> list[RankedStory]:
    topics = {t.lower() for t in (declared_topics or set())}
    now = now or datetime.now(UTC)
    ranked = []
    for candidate in candidates:
        score, matched, parts = score_parts(
            candidate, profile=profile, declared_topics=topics, now=now, affinity=affinity
        )
        interests, _avoided = match_interests(f"{candidate.title} {candidate.summary}", profile)
        ranked.append(
            RankedStory(
                candidate=candidate,
                score=score,
                heuristic_score=score,
                matched_dependencies=matched,
                matched_interests=interests,
                signals=parts,
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
You decide what goes into one specific engineer's daily audio brief. The brief is as long
as the day's news deserves: a few minutes when a lot happened, a minute or two when little
did. Nothing is included to fill time, so be strict.

You are given that engineer's stack profile and a numbered list of candidate stories.
Score each story 0-10 for its IMPACT: how much it changes for someone like them.

Judge impact, not novelty. A good test: can you say in one sentence what this changes for
a person who uses it, and would that sentence matter to them? If the honest sentence is
"it adds a small option", it is not impact.

- 9-10: changes how they work or what they can build: a major release or breaking change
  in a library they depend on, a capability that removes a real workaround, a result that
  moves their field.
- 7-8: clearly worth their time: a meaningful gain in speed, cost, quality or capability
  in something they use or could adopt this week, or a notable release or finding any
  practitioner should hear.
- 5-6: interesting, optional. Worth a line on a slow day, not a place in the brief.
- 0-4: routine. Bug fixes, small features, minor and patch releases, version bumps,
  dependency updates, documentation, incremental research, marketing, funding rounds,
  anything unrelated to their work and significant to nobody. Most candidates belong here,
  including most release notes, even for libraries they use.

Also give every story a `buzz` score 0-10, separate from impact: how much the whole
industry is talking about it today, whoever is listening. 9-10 is for the stories everyone
in tech will be discussing (a new flagship model from a leading lab, a major acquisition, a
serious security incident, a launch that changes what developers can do). 4-6 is notable.
Most stories are 0-3. For a new repository, buzz is how remarkable it is to any developer,
whatever its field.

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
                    "buzz": {"type": "integer", "minimum": 0, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["index", "score", "buzz", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}


def _personal_brief(profile: StackProfileData) -> str:
    """What the listener has told their memory system, for the re-rank prompt."""
    lines: list[str] = []
    if profile.notes:
        lines.append("\nHow they say they work (from their own notes):")
        lines.extend(f"- {note[:220]}" for note in profile.notes[:12])
    if profile.avoid:
        lines.append("Tools they have moved away from: " + ", ".join(profile.avoid[:12]))
    if profile.projects:
        lines.append("Recently building: " + ", ".join(profile.projects[:8]))
    return "\n".join(lines)


def _profile_brief(profile: StackProfileData | None) -> str:
    if profile is None or profile.is_empty:
        return "No stack profile available. Judge on general practitioner relevance."

    languages = ", ".join(list(profile.languages)[:6]) or "unknown"
    personal = _personal_brief(profile)
    if not profile.has_stack:
        return personal or "No stack profile available. Judge on general practitioner relevance."
    deps = sorted(profile.dependencies, key=lambda d: -len(profile.dependencies[d]["repos"]))
    dependency_lines = [
        f"- {name} (used in {', '.join(r.split('/')[-1] for r in profile.dependencies[name]['repos'][:3])})"
        for name in deps[:25]
    ]
    return (
        f"Languages: {languages}\n"
        f"Repositories: {len(profile.repos)}\n"
        f"Dependencies they actually build on:\n" + "\n".join(dependency_lines) + personal
    )


def rerank(
    stories: list[RankedStory],
    client: LLMClient,
    *,
    profile: StackProfileData | None = None,
    limit: int = RERANK_CANDIDATES,
    affinity: Affinity | None = None,
) -> list[RankedStory]:
    """Re-score the top `limit` stories in one batched call.

    One call for thirty stories rather than thirty calls: the system prompt and the
    stack profile are the bulk of the input, and repeating them per story would cost
    roughly ten times as much for the same judgement.

    On any model failure the heuristic order stands. Ranking is a quality improvement,
    not a dependency — an episode with heuristic ordering is still a good episode.
    """
    head = stories[:limit]
    if not head:
        return stories
    # The model scores what the heuristic ranks highest, and the heuristic favours the
    # listener's stack. A rising repository in some other field would never be seen, so a
    # few of those are always added to the call.
    extras = [s for s in stories[limit:] if is_spotlight_candidate(s)][:SPOTLIGHT_EXTRAS]
    head = head + extras

    listing = "\n\n".join(
        f"[{i}] {s.candidate.title}\n"
        f"    source: {s.candidate.source_slug} | covered by {s.candidate.cluster_size} item(s)\n"
        f"    {s.candidate.summary[:400]}"
        + (
            f"\n    touches their dependencies: {', '.join(s.matched_dependencies)}"
            if s.matched_dependencies
            else ""
        )
        + (
            f"\n    matches their stated interests: {', '.join(s.matched_interests)}"
            if s.matched_interests
            else ""
        )
        for i, s in enumerate(head)
    )

    prompt = (
        f"Engineer's stack profile:\n{_profile_brief(profile)}{examples_block(affinity)}\n\n"
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
    except BudgetExceeded:
        raise
    except Exception as exc:
        log.warning("rank.rerank_failed", error=str(exc), fallback="heuristic order")
        return stories

    payload = completion.parsed
    if payload is None and completion.text:
        try:
            payload = json.loads(completion.text)
        except json.JSONDecodeError:
            log.warning("rank.rerank_unparseable", fallback="heuristic order")
            return stories

    for entry in (payload if isinstance(payload, dict) else {}).get("rankings", []):
        index = entry.get("index")
        if not isinstance(index, int) or not 0 <= index < len(head):
            continue
        story = head[index]
        # The schema bounds the score to 0-10, but the API cannot enforce that, so it is
        # enforced here.
        story.llm_score = max(0.0, min(10.0, float(entry.get("score", 0))))
        story.rationale = str(entry.get("reason", ""))[:400]
        if isinstance(entry.get("buzz"), int | float):
            story.buzz = max(0.0, min(10.0, float(entry["buzz"])))
        # Blend rather than replace. The heuristic carries the dependency evidence,
        # which is factual; the model contributes judgement about significance.
        story.score = 0.4 * story.heuristic_score + 0.6 * story.llm_score

    head.sort(key=lambda s: s.score, reverse=True)
    scored = sum(1 for s in head if s.llm_score is not None)
    log.info("rank.reranked", scored=scored, of=len(head))

    # Stories the model never saw must live on the same scale as those it did. Left on
    # the raw heuristic they could outrank a story the model explicitly scored low, so
    # they are blended against the lowest model score seen: unreviewed never beats
    # reviewed-and-judged-relevant, but still orders by heuristic among themselves.
    tail = [s for s in stories[limit:] if s not in extras]
    judged = [s.llm_score for s in head if s.llm_score is not None]
    if judged:
        floor = min(judged)
        for story in tail:
            story.score = 0.4 * story.heuristic_score + 0.6 * floor
        tail.sort(key=lambda s: s.score, reverse=True)
    return head + tail


# Split on hyphens and slashes too, so "mistralai/Mistral-Large-4-1T" yields "mistral".
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TITLE_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "its", "has", "now", "new",
        "are", "was", "will", "you", "your", "our", "how", "why", "what", "into", "out",
        "release", "released", "releases", "version", "update", "updates", "announcing",
        "introducing", "model", "models", "dataset", "open", "source",
    }
)  # fmt: skip


def story_tokens(story: RankedStory) -> frozenset[str]:
    """Distinctive words across a story's titles — what it is *about*."""
    titles = [story.candidate.title, *(getattr(s, "title", "") for s in story.candidate.sources)]
    tokens: set[str] = set()
    for title in titles:
        tokens.update(
            t
            for t in _TOKEN_RE.findall((title or "").lower())
            if len(t) > 2 and t not in _TITLE_STOPWORDS
        )
    return frozenset(tokens)


def covers_same_ground(a: frozenset[str], b: frozenset[str]) -> bool:
    """Two stories about the same event, even when clustering kept them apart.

    Embedding clustering leaves a launch, the post announcing its weights and the
    library release that supports it as separate groups (they sit at 0.79-0.81, under
    the dedup threshold). Read aloud back to back that is the same news three times.
    """
    shared = a & b
    return len(shared) >= 2 and len(shared) / max(min(len(a), len(b)), 1) >= 0.4


def _without_repeats(ordered: list[RankedStory], limit: int) -> list[RankedStory]:
    """Greedy pick in score order, deferring stories that repeat an earlier pick."""
    picked: list[RankedStory] = []
    deferred: list[RankedStory] = []
    seen: list[frozenset[str]] = []
    for story in ordered:
        tokens = story_tokens(story)
        repeats = any(covers_same_ground(tokens, earlier) for earlier in seen)
        # Deferred stories still count as "already told": a third fragment of the same
        # event may overlap the second more than the first.
        seen.append(tokens)
        if repeats:
            deferred.append(story)
            continue
        picked.append(story)
        if len(picked) == limit:
            return picked
    # Better a slightly repetitive brief than a short one when the day is thin.
    return picked + deferred[: limit - len(picked)]


# Release notes for the listener's own dependencies are the most personal thing in the
# brief, and also the dullest to hear three of in a row: a list of version bumps and pull
# requests. Two is the most a brief carries; the rest wait for tomorrow or fall to the
# next best story.
MAX_RELEASE_STORIES = 2
RELEASE_SOURCE_PREFIX = "github-releases"


def is_release_story(story: RankedStory) -> bool:
    return story.candidate.source_slug.startswith(RELEASE_SOURCE_PREFIX)


def _cap_releases(
    selected: list[RankedStory], ordered: list[RankedStory], *, min_personal: int
) -> list[RankedStory]:
    """Swap surplus release stories for the best stories that are not releases."""
    capped = list(selected)
    spare = [s for s in ordered if s not in capped and not is_release_story(s)]
    while sum(1 for s in capped if is_release_story(s)) > MAX_RELEASE_STORIES and spare:
        victim = min((s for s in capped if is_release_story(s)), key=lambda s: s.score)
        personal_left = sum(1 for s in capped if s.is_personal) - (1 if victim.is_personal else 0)
        pick = next((s for s in spare if s.is_personal or personal_left >= min_personal), None)
        if pick is None:
            break
        capped[capped.index(victim)] = pick
        spare.remove(pick)
    capped.sort(key=lambda s: s.score, reverse=True)
    return capped


# The running order: the day's big news first, whether or not it touches the listener's
# code; then a rising repository or two; then what was chosen for them.
HEADLINE_BUZZ = 8.0
MAX_HEADLINES = 3
MAX_SPOTLIGHTS = 2
SPOTLIGHT_SOURCE_PREFIX = "github-trending"
SPOTLIGHT_MIN_CHARS = 400
SPOTLIGHT_EXTRAS = 4  # trending repos added to the re-rank call even when the heuristic is cold  # a repo with no README text has nothing to say on air


def is_spotlight_candidate(story: RankedStory) -> bool:
    candidate = story.candidate
    if not candidate.source_slug.startswith(SPOTLIGHT_SOURCE_PREFIX):
        return False
    return (candidate.substance_chars or 0) >= SPOTLIGHT_MIN_CHARS


def _swap_in(selected: list[RankedStory], story: RankedStory, protected: set[int]) -> None:
    """Put `story` in by displacing the weakest pick that is not protected."""
    if story in selected:
        return
    free = [s for s in selected if id(s) not in protected]
    if not free:
        return
    selected[selected.index(min(free, key=lambda s: (s.is_personal, s.score)))] = story


def arrange_running_order(
    selected: list[RankedStory], ordered: list[RankedStory], *, make_room: bool = True
) -> None:
    """Mark and order the picks: headlines, spotlights, then the rest by score.

    `selected` is changed in place. Headlines are the stories the model scored as
    industry-wide news; they are guaranteed a slot (and the first slots) even with no
    link to the listener, because missing the day's big story is the failure a daily
    brief cannot afford. Spotlights are the best rising repositories, one or two, also
    regardless of fit. Neither displaces a story about the listener's own dependencies.
    """
    hot = sorted(
        (s for s in ordered if (s.buzz or 0.0) >= HEADLINE_BUZZ and not is_release_story(s)),
        key=lambda s: s.buzz or 0.0,
        reverse=True,
    )[:MAX_HEADLINES]
    sparks = sorted(
        (s for s in ordered if is_spotlight_candidate(s) and s not in hot),
        key=lambda s: ((s.llm_score or 0.0) + (s.buzz or 0.0), s.score),
        reverse=True,
    )[:MAX_SPOTLIGHTS]

    if not make_room:  # the caller has chosen who is in; this only labels and orders
        hot = [s for s in hot if s in selected]
        sparks = [s for s in selected if is_spotlight_candidate(s) and s not in hot]
    protected = {id(s) for s in selected if s.matched_dependencies}
    for story in [*hot, *sparks]:
        if make_room:
            _swap_in(selected, story, protected)
        elif story not in selected:
            selected.append(story)
        protected.add(id(story))

    for story in selected:
        story.role = "relevant"
    for story in sparks:
        story.role = "spotlight"
    for story in hot:
        story.role = "headline"

    rank = {"headline": 0, "spotlight": 1, "relevant": 2}
    selected.sort(
        key=lambda s: (rank[s.role], -(s.buzz or 0.0) if s.role == "headline" else -s.score)
    )


def select_for_episode(
    stories: list[RankedStory], *, max_stories: int = 10, min_personal: int = 2
) -> list[RankedStory]:
    """Pick the final running order.

    Guarantees at least `min_personal` stack-relevant stories when any exist. Without
    this the ranker drifts toward whatever is objectively biggest that day, and the
    brief stops being personal — which is the entire product. No two picks should cover
    the same event.
    """
    ordered = sorted(stories, key=lambda s: s.score, reverse=True)
    selected = _without_repeats(ordered, max_stories)

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

    selected = _cap_releases(selected, ordered, min_personal=min_personal)
    arrange_running_order(selected, ordered)

    log.info(
        "rank.selected",
        stories=len(selected),
        personal=sum(1 for s in selected if s.is_personal),
    )
    return selected


# How long a brief runs is decided by the news, not by a story count. A story earns its
# place by impact (the re-rank score, 0-10); a quiet day yields a minute or two and a big
# one fills the ceiling. Small additions to a library are not impact, so they stay out
# however fresh they are.
IMPACT_FLOOR = 7.0
SPOTLIGHT_FLOOR = 5.0  # a rising repository only has to be interesting, not consequential
MAX_STORIES_CEILING = 14
FALLBACK_STORIES = 3  # with no model score at all, the heuristic's best few


def impact(story: RankedStory) -> float | None:
    return story.llm_score


def select_by_impact(
    stories: list[RankedStory], *, ceiling: int = MAX_STORIES_CEILING
) -> list[RankedStory]:
    """Every story whose impact clears the bar, the day's headlines, and a spotlight repo.

    A brief is never empty while any story exists: if nothing clears the bar, the best
    story or two are told short. Sources that reported nothing worth hearing are the
    quiet days the listener asked to be told about in a minute.
    """
    ordered = sorted(stories, key=lambda s: s.score, reverse=True)
    scored = [s for s in ordered if impact(s) is not None]

    if scored:
        qualified = [s for s in scored if (impact(s) or 0.0) >= IMPACT_FLOOR]
        qualified += [s for s in scored if (s.buzz or 0.0) >= HEADLINE_BUZZ and s not in qualified]
    else:  # the model could not be reached: the heuristic's order is all there is
        qualified = ordered[:FALLBACK_STORIES]

    selected = _without_repeats(sorted(qualified, key=lambda s: s.score, reverse=True), ceiling)
    selected = _cap_releases(selected, ordered, min_personal=0)

    # Spotlights are added on their own bar and never crowd out a qualifying story.
    sparks = [
        s
        for s in ordered
        if is_spotlight_candidate(s) and (impact(s) is None or (impact(s) or 0) >= SPOTLIGHT_FLOOR)
    ]
    sparks.sort(key=lambda s: ((s.llm_score or 0.0) + (s.buzz or 0.0), s.score), reverse=True)
    for story in sparks[:MAX_SPOTLIGHTS]:
        if story not in selected:
            selected.append(story)
    if not selected and ordered:
        selected = ordered[:1]

    arrange_running_order(selected, ordered, make_room=False)
    log.info(
        "rank.selected_by_impact",
        stories=len(selected),
        headlines=sum(1 for s in selected if s.role == "headline"),
        spotlights=sum(1 for s in selected if s.role == "spotlight"),
        floor=IMPACT_FLOOR,
    )
    return selected
