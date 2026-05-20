"""Brief generation graph.

LangGraph orchestrates the episode because the stages have real dependencies and real
failure modes, and a graph makes both inspectable in a trace rather than buried in a
function. The shape:

    rank -> summarize -> write segments -> assemble -> verify

`verify` is the important node. The PRD sets a hard bar — every sentence attributable
to a source span, zero URLs outside the source set — and a target nobody measures is a
wish. So verification runs inside the graph and its result is attached to the episode,
not left to a separate script somebody remembers to run.

Segments are written one call per story rather than one call for the whole episode.
That costs more tokens, and it is worth it: a single long call lets the model blur
facts between adjacent stories, and when grounding fails there is no way to tell which
story caused it. Per-segment calls make attribution checkable per segment.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from voicebrief.llm.base import Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger
from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.pipeline.grounding import verify_segment
from voicebrief.pipeline.ranking import (
    RankedStory,
    prerank,
    rerank,
    select_for_episode,
)
from voicebrief.pipeline.summarize import ClusterSummary, summarize_cluster

log = get_logger(__name__)

# Words per minute for a brisk news register. Used to target episode length.
WORDS_PER_MINUTE = 155
TARGET_MINUTES = 12


def _keep_last(_current, new):
    return new


class BriefState(TypedDict, total=False):
    """Graph state. Everything the nodes read or write."""

    candidates: Annotated[list, _keep_last]
    profile: Annotated[object, _keep_last]
    declared_topics: Annotated[set, _keep_last]
    language: Annotated[str, _keep_last]
    style: Annotated[str, _keep_last]

    selected: Annotated[list, _keep_last]
    summaries: Annotated[list, _keep_last]
    segments: Annotated[list, _keep_last]
    episode: Annotated[dict, _keep_last]
    errors: Annotated[list, _keep_last]


@dataclass(slots=True)
class Segment:
    position: int
    kind: str  # cold_open | agenda | story | also_noted | sign_off
    heading: str
    script: str
    citations: list[dict] = field(default_factory=list)
    cluster_id: uuid.UUID | None = None
    grounded_ratio: float = 1.0
    unsupported: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.script.split())


SEGMENT_SYSTEM = """\
You write one segment of a daily technical audio brief for a specific engineer.

This is spoken, not read. That changes the rules:
- Short sentences. No bullet points, no headings, no markdown, no emoji.
- Never read a URL aloud. Say "the release notes" or "the paper", not the address.
- Expand notation into speech: "version zero point four", "twelve percent".
- No preamble, no "in this segment", no sign-posting. Start with the news.

Grounding is not negotiable:
- Every factual claim must come from the FACTS provided. If it is not there, it does
  not go in the script.
- Do not estimate, extrapolate, or add context you happen to know. An absent detail
  stays absent.
- No speculation about what might happen next, what to watch for, or what something
  suggests. If the sources do not say it, neither do you.
- No editorial verdicts. Skip "that's a real jump", "worth keeping an eye on",
  "this is significant". State the fact and let it stand.
- If the facts are thin, write a shorter segment. Short and true beats long and
  padded.

When the story touches a dependency the engineer actually uses, say so plainly and
name the repository. That connection is the reason they are listening. Do not
manufacture one where it does not exist.\
"""


class BriefGraph:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_stories: int = 8,
        target_minutes: int = TARGET_MINUTES,
    ) -> None:
        self.client = client
        self.max_stories = max_stories
        self.target_minutes = target_minutes
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(BriefState)
        builder.add_node("rank", self.rank_node)
        builder.add_node("summarize", self.summarize_node)
        builder.add_node("write", self.write_node)
        builder.add_node("assemble", self.assemble_node)
        builder.add_node("verify", self.verify_node)

        builder.set_entry_point("rank")
        builder.add_edge("rank", "summarize")
        builder.add_edge("summarize", "write")
        builder.add_edge("write", "assemble")
        builder.add_edge("assemble", "verify")
        builder.add_edge("verify", END)
        return builder.compile()

    # ── nodes ─────────────────────────────────────────────────────────────────
    def rank_node(self, state: BriefState) -> dict:
        """Cheap heuristic pass over everything, then one model call over the head.

        Both stages, in order — skipping the prerank would send all ~60 clusters to
        the model and blow the per-episode budget on stories that cannot make the cut.
        """
        candidates = state.get("candidates", [])
        profile = state.get("profile")

        preranked: list[RankedStory] = prerank(
            candidates, profile=profile, declared_topics=state.get("declared_topics", set())
        )
        stories = rerank(preranked, self.client, profile=profile)
        selected = select_for_episode(stories, max_stories=self.max_stories)
        return {"selected": selected, "errors": state.get("errors", [])}

    def summarize_node(self, state: BriefState) -> dict:
        summaries = []
        errors = list(state.get("errors", []))
        for story in state.get("selected", []):
            summary = summarize_cluster(
                self.client,
                cluster_id=story.candidate.cluster_id,
                sources=story.candidate.sources,
                bodies=story.candidate.bodies,
            )
            if summary is None:
                errors.append(f"summary failed for {story.candidate.title[:60]}")
                continue
            summaries.append((story, summary))
        log.info("brief.summarized", stories=len(summaries), failed=len(errors))
        return {"summaries": summaries, "errors": errors}

    def write_node(self, state: BriefState) -> dict:
        summaries = state.get("summaries", [])
        if not summaries:
            return {"segments": [], "errors": [*state.get("errors", []), "no stories to write"]}

        profile = state.get("profile")
        language = state.get("language", "en")
        style = state.get("style", "solo_anchor")

        # Budget words across stories so the episode lands near its target length.
        story_budget = max(
            90, (self.target_minutes * WORDS_PER_MINUTE - 260) // max(len(summaries), 1)
        )

        segments: list[Segment] = [
            self._write_cold_open(summaries, language, style),
            self._write_agenda(summaries, language, style),
        ]
        for index, (story, summary) in enumerate(summaries):
            segment = self._write_story(
                story, summary, index + 2, story_budget, profile, language, style
            )
            if segment:
                segments.append(segment)

        segments.append(self._write_sign_off(len(segments), language, style))
        return {"segments": segments, "errors": state.get("errors", [])}

    def assemble_node(self, state: BriefState) -> dict:
        segments: list[Segment] = state.get("segments", [])
        for position, segment in enumerate(segments):
            segment.position = position

        words = sum(s.word_count for s in segments)
        episode = {
            "title": self._title(state, segments),
            "segments": segments,
            "word_count": words,
            "estimated_minutes": round(words / WORDS_PER_MINUTE, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "language": state.get("language", "en"),
            "style": state.get("style", "solo_anchor"),
        }
        return {"episode": episode}

    def verify_node(self, state: BriefState) -> dict:
        """Grounding gate. Runs inside the graph so no episode escapes unmeasured."""
        episode = dict(state.get("episode", {}))
        segments: list[Segment] = episode.get("segments", [])
        summaries = {
            story.candidate.cluster_id: summary for story, summary in state.get("summaries", [])
        }
        # Raw source text per cluster, so grounding is checked against the articles
        # themselves rather than the summarizer's compression of them.
        source_text = {
            story.candidate.cluster_id: " ".join(
                [*(story.candidate.bodies or {}).values(),
                 *[s.title for s in (story.candidate.sources or [])]]
            )
            for story, _ in state.get("summaries", [])
        }
        # The listener's own repos and dependencies are grounded facts about them,
        # sourced from the stack profile rather than from the day's articles.
        profile = state.get("profile")
        own_vocabulary: set[str] = set()
        if profile is not None:
            own_vocabulary |= {r.split("/")[-1].lower() for r in getattr(profile, "repos", [])}
            own_vocabulary |= {d.lower() for d in getattr(profile, "dependencies", {})}
            own_vocabulary |= {"you", "your", "listener"}

        total_sentences = 0
        supported_sentences = 0
        hallucinated_links: list[str] = []

        for segment in segments:
            if segment.kind != "story" or segment.cluster_id not in summaries:
                continue
            summary = summaries[segment.cluster_id]
            report = verify_segment(
                segment.script,
                summary,
                extra_vocabulary=own_vocabulary,
                source_text=source_text.get(segment.cluster_id, ""),
            )
            segment.grounded_ratio = report.grounded_ratio
            segment.unsupported = report.unsupported
            total_sentences += report.total_sentences
            supported_sentences += report.supported_sentences
            hallucinated_links.extend(report.hallucinated_links)

        episode["grounding"] = {
            "sentences": total_sentences,
            "supported": supported_sentences,
            "attribution_rate": round(
                supported_sentences / total_sentences if total_sentences else 1.0, 4
            ),
            "hallucinated_links": hallucinated_links,
        }
        log.info("brief.verified", **{k: v for k, v in episode["grounding"].items()
                                      if k != "hallucinated_links"})
        return {"episode": episode}

    # ── writers ───────────────────────────────────────────────────────────────
    def _write_story(
        self,
        story: RankedStory,
        summary: ClusterSummary,
        position: int,
        word_budget: int,
        profile: StackProfileData | None,
        language: str,
        style: str,
    ) -> Segment | None:
        personal = ""
        if story.matched_dependencies:
            uses = "; ".join(
                f"{dep} in {', '.join(r.split('/')[-1] for r in repos)}"
                for dep, repos in list(story.matched_dependencies.items())[:3]
                if repos
            )
            if uses:
                personal = (
                    f"\n\nTHIS LISTENER'S STACK: they use {uses}. "
                    f"Make the connection explicit and concrete."
                )

        prompt = (
            f"{summary.as_context()}{personal}\n\n"
            f"Write this as one spoken segment of about {word_budget} words"
            f"{' in Hindi' if language == 'hi' else ''}"
            f"{', as a two-host exchange marked HOST A: and HOST B:' if style == 'two_host' else ''}.\n"
            f"Use only the facts above."
        )

        try:
            completion = self.client.complete(
                prompt=prompt,
                stage=Stage.script,
                tier=Tier.flagship,
                system=SEGMENT_SYSTEM,
                max_tokens=1500,
                cache_system=True,
                fallback_tier=Tier.utility,
            )
        except Exception as exc:  # noqa: BLE001 — drop a story, keep the episode
            log.warning("brief.segment_failed", cluster=str(summary.cluster_id), error=str(exc))
            return None

        return Segment(
            position=position,
            kind="story",
            heading=summary.headline or story.candidate.title,
            script=completion.text.strip(),
            citations=[
                {"url": s.url, "title": s.title, "item_id": str(s.item_id)}
                for s in summary.sources
            ],
            cluster_id=summary.cluster_id,
        )

    def _write_cold_open(self, summaries, language: str, style: str) -> Segment:
        headlines = "; ".join(s.headline for _, s in summaries[:3] if s.headline)
        prompt = (
            f"Write a 25-word cold open for today's brief. Today's top items: "
            f"{headlines}.\nNo greeting cliches. Land on the single most important thing."
            f"{' Write in Hindi.' if language == 'hi' else ''}"
        )
        return self._simple_segment(prompt, "cold_open", "Cold open", 200)

    def _write_agenda(self, summaries, language: str, style: str) -> Segment:
        headlines = "\n".join(f"- {s.headline}" for _, s in summaries if s.headline)
        prompt = (
            f"Write a 45-word agenda naming what is coming up:\n{headlines}\n"
            f"Spoken, flowing prose. Not a list read aloud."
            f"{' Write in Hindi.' if language == 'hi' else ''}"
        )
        return self._simple_segment(prompt, "agenda", "Agenda", 250)

    def _write_sign_off(self, position: int, language: str, style: str) -> Segment:
        prompt = (
            "Write a 20-word sign-off for a daily technical brief. Warm, brief, no "
            "call to action, no 'like and subscribe'."
            + (" Write in Hindi." if language == "hi" else "")
        )
        return self._simple_segment(prompt, "sign_off", "Sign-off", 150)

    def _simple_segment(self, prompt: str, kind: str, heading: str, max_tokens: int) -> Segment:
        try:
            completion = self.client.complete(
                prompt=prompt,
                stage=Stage.script,
                tier=Tier.flagship,
                system=SEGMENT_SYSTEM,
                max_tokens=max_tokens,
                cache_system=True,
                fallback_tier=Tier.utility,
            )
            text = completion.text.strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("brief.simple_segment_failed", kind=kind, error=str(exc))
            text = ""
        return Segment(position=0, kind=kind, heading=heading, script=text)

    @staticmethod
    def _title(state: BriefState, segments: list[Segment]) -> str:
        stories = [s for s in segments if s.kind == "story"]
        date = datetime.now(timezone.utc).strftime("%d %b %Y")
        if not stories:
            return f"VoiceBrief — {date}"
        return f"{stories[0].heading} — and {len(stories) - 1} more · {date}"

    # ── entry point ───────────────────────────────────────────────────────────
    def run(
        self,
        candidates: list,
        *,
        profile: StackProfileData | None = None,
        declared_topics: set[str] | None = None,
        language: str = "en",
        style: str = "solo_anchor",
    ) -> dict:
        result = self.graph.invoke(
            {
                "candidates": candidates,
                "profile": profile,
                "declared_topics": declared_topics or set(),
                "language": language,
                "style": style,
                "errors": [],
            }
        )
        episode = result.get("episode", {})
        episode["errors"] = result.get("errors", [])
        episode["cost_inr"] = round(self.client.run_cost_inr, 4)
        return episode
