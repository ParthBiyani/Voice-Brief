"""Cluster summarization.

One LLM call turns a cluster of related items into a single grounded story summary
that the script generator can work from.

Two design decisions carry the cost story:

* **Only clusters that survive pre-ranking are summarized.** Summarizing all ~60
  costs ₹7.5 of an ₹8 budget for work that ~45 of them will never use.
* **The system prompt is cached.** It is identical across every call in a run, so
  caching drops its input cost to ~10%.

The output is deliberately constrained: facts only, with the source each fact came
from. The script stage cannot cite what the summary stage did not record, so
grounding is enforced here rather than hoped for later.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from voicebrief.llm.base import Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger

log = get_logger(__name__)

SUMMARY_SYSTEM = """\
You compress a group of related tech news items into one factual brief.

Rules:
- State only what the sources state. If a number, version, or name is not in the
  sources, it does not go in the summary.
- Prefer concrete specifics: version numbers, benchmark deltas, API changes, what
  breaks. Avoid "revolutionary", "game-changing", and every other adjective that
  survives being deleted.
- If the sources disagree, say so rather than picking one.
- `what_changed` is the single sentence a busy engineer needs.
- `why_it_matters` is the practical consequence, or an empty string if there is no
  honest one. An empty string is a valid and often correct answer.
- `key_facts` are standalone claims, each traceable to one source index.

Return JSON only.\
"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "what_changed": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "key_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "source_index": {"type": "integer"},
                },
                "required": ["fact", "source_index"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["headline", "what_changed", "why_it_matters", "key_facts"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class SourceRef:
    item_id: uuid.UUID
    title: str
    url: str
    source_slug: str


@dataclass(slots=True)
class ClusterSummary:
    cluster_id: uuid.UUID
    headline: str
    what_changed: str
    why_it_matters: str
    # [{"fact": str, "url": str, "title": str, "item_id": str}]
    key_facts: list[dict] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)

    @property
    def citation_urls(self) -> set[str]:
        return {s.url for s in self.sources}

    def as_context(self) -> str:
        """Render for the script generator's prompt."""
        facts = "\n".join(f"  - {f['fact']} [{f['url']}]" for f in self.key_facts)
        return (
            f"HEADLINE: {self.headline}\n"
            f"WHAT CHANGED: {self.what_changed}\n"
            f"WHY IT MATTERS: {self.why_it_matters or '(no clear practical consequence)'}\n"
            f"FACTS:\n{facts}\n"
            f"SOURCES: {', '.join(s.url for s in self.sources)}"
        )


def _render_sources(sources: list[SourceRef], bodies: dict[uuid.UUID, str]) -> str:
    lines = []
    for index, source in enumerate(sources):
        body = (bodies.get(source.item_id) or "")[:1200]
        lines.append(
            f"[{index}] {source.title}\n"
            f"    from: {source.source_slug} | {source.url}\n"
            f"    {body}"
        )
    return "\n\n".join(lines)


def summarize_cluster(
    client: LLMClient,
    *,
    cluster_id: uuid.UUID,
    sources: list[SourceRef],
    bodies: dict[uuid.UUID, str],
) -> ClusterSummary | None:
    """Summarize one cluster. Returns None if the model call fails.

    A failed summary drops one story rather than the episode — the caller simply
    moves to the next-ranked cluster.
    """
    if not sources:
        return None

    prompt = (
        f"Sources covering one story:\n\n{_render_sources(sources, bodies)}\n\n"
        f"Produce the brief. Cite each fact by its source index."
    )

    try:
        completion = client.complete(
            prompt=prompt,
            stage=Stage.cluster_summary,
            tier=Tier.utility,
            system=SUMMARY_SYSTEM,
            schema=SUMMARY_SCHEMA,
            max_tokens=1200,
            cache_system=True,
        )
    except Exception as exc:  # noqa: BLE001 — one story, not the episode
        log.warning("summarize.failed", cluster=str(cluster_id), error=str(exc))
        return None

    payload = completion.parsed or {}

    # Resolve source indices to real URLs here. If the model invents an index, the
    # fact is dropped rather than carried forward with a fabricated citation — this
    # is the mechanism behind the "zero hallucinated links" target.
    facts = []
    dropped = 0
    for entry in payload.get("key_facts", []):
        index = entry.get("source_index")
        if not isinstance(index, int) or not 0 <= index < len(sources):
            dropped += 1
            continue
        source = sources[index]
        facts.append(
            {
                "fact": str(entry.get("fact", "")).strip(),
                "url": source.url,
                "title": source.title,
                "item_id": str(source.item_id),
            }
        )

    if dropped:
        log.warning("summarize.dropped_uncited_facts", cluster=str(cluster_id), count=dropped)

    return ClusterSummary(
        cluster_id=cluster_id,
        headline=str(payload.get("headline", "")).strip(),
        what_changed=str(payload.get("what_changed", "")).strip(),
        why_it_matters=str(payload.get("why_it_matters", "")).strip(),
        key_facts=[f for f in facts if f["fact"]],
        sources=sources,
    )
