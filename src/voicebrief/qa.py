"""Questions about an episode, answered from what the listener heard and its sources.

Two scopes share one rule (answer only from the material, say so when it is not there):

- a segment (the web player's "Ask about this"): that segment and its sources;
- a whole episode (a reply in the Slack thread, where nobody picks a segment): every
  segment's script, plus the sources of the segments the question is most about.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import Episode, Item, Segment
from voicebrief.history import record_interaction, timed
from voicebrief.llm.base import Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger

log = get_logger(__name__)

MAX_SOURCE_CHARS = 1500
# Sources of this many best-matching segments go into an episode question. Six stories
# with full sources would be ~20k characters for a question that is usually about one.
EPISODE_SOURCE_SEGMENTS = 3

QA_SYSTEM = """\
You answer a listener's question about one segment of an audio brief they just heard.

You are given that segment's script and the sources behind it. Answer only from those.

- If the sources do not contain the answer, say so plainly and stop. Do not fill the
  gap from general knowledge — the listener cannot tell the difference, and the whole
  point of the citations is that they can check.
- Be direct and short. This is a follow-up, not an essay.
- Never invent a URL, version number, or benchmark figure.\
"""

EPISODE_QA_SYSTEM = """\
You answer a listener's question about an audio brief they received today.

You are given every segment of the brief and the sources behind the segments the
question seems to be about. Answer only from those.

- If the material does not contain the answer, say so plainly and stop. Do not fill the
  gap from general knowledge — the listener cannot tell the difference, and the whole
  point of the citations is that they can check.
- Name the story you are drawing on, so the listener knows which one you mean.
- Be direct and short; this is a chat reply. Plain sentences, no headings.
- Never invent a URL, version number, or benchmark figure.\
"""


@dataclass
class Answer:
    text: str
    citations: list[dict]
    cost_inr: float


def source_text(session: Session, citations: list[dict]) -> str:
    """The sources behind a segment, with their text.

    The prompt used to list only titles and URLs, so the model could answer from nothing
    but the segment's own script while the UI claimed the answer came from the sources.
    """
    if not citations:
        return "(no sources recorded for this segment)"

    ids = []
    for citation in citations:
        try:
            ids.append(uuid.UUID(str(citation["item_id"])))
        except (KeyError, ValueError, TypeError):
            continue
    items = (
        {i.id: i for i in session.execute(select(Item).where(Item.id.in_(ids))).scalars()}
        if ids
        else {}
    )

    blocks = []
    for number, citation in enumerate(citations, start=1):
        head = f"[{number}] {citation.get('title') or 'source'} ({citation.get('url', '')})"
        try:
            item = items.get(uuid.UUID(str(citation.get("item_id"))))
        except (ValueError, TypeError):
            item = None
        text = ((item.body or item.summary or "") if item else "").strip()
        blocks.append(f"{head}\n{text[:MAX_SOURCE_CHARS]}" if text else head)
    return "\n\n".join(blocks)


def most_relevant(question: str, segments: list[Segment], k: int) -> list[Segment]:
    """The k story segments closest to the question, by embedding similarity.

    Falls back to the first k when the embedding model is unavailable: a slower or less
    focused answer beats none.
    """
    if len(segments) <= k:
        return segments
    try:
        from voicebrief.pipeline.embedding import get_embedding_service, item_text

        service = get_embedding_service()
        query = service.encode_one(question)
        vectors = service.encode([item_text(s.heading or "", s.script) for s in segments])
        order = sorted(range(len(segments)), key=lambda i: -float(vectors[i] @ query))
        return [segments[i] for i in order[:k]]
    except Exception as exc:
        log.warning("qa.ranking_unavailable", error=str(exc)[:200])
        return segments[:k]


def answer_about_episode(
    session: Session, episode_id: uuid.UUID, question: str, *, channel: str
) -> Answer:
    """Answer a free-form question about a whole episode, and record it in history."""
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise LookupError(f"episode {episode_id} not found")

    segments = sorted(episode.segments, key=lambda s: s.position)
    stories = [s for s in segments if s.kind == "story" and s.script.strip()]
    focus = most_relevant(question, stories, EPISODE_SOURCE_SEGMENTS)

    heard = "\n\n".join(
        f"## {s.heading or s.kind}\n{s.script}" for s in segments if s.script.strip()
    )
    sources = "\n\n".join(
        f"SOURCES FOR “{s.heading}”:\n{source_text(session, s.citations or [])}" for s in focus
    )
    prompt = (
        f"THE BRIEF ({episode.title}):\n{heard}\n\n{sources}\n\nLISTENER'S QUESTION: {question}"
    )

    with timed() as clock:
        client = LLMClient(session, episode_id=episode.id)
        completion = client.complete(
            prompt=prompt,
            stage=Stage.segment_qa,
            tier=Tier.utility,
            system=EPISODE_QA_SYSTEM,
            max_tokens=900,
            cache_system=True,
        )

    citations = [c for s in focus for c in (s.citations or [])]
    answer = Answer(
        text=completion.text.strip(),
        citations=citations,
        cost_inr=round(client.run_cost_inr, 4),
    )
    record_interaction(
        "qa",
        channel,
        request={"question": question, "scope": "episode"},
        response={
            "answer": answer.text,
            "sources": [c.get("url") for c in citations],
            "cost_inr": answer.cost_inr,
        },
        episode_id=episode.id,
        latency_ms=clock.get("ms"),
    )
    return answer
