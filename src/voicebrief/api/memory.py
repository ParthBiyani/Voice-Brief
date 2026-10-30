"""Episode memory: semantic search and segment Q&A.

The PRD's job-to-be-done #3 — "you mentioned something about memory architectures last
week, find it" — is only answerable if every segment was indexed when the episode was
made. So indexing happens at generation time, not at query time, and the search here
is a read against that index.

Segment Q&A is scoped deliberately tightly: a question about a segment is answered
only from that segment's sources. Widening it to the whole corpus would let the model
answer confidently from an article the listener never heard, which is exactly the
failure the citations exist to prevent.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    SearchHit,
    SearchResponse,
)
from voicebrief.db import get_session
from voicebrief.db.models import Episode, Item, Segment
from voicebrief.history import record_interaction, timed
from voicebrief.llm.base import Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger
from voicebrief.pipeline.embedding import get_embedding_service, item_text
from voicebrief.pipeline.vectorstore import get_vector_store

log = get_logger(__name__)

router = APIRouter(tags=["memory"])

# One-tap prompts from the PRD's interactive layer.
PRESETS = {
    "explain_simply": "Explain this as if I have no background in the area.",
    "show_code": "Show what this looks like in code, if the sources support it.",
    "compare_with": "How does this compare with the alternatives mentioned?",
}

QA_SYSTEM = """\
You answer a listener's question about one segment of an audio brief they just heard.

You are given that segment's script and the sources behind it. Answer only from those.

- If the sources do not contain the answer, say so plainly and stop. Do not fill the
  gap from general knowledge — the listener cannot tell the difference, and the whole
  point of the citations is that they can check.
- Be direct and short. This is a follow-up, not an essay.
- Never invent a URL, version number, or benchmark figure.\
"""


MAX_SOURCE_CHARS = 1500


def _source_text(session: Session, citations: list[dict]) -> str:
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


def index_episode_segments(session: Session, episode_id: uuid.UUID) -> int:
    """Embed an episode's segments into the memory collection.

    Called at generation time. Append-only and never pruned — this collection *is* the
    "what did you mention last week" feature.
    """
    episode = session.get(Episode, episode_id)
    if episode is None:
        return 0

    segments = [s for s in episode.segments if s.kind == "story" and s.script.strip()]
    if not segments:
        return 0

    service = get_embedding_service()
    vectors = service.encode([item_text(s.heading or "", s.script) for s in segments])

    store = get_vector_store()
    store.ensure_collections()
    store.upsert_segments(
        [s.id for s in segments],
        vectors,
        [
            {
                "episode_id": str(episode.id),
                "user_id": str(episode.user_id),
                "episode_title": episode.title or "",
                "heading": s.heading or "",
                "script": s.script[:2000],
                "start_seconds": s.start_seconds or 0.0,
                "citations": s.citations or [],
                "created_at": episode.created_at.isoformat(),
            }
            for s in segments
        ],
    )
    log.info("memory.indexed", episode=str(episode.id), segments=len(segments))
    return len(segments)


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, description="Natural-language query"),
    limit: int = Query(8, ge=1, le=30),
    session: Session = Depends(get_session),
) -> SearchResponse:
    """Semantic search across every past episode."""
    with timed() as clock:
        service = get_embedding_service()
        vector = service.encode_one(q)

        try:
            neighbours = get_vector_store().search_memory(vector, limit=limit)
        except Exception as exc:
            log.warning("memory.search_failed", error=str(exc))
            return SearchResponse(query=q, hits=[])

    hits = []
    for neighbour in neighbours:
        payload = neighbour.payload
        start = float(payload.get("start_seconds", 0.0))
        script = payload.get("script", "")
        hits.append(
            SearchHit(
                segment_id=uuid.UUID(neighbour.id),
                episode_id=uuid.UUID(payload["episode_id"]),
                episode_title=payload.get("episode_title"),
                heading=payload.get("heading"),
                excerpt=script[:280] + ("…" if len(script) > 280 else ""),
                timestamp=f"{int(start) // 60:02d}:{int(start) % 60:02d}",
                start_seconds=start,
                score=round(neighbour.score, 4),
                citations=[Citation(**c) for c in payload.get("citations", [])],
            )
        )
    record_interaction(
        "search",
        "web",
        request={"q": q, "limit": limit},
        response={
            "hits": [
                {"segment_id": str(h.segment_id), "heading": h.heading, "score": h.score}
                for h in hits
            ]
        },
        latency_ms=clock.get("ms"),
    )
    return SearchResponse(query=q, hits=hits)


@router.post("/segments/{segment_id}/chat", response_model=ChatResponse)
def segment_chat(
    segment_id: uuid.UUID,
    request: ChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    """Ask about one segment, answered only from that segment's sources."""
    segment = session.get(Segment, segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail="segment not found")

    question = PRESETS.get(request.preset or "", request.question).strip()
    if not question:
        raise HTTPException(status_code=422, detail="question or preset is required")

    citations = segment.citations or []
    sources = _source_text(session, citations)

    prompt = (
        f"SEGMENT THE LISTENER HEARD:\n{segment.script}\n\n"
        f"SOURCES BEHIND IT:\n{sources}\n\n"
        f"LISTENER'S QUESTION: {question}"
    )

    with timed() as clock:
        client = LLMClient(session, episode_id=segment.episode_id)
        completion = client.complete(
            prompt=prompt,
            stage=Stage.segment_qa,
            tier=Tier.utility,
            system=QA_SYSTEM,
            max_tokens=800,
            cache_system=True,
        )

    answer = completion.text.strip()
    record_interaction(
        "qa",
        "web",
        request={"question": question, "preset": request.preset},
        response={
            "answer": answer,
            "sources": [c.get("url") for c in citations],
            "cost_inr": round(client.run_cost_inr, 4),
        },
        episode_id=segment.episode_id,
        segment_id=segment_id,
        latency_ms=clock.get("ms"),
    )
    return ChatResponse(
        answer=answer,
        segment_id=segment_id,
        citations=[Citation(**c) for c in citations],
        cost_inr=round(client.run_cost_inr, 4),
    )
