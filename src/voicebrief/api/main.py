"""FastAPI application.

Four route groups, matching the PRD's architecture sketch: /ingest, /episodes, /chat,
/search. Generation runs in a background task with progress streamed over SSE, because
an episode takes minutes and a request that blocks that long dies to a proxy timeout
long before it finishes.

Scheduling is a cron hit to `/ingest/run` guarded by a Postgres advisory lock. That is
the whole scheduler, deliberately — Celery and Temporal are v2 (PRD §8).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from voicebrief import __version__
from voicebrief.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    EpisodeDetail,
    EpisodeSummary,
    FeedbackIn,
    GenerateAccepted,
    GenerateRequest,
    HealthResponse,
    IngestStats,
    SegmentOut,
    TranscriptLine,
)
from voicebrief.config import get_settings
from voicebrief.db import get_session, session_scope
from voicebrief.db.models import Episode, Feedback, Language, Segment, User
from voicebrief.logging import configure_logging, get_logger
from voicebrief.pipeline.episodes import presigned_url

log = get_logger(__name__)

# Per-episode progress channels for SSE. In-process because v1 runs as a single
# instance; a second replica would need Redis pub/sub, which is a v2 concern and a
# one-file change to this module.
_progress: dict[uuid.UUID, asyncio.Queue] = defaultdict(asyncio.Queue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("api.start", env=settings.env, provider=settings.llm_provider)
    yield
    log.info("api.stop")


app = FastAPI(
    title="VoiceBrief",
    version=__version__,
    description="A personalized daily audio brief that knows what you're building.",
    lifespan=lifespan,
)

from voicebrief.api.memory import router as memory_router  # noqa: E402

app.include_router(memory_router)

app.add_middleware(
    CORSMiddleware,
    # The web app is served from a different origin in development.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(session: Session = Depends(get_session)) -> HealthResponse:
    settings = get_settings()

    postgres_ok = True
    try:
        session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 — health must report, never raise
        postgres_ok = False

    qdrant_ok = True
    try:
        from voicebrief.pipeline.vectorstore import get_vector_store

        get_vector_store().client.get_collections()
    except Exception:  # noqa: BLE001
        qdrant_ok = False

    return HealthResponse(
        status="ok" if postgres_ok else "degraded",
        version=__version__,
        postgres=postgres_ok,
        qdrant=qdrant_ok,
        llm_provider=settings.llm_provider,
        tts_engine=settings.tts_engine,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ingest/run", response_model=list[IngestStats], tags=["ingest"])
async def ingest_run(force: bool = False) -> list[IngestStats]:
    """Run one ingestion pass. This is the cron target.

    Concurrent calls are a no-op rather than a duplicate crawl — the advisory lock
    inside `run_ingest` makes overlapping cron hits safe.
    """
    from voicebrief.pipeline.ingest import run_ingest

    stats = await run_ingest(force=force)
    return [
        IngestStats(
            slug=s.slug, fetched=s.fetched, inserted=s.inserted,
            skipped=s.skipped, ok=s.ok, error=s.error,
        )
        for s in stats
    ]


@app.post("/ingest/enrich", tags=["ingest"])
def ingest_enrich(
    keep: int = Query(300, ge=10, le=2000),
    session: Session = Depends(get_session),
) -> dict:
    """Embed, deduplicate and cluster the recent crawl."""
    from dataclasses import asdict

    from voicebrief.pipeline.enrich import enrich

    return asdict(enrich(session, keep=keep))


# ─────────────────────────────────────────────────────────────────────────────
# Episodes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/episodes", response_model=list[EpisodeSummary], tags=["episodes"])
def list_episodes(
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[EpisodeSummary]:
    rows = session.execute(
        select(Episode).order_by(Episode.created_at.desc()).limit(limit)
    ).scalars()
    return [_to_summary(e) for e in rows]


@app.get("/episodes/{episode_id}", response_model=EpisodeDetail, tags=["episodes"])
def get_episode(
    episode_id: uuid.UUID, session: Session = Depends(get_session)
) -> EpisodeDetail:
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="episode not found")

    return EpisodeDetail(
        **_to_summary(episode).model_dump(),
        segments=[_to_segment(s) for s in episode.segments],
        audio_url=presigned_url(episode.audio_key) if episode.audio_key else None,
        generation_seconds=episode.generation_seconds,
        error=episode.error,
    )


@app.get(
    "/episodes/{episode_id}/transcript",
    response_model=list[TranscriptLine],
    tags=["episodes"],
)
def get_transcript(
    episode_id: uuid.UUID, session: Session = Depends(get_session)
) -> list[TranscriptLine]:
    """Timestamped transcript. The timestamps are measured from rendered audio, so
    seeking to one lands on the right words."""
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="episode not found")

    lines = []
    for segment in episode.segments:
        start = segment.start_seconds or 0.0
        lines.append(
            TranscriptLine(
                position=segment.position,
                timestamp=f"{int(start) // 60:02d}:{int(start) % 60:02d}",
                heading=segment.heading,
                text=segment.script,
                start_seconds=start,
                citations=[Citation(**c) for c in (segment.citations or [])],
            )
        )
    return lines


@app.post(
    "/episodes/generate",
    response_model=GenerateAccepted,
    status_code=202,
    tags=["episodes"],
)
def generate(
    request: GenerateRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> GenerateAccepted:
    """Start generation and return immediately.

    Generation takes minutes; holding the request open for that long dies to a proxy
    timeout. Progress is streamed from the returned `stream_url`.
    """
    user = session.execute(
        select(User).where(User.email == request.email)
    ).scalar_one_or_none()
    if user is None:
        user = User(
            email=request.email,
            language=Language(request.language),
            github_login=request.github_login,
            topics=request.topics,
        )
        session.add(user)
        session.flush()

    episode_id = uuid.uuid4()
    background.add_task(_run_generation, episode_id, user.id, request)

    return GenerateAccepted(
        episode_id=episode_id,
        status="accepted",
        stream_url=f"/episodes/{episode_id}/stream",
    )


@app.get("/episodes/{episode_id}/stream", tags=["episodes"])
async def stream_progress(episode_id: uuid.UUID) -> StreamingResponse:
    """Server-sent events for one generation run."""

    async def event_source() -> AsyncIterator[str]:
        queue = _progress[episode_id]
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    # Keep-alive: proxies close an idle SSE connection.
                    yield ": keep-alive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"
                if event.get("stage") in ("ready", "failed"):
                    break
        finally:
            _progress.pop(episode_id, None)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _emit(episode_id: uuid.UUID, stage: str, **fields) -> None:
    event = {"stage": stage, "at": datetime.now(timezone.utc).isoformat(), **fields}
    queue = _progress[episode_id]
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:  # pragma: no cover - unbounded by default
        pass


def _run_generation(episode_id: uuid.UUID, user_id: uuid.UUID, request: GenerateRequest) -> None:
    """Background worker. Reports progress and never lets an exception escape."""
    from voicebrief.pipeline.candidates import build_candidates
    from voicebrief.pipeline.episodes import generate_episode

    try:
        _emit(episode_id, "started")
        profile = None
        if request.github_login:
            from voicebrief.personalization.github_profile import GitHubProfileBuilder

            _emit(episode_id, "profiling", login=request.github_login)
            profile = asyncio.run(GitHubProfileBuilder().build(request.github_login))

        with session_scope() as session:
            _emit(episode_id, "selecting")
            candidates = build_candidates(session)
            if not candidates:
                _emit(episode_id, "failed", error="no clusters available; run ingest first")
                return

            _emit(episode_id, "writing", candidates=len(candidates))
            result = generate_episode(
                session,
                user_id=user_id,
                candidates=candidates,
                profile=profile,
                declared_topics=set(request.topics),
                language=request.language,
                style=request.style,
                max_stories=request.max_stories,
                target_minutes=request.target_minutes,
                render_audio=request.render_audio,
            )

        try:
            with session_scope() as session:
                from voicebrief.api.memory import index_episode_segments

                index_episode_segments(session, result.episode_id)
        except Exception as exc:  # noqa: BLE001 — a searchable episode is a bonus,
            # not a precondition for having generated one.
            log.warning("api.memory_index_failed", error=str(exc))

        _emit(
            episode_id,
            "ready",
            episode_id=str(result.episode_id),
            title=result.title,
            duration_seconds=result.duration_seconds,
            cost_inr=result.cost_inr,
            attribution_rate=result.attribution_rate,
        )
    except Exception as exc:  # noqa: BLE001 — a background task must not die silently
        log.error("api.generation_failed", error=str(exc))
        _emit(episode_id, "failed", error=f"{type(exc).__name__}: {exc}"[:300])


# ─────────────────────────────────────────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/feedback", status_code=204, tags=["feedback"])
def submit_feedback(
    payload: FeedbackIn,
    email: str = Query("me@example.com"),
    session: Session = Depends(get_session),
) -> None:
    """Thumbs per segment. Replayed into the ranking prompt as few-shot examples."""
    if payload.vote not in (-1, 1):
        raise HTTPException(status_code=422, detail="vote must be 1 or -1")

    user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    if session.get(Segment, payload.segment_id) is None:
        raise HTTPException(status_code=404, detail="segment not found")

    existing = session.execute(
        select(Feedback).where(
            Feedback.user_id == user.id, Feedback.segment_id == payload.segment_id
        )
    ).scalar_one_or_none()

    if existing:
        existing.vote = payload.vote
        existing.note = payload.note
    else:
        session.add(
            Feedback(
                user_id=user.id,
                segment_id=payload.segment_id,
                vote=payload.vote,
                note=payload.note,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_summary(episode: Episode) -> EpisodeSummary:
    return EpisodeSummary(
        id=episode.id,
        title=episode.title,
        status=episode.status.value,
        mode=episode.mode.value,
        language=episode.language.value,
        style=episode.style.value,
        duration_seconds=episode.duration_seconds,
        cost_inr=float(episode.cost_inr) if episode.cost_inr is not None else None,
        created_at=episode.created_at,
    )


def _to_segment(segment: Segment) -> SegmentOut:
    return SegmentOut(
        id=segment.id,
        position=segment.position,
        kind=segment.kind,
        heading=segment.heading,
        script=segment.script,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        citations=[Citation(**c) for c in (segment.citations or [])],
    )
