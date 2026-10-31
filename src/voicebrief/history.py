"""History and audit trail.

Everything the system does and everything it is asked is recorded here, so it can be
improved from evidence rather than from memory: why a story was picked or skipped, what
a prompt actually said, what answer a person got, how a run went.

Two rules shape the module:

* **Recording must never break the thing being recorded.** Every writer swallows and
  logs its own failures. A broken history table costs history, not the morning's brief.
* **Writers use their own short transactions.** A run that fails and rolls back still
  leaves its record, which is exactly when the record matters most.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any, cast
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db import session_scope
from voicebrief.db.models import (
    CostEntry,
    Episode,
    EpisodeStatus,
    Feedback,
    Interaction,
    RankingSnapshot,
    RunLog,
    Segment,
)
from voicebrief.logging import get_logger

log = get_logger(__name__)

# Responses can be long; keep what a human reviewing history actually needs.
MAX_TEXT = 8000


def _clip(value: Any, limit: int = MAX_TEXT) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"… [{len(value) - limit} more characters]"
    return value


def _jsonable(value: Any) -> Any:
    """Make an arbitrary request/response payload safe for a JSONB column."""
    return json.loads(json.dumps(value, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────────────
def start_run(
    kind: str, trigger: str = "cli", *, user_id: uuid.UUID | None = None
) -> uuid.UUID | None:
    try:
        with session_scope() as session:
            run = RunLog(
                kind=kind,
                trigger=trigger,
                status="running",
                started_at=datetime.now(UTC),
                user_id=user_id,
            )
            session.add(run)
            session.flush()
            return run.id
    except Exception as exc:
        log.warning("history.start_run_failed", error=str(exc))
        return None


def finish_run(
    run_id: uuid.UUID | None,
    status: str,
    *,
    summary: dict | None = None,
    episode_id: uuid.UUID | None = None,
    error: str | None = None,
) -> None:
    if run_id is None:
        return
    try:
        with session_scope() as session:
            run = session.get(RunLog, run_id)
            if run is None:
                return
            run.status = status
            run.finished_at = datetime.now(UTC)
            run.summary = _jsonable(summary or {})
            run.episode_id = episode_id
            run.error = _clip(error, 2000) if error else None
    except Exception as exc:
        log.warning("history.finish_run_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Crash recovery
# ─────────────────────────────────────────────────────────────────────────────
INTERRUPTED_AFTER = timedelta(hours=1)
INTERRUPTED_MESSAGE = (
    "interrupted: the process ended before this finished (crash, power loss or restart)"
)


def _updated(result: object) -> int:
    """Rows an UPDATE touched; `Session.execute` is typed as the generic Result."""
    return cast(CursorResult, result).rowcount or 0


def reap_interrupted(*, older_than: timedelta = INTERRUPTED_AFTER) -> tuple[int, int]:
    """Close out runs and episodes a crash left in `running` forever.

    A laptop that freezes mid-brief leaves its run and its episode marked running, which
    reads as "still going" indefinitely and hides that the morning's brief never
    happened. A generation takes minutes, so anything older than the cutoff is dead.
    Returns (runs, episodes) closed. Never raises.
    """
    cutoff = datetime.now(UTC) - older_than
    try:
        with session_scope() as session:
            runs = _updated(
                session.execute(
                    update(RunLog)
                    .where(RunLog.status == "running", RunLog.started_at < cutoff)
                    .values(
                        status="interrupted",
                        finished_at=datetime.now(UTC),
                        error=INTERRUPTED_MESSAGE,
                    )
                )
            )
            episodes = _updated(
                session.execute(
                    update(Episode)
                    .where(
                        Episode.status.in_([EpisodeStatus.running, EpisodeStatus.pending]),
                        Episode.created_at < cutoff,
                    )
                    .values(status=EpisodeStatus.failed, error=INTERRUPTED_MESSAGE)
                )
            )
        if runs or episodes:
            log.warning("history.reaped_interrupted", runs=runs, episodes=episodes)
        return runs, episodes
    except Exception as exc:
        log.warning("history.reap_failed", error=str(exc))
        return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Ranking decisions
# ─────────────────────────────────────────────────────────────────────────────
def record_ranking(
    session: Session,
    episode_id: uuid.UUID,
    trace: list[dict],
    excluded: list[dict] | None = None,
) -> int:
    """Store every candidate considered for an episode and its fate.

    Written on the episode's own session so a snapshot can never exist without its
    episode. Failure is logged rather than raised for the same reason as the rest of
    the module.
    """
    try:
        count = 0
        for entry in trace:
            session.add(
                RankingSnapshot(
                    episode_id=episode_id,
                    cluster_id=entry.get("cluster_id"),
                    outcome="picked" if entry.get("picked") else "not_picked",
                    position=entry.get("rank"),
                    title=_clip(entry.get("title") or "(untitled)", 500),
                    url=entry.get("url"),
                    source_slug=entry.get("source_slug"),
                    cluster_size=entry.get("cluster_size"),
                    heuristic_score=entry.get("heuristic_score"),
                    llm_score=entry.get("llm_score"),
                    final_score=entry.get("score"),
                    signals=_jsonable(entry.get("signals") or {}),
                    matched_dependencies=_jsonable(entry.get("matched_dependencies") or []),
                    rationale=_clip(entry.get("rationale"), 1000),
                )
            )
            count += 1
        for entry in excluded or []:
            session.add(
                RankingSnapshot(
                    episode_id=episode_id,
                    cluster_id=entry.get("cluster_id"),
                    outcome="aired",
                    title=_clip(entry.get("title") or "(untitled)", 500),
                    url=entry.get("url"),
                    cluster_size=entry.get("cluster_size"),
                    signals={},
                    matched_dependencies=[],
                    rationale="already covered in an earlier episode",
                )
            )
            count += 1
        session.flush()
        return count
    except Exception as exc:
        log.warning("history.record_ranking_failed", error=str(exc))
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Interactions
# ─────────────────────────────────────────────────────────────────────────────
def record_interaction(
    kind: str,
    channel: str,
    *,
    user_id: uuid.UUID | None = None,
    request: dict | None = None,
    response: dict | None = None,
    episode_id: uuid.UUID | None = None,
    segment_id: uuid.UUID | None = None,
    latency_ms: int | None = None,
) -> None:
    """Record something a person asked or did, and what came back."""
    try:
        with session_scope() as session:
            session.add(
                Interaction(
                    user_id=user_id,
                    channel=channel,
                    kind=kind,
                    request=_jsonable(request or {}),
                    response=_jsonable(response or {}),
                    episode_id=episode_id,
                    segment_id=segment_id,
                    latency_ms=latency_ms,
                )
            )
    except Exception as exc:
        log.warning("history.record_interaction_failed", kind=kind, error=str(exc))


@contextmanager
def timed() -> Iterator[dict]:
    """`with timed() as t: ...` then `t["ms"]` — for interaction latency."""
    box: dict = {}
    started = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = int((time.perf_counter() - started) * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Retention
# ─────────────────────────────────────────────────────────────────────────────
def prune_llm_bodies(session: Session, *, days: int | None = None) -> int:
    """Drop old prompt/response text. Cost, tokens and timing are kept forever."""
    days = get_settings().history_retention_days if days is None else days
    cutoff = datetime.now(UTC) - timedelta(days=days)
    result = session.execute(
        update(CostEntry)
        .where(
            CostEntry.created_at < cutoff,
            (CostEntry.prompt.is_not(None))
            | (CostEntry.response.is_not(None))
            | (CostEntry.system_prompt.is_not(None)),
        )
        .values(prompt=None, response=None, system_prompt=None)
    )
    return _updated(result)


# ─────────────────────────────────────────────────────────────────────────────
# Reading it back
# ─────────────────────────────────────────────────────────────────────────────
def _day_bounds(day: date, timezone: str) -> tuple[datetime, datetime]:
    tz: tzinfo
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = UTC
    start = datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(UTC)
    return start, start + timedelta(days=1)


def timeline(session: Session, day: date, *, timezone: str | None = None) -> dict:
    """Everything that happened on one local day, shaped for the CLI and the web app."""
    start, end = _day_bounds(day, timezone or get_settings().timezone)

    runs = (
        session.execute(
            select(RunLog)
            .where(RunLog.started_at >= start, RunLog.started_at < end)
            .order_by(RunLog.started_at)
        )
        .scalars()
        .all()
    )

    episodes = (
        session.execute(
            select(Episode)
            .where(Episode.created_at >= start, Episode.created_at < end)
            .order_by(Episode.created_at)
        )
        .scalars()
        .all()
    )

    interactions = (
        session.execute(
            select(Interaction)
            .where(Interaction.created_at >= start, Interaction.created_at < end)
            .order_by(Interaction.created_at)
        )
        .scalars()
        .all()
    )

    episode_payload = []
    for episode in episodes:
        snapshots = (
            session.execute(
                select(RankingSnapshot)
                .where(RankingSnapshot.episode_id == episode.id)
                .order_by(RankingSnapshot.position.nulls_last(), RankingSnapshot.created_at)
            )
            .scalars()
            .all()
        )
        calls = (
            session.execute(
                select(CostEntry)
                .where(CostEntry.episode_id == episode.id)
                .order_by(CostEntry.created_at)
            )
            .scalars()
            .all()
        )
        episode_payload.append(
            {
                "id": str(episode.id),
                "title": episode.title,
                "status": episode.status.value,
                "error": episode.error,
                "duration_seconds": episode.duration_seconds,
                "cost_inr": float(episode.cost_inr or 0),
                "created_at": episode.created_at.isoformat(),
                "grounding": {
                    k: v
                    for k, v in (episode.grounding or {}).items()
                    if k in ("attribution_rate", "sentences", "supported", "warnings", "tts_engine")
                },
                "ranking": [
                    {
                        "outcome": s.outcome,
                        "title": s.title,
                        "source": s.source_slug,
                        "score": s.final_score,
                        "heuristic": s.heuristic_score,
                        "llm": s.llm_score,
                        "signals": s.signals,
                        "stack_match": s.matched_dependencies,
                        "why": s.rationale,
                    }
                    for s in snapshots
                ],
                "llm_calls": [
                    {
                        "stage": c.stage,
                        "model": c.model,
                        "tokens": c.input_tokens + c.output_tokens,
                        "cost_inr": float(c.cost_inr),
                        "latency_ms": c.latency_ms,
                        "stop_reason": c.stop_reason,
                    }
                    for c in calls
                ],
            }
        )

    return {
        "date": day.isoformat(),
        "runs": [
            {
                "id": str(r.id),
                "kind": r.kind,
                "trigger": r.trigger,
                "status": r.status,
                "started_at": r.started_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "episode_id": str(r.episode_id) if r.episode_id else None,
                "summary": r.summary,
                "error": r.error,
            }
            for r in runs
        ],
        "episodes": episode_payload,
        "interactions": [
            {
                "id": str(i.id),
                "kind": i.kind,
                "channel": i.channel,
                "at": i.created_at.isoformat(),
                "request": i.request,
                "response": i.response,
                "episode_id": str(i.episode_id) if i.episode_id else None,
                "latency_ms": i.latency_ms,
            }
            for i in interactions
        ],
    }


def export_records(session: Session, since: datetime) -> Iterator[dict]:
    """Every history record since `since`, one JSON-able dict each, typed.

    The shape is deliberately flat so it loads straight into a notebook or a dataframe:
    ranking snapshots joined with thumbs make labelled examples of what was relevant.
    """

    def stamp(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    for run in session.execute(select(RunLog).where(RunLog.started_at >= since)).scalars():
        yield {
            "type": "run",
            "id": str(run.id),
            "kind": run.kind,
            "trigger": run.trigger,
            "status": run.status,
            "started_at": stamp(run.started_at),
            "finished_at": stamp(run.finished_at),
            "episode_id": str(run.episode_id) if run.episode_id else None,
            "summary": run.summary,
            "error": run.error,
        }

    for snap in session.execute(
        select(RankingSnapshot).where(RankingSnapshot.created_at >= since)
    ).scalars():
        yield {
            "type": "ranking",
            "episode_id": str(snap.episode_id),
            "outcome": snap.outcome,
            "position": snap.position,
            "title": snap.title,
            "url": snap.url,
            "source": snap.source_slug,
            "cluster_size": snap.cluster_size,
            "heuristic_score": snap.heuristic_score,
            "llm_score": snap.llm_score,
            "final_score": snap.final_score,
            "signals": snap.signals,
            "stack_match": snap.matched_dependencies,
            "rationale": snap.rationale,
            "at": stamp(snap.created_at),
        }

    for call in session.execute(select(CostEntry).where(CostEntry.created_at >= since)).scalars():
        yield {
            "type": "llm_call",
            "episode_id": str(call.episode_id) if call.episode_id else None,
            "stage": call.stage,
            "provider": call.provider,
            "model": call.model,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cached_tokens": call.cached_tokens,
            "cost_inr": float(call.cost_inr),
            "latency_ms": call.latency_ms,
            "stop_reason": call.stop_reason,
            "system_prompt": call.system_prompt,
            "prompt": call.prompt,
            "response": call.response,
            "at": stamp(call.created_at),
        }

    for item in session.execute(
        select(Interaction).where(Interaction.created_at >= since)
    ).scalars():
        yield {
            "type": "interaction",
            "kind": item.kind,
            "channel": item.channel,
            "request": item.request,
            "response": item.response,
            "episode_id": str(item.episode_id) if item.episode_id else None,
            "segment_id": str(item.segment_id) if item.segment_id else None,
            "latency_ms": item.latency_ms,
            "at": stamp(item.created_at),
        }

    for vote in session.execute(select(Feedback).where(Feedback.created_at >= since)).scalars():
        segment = session.get(Segment, vote.segment_id)
        yield {
            "type": "feedback",
            "vote": vote.vote,
            "note": vote.note,
            "segment_id": str(vote.segment_id),
            "heading": segment.heading if segment else None,
            "episode_id": str(segment.episode_id) if segment else None,
            "cluster_id": str(segment.cluster_id) if segment and segment.cluster_id else None,
            "at": stamp(vote.created_at),
        }
