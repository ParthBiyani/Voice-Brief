"""The morning loop as one command.

`run_daily` is what the scheduler fires and what a person types to "just get today's
brief". It is built to be fired carelessly: twice in a row, from a laptop that was
asleep at 07:00, while another run is still going.

* **Idempotent.** If today's brief (by the configured local timezone, not UTC) already
  exists in a usable state, it does nothing and says so. `force` overrides that.
* **Exclusive.** A Postgres advisory lock held on a dedicated connection means two
  overlapping runs cannot both generate (and both pay for) an episode. The lock is
  deliberately not taken on a work session: sessions commit mid-run, return their
  connection to the pool and may continue on another one, which would strand a
  session-level lock on a connection nobody will ever unlock.
* **Self-contained.** Ingest, enrich, generate and index happen here, in order, so a
  scheduler needs exactly one entry.
"""

from __future__ import annotations

import asyncio
import uuid
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db import session_scope
from voicebrief.db.models import Episode, EpisodeStatus, EpisodeStyle, Language, User
from voicebrief.db.session import get_engine
from voicebrief.logging import get_logger

log = get_logger(__name__)

DAILY_LOCK = "voicebrief:daily"
DEFAULT_EMAIL = "me@example.com"
DEFAULT_TOPICS = ("agentic-ai", "tooling")
DEFAULT_STORIES = 6
DEFAULT_MINUTES = 10


class NoFreshStories(RuntimeError):
    """Everything recent is either missing or already aired."""


@dataclass(slots=True)
class BriefRequest:
    """What to generate. Anything left None falls back to the user's saved settings."""

    email: str = DEFAULT_EMAIL
    github_login: str | None = None
    topics: list[str] | None = None
    language: str | None = None
    style: str | None = None
    stories: int | None = None
    minutes: int | None = None
    render_audio: bool = True


@dataclass(slots=True)
class DailyResult:
    outcome: str  # generated | already_done | locked | failed
    episode_id: uuid.UUID | None = None
    status: str | None = None
    title: str | None = None
    items_ingested: int = 0
    clusters: int = 0
    duration_seconds: float = 0.0
    cost_inr: float = 0.0
    attribution_rate: float | None = None
    issues: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def exit_code(self) -> int:
        """0 good (or nothing to do), 2 degraded episode, 1 failure."""
        if self.outcome == "failed" or self.status == "failed":
            return 1
        if self.status == "degraded":
            return 2
        return 0


@contextmanager
def daily_lock() -> Iterator[bool]:
    """Cross-process lock for the whole run, on its own connection."""
    lock_id = zlib.crc32(DAILY_LOCK.encode()) % (2**31)
    with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        acquired = bool(
            conn.execute(text("SELECT pg_try_advisory_lock(:i)"), {"i": lock_id}).scalar_one()
        )
        try:
            yield acquired
        finally:
            if acquired:
                conn.execute(text("SELECT pg_advisory_unlock(:i)"), {"i": lock_id})


def local_day_start(now: datetime, timezone: str) -> datetime:
    """Midnight, in the user's timezone, as a UTC instant.

    "Today's brief" means the user's calendar day. At 00:30 IST the UTC date is still
    yesterday, which used to make a morning brief look like it belonged to the day
    before.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        log.warning("daily.unknown_timezone", timezone=timezone, fallback="UTC")
        tz = UTC
    local = now.astimezone(tz)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def todays_episode(
    session: Session, user_id: uuid.UUID, *, now: datetime, timezone: str
) -> Episode | None:
    """The user's usable episode from their current local day, if any."""
    since = local_day_start(now, timezone)
    return (
        session.execute(
            select(Episode)
            .where(
                Episode.user_id == user_id,
                Episode.created_at >= since,
                Episode.status.in_([EpisodeStatus.ready, EpisodeStatus.degraded]),
            )
            .order_by(Episode.created_at.desc())
        )
        .scalars()
        .first()
    )


def resolve_user(session: Session, request: BriefRequest) -> User:
    """Load the user, creating them on first use, and persist what was just supplied.

    Settings given once (GitHub login, topics, language, style) are stored on the user
    so the scheduler never needs them repeated and a later run cannot forget them.
    """
    user = session.execute(select(User).where(User.email == request.email)).scalar_one_or_none()
    if user is None:
        user = User(email=request.email)
        session.add(user)

    if request.github_login:
        user.github_login = request.github_login
    if request.topics:
        user.topics = list(request.topics)
    if request.language:
        user.language = Language(request.language)
    if request.style:
        user.style = EpisodeStyle(request.style)
    session.flush()
    return user


def resolve_profile(session: Session, user: User, *, refresh: bool = False):
    """The user's stack profile: cached when fresh, rebuilt when stale, never worse.

    A profile is a quality improvement, not a precondition. Rate limits and network
    errors cost, at worst, a stale profile, never the day's brief and never a good
    profile replaced by an empty one.
    """
    login = user.github_login
    if not login:
        return None

    from voicebrief.personalization.github_profile import GitHubProfileBuilder
    from voicebrief.personalization.profile_cache import is_fresh, load_profile, save_profile

    cached = load_profile(session, user.id, login=login)
    if cached and not refresh and is_fresh(cached[1]):
        log.info("daily.profile_cached", login=login, dependencies=len(cached[0].dependencies))
        return cached[0]

    try:
        built = asyncio.run(GitHubProfileBuilder().build(login))
    except Exception as exc:
        log.warning("daily.profile_failed", login=login, error=f"{type(exc).__name__}: {exc}")
        return cached[0] if cached else None

    if built.complete and not built.is_empty:
        save_profile(session, user.id, built)
        log.info(
            "daily.profile",
            login=login,
            dependencies=len(built.dependencies),
            repos=len(built.repos),
        )
        return built

    if cached:
        log.warning("daily.profile_kept_stale", login=login, reason=built.incomplete_reason)
        return cached[0]
    log.warning("daily.profile_unusable", login=login, reason=built.incomplete_reason)
    return built


def generate_brief(session: Session, request: BriefRequest):
    """Everything between "fresh clusters exist" and "an episode is stored".

    Shared by `brief generate` and `daily`, so there is one definition of how a brief
    is made. Raises `NoFreshStories` when there is nothing new to say.
    """
    from voicebrief.pipeline.candidates import build_candidates
    from voicebrief.pipeline.episodes import generate_episode

    user = resolve_user(session, request)
    profile = resolve_profile(session, user)

    candidates = build_candidates(session, user_id=user.id)
    if not candidates:
        raise NoFreshStories(
            "no fresh stories to cover: run ingest and enrich, or every recent story "
            "has already been aired"
        )

    return generate_episode(
        session,
        user_id=user.id,
        candidates=candidates,
        profile=profile,
        declared_topics=set(user.topics or DEFAULT_TOPICS),
        language=user.language.value,
        style=user.style.value,
        max_stories=request.stories or DEFAULT_STORIES,
        target_minutes=request.minutes or DEFAULT_MINUTES,
        render_audio=request.render_audio,
    )


def run_daily(
    request: BriefRequest,
    *,
    force: bool = False,
    skip_ingest: bool = False,
    now: datetime | None = None,
) -> DailyResult:
    """Ingest, enrich, generate and index today's brief, once."""
    now = now or datetime.now(UTC)
    timezone = get_settings().timezone

    with daily_lock() as acquired:
        if not acquired:
            log.warning("daily.skipped", reason="another daily run holds the lock")
            return DailyResult(outcome="locked")

        try:
            with session_scope() as session:
                user = resolve_user(session, request)
                existing = todays_episode(session, user.id, now=now, timezone=timezone)
                if existing is not None and not force:
                    log.info("daily.already_done", episode=str(existing.id))
                    return DailyResult(
                        outcome="already_done",
                        episode_id=existing.id,
                        status=existing.status.value,
                        title=existing.title,
                    )
                topics = set(user.topics or DEFAULT_TOPICS)

            result = DailyResult(outcome="generated")

            if not skip_ingest:
                from voicebrief.pipeline.ingest import run_ingest

                stats = asyncio.run(run_ingest())
                result.items_ingested = sum(s.inserted for s in stats)

            from voicebrief.pipeline.enrich import enrich

            with session_scope() as session:
                enriched = enrich(session, user_topics=topics)
            result.clusters = enriched.clusters

            with session_scope() as session:
                generated = generate_brief(session, request)

            result.episode_id = generated.episode_id
            result.status = generated.status
            result.title = generated.title
            result.duration_seconds = generated.duration_seconds
            result.cost_inr = generated.cost_inr
            result.attribution_rate = generated.attribution_rate
            result.issues = list(generated.errors)
            return result

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.error("daily.failed", error=error)
            return DailyResult(outcome="failed", status="failed", error=error[:500])
