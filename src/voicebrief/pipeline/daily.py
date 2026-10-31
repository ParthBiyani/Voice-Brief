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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db import session_scope
from voicebrief.db.models import Episode, EpisodeStatus, EpisodeStyle, Language, User
from voicebrief.db.session import get_engine
from voicebrief.delivery.deliver import deliver_episode, notify_failure
from voicebrief.history import (
    finish_run,
    prune_llm_bodies,
    reap_interrupted,
    record_interaction,
    start_run,
)
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
class BriefOutcome:
    """A generated episode plus what the selection step saw."""

    result: object  # GenerationResult
    candidates_considered: int
    excluded_aired: int
    window_hours: float


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

    changes: dict[str, dict] = {}

    def apply(name: str, new, current) -> None:
        if new is not None and new != current:
            changes[name] = {"from": current, "to": new}
            setattr(user, name, new)

    apply("github_login", request.github_login or None, user.github_login)
    apply("topics", list(request.topics) if request.topics else None, list(user.topics or []))
    apply(
        "language",
        Language(request.language) if request.language else None,
        user.language,
    )
    apply("style", EpisodeStyle(request.style) if request.style else None, user.style)
    session.flush()

    if changes:
        record_interaction(
            "settings",
            "cli",
            user_id=user.id,
            request={name: change["to"] for name, change in changes.items()},
            response={"previous": {n: c["from"] for n, c in changes.items()}},
        )
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


def seed_terms(profile, limit: int = 4) -> list[str]:
    """The technologies the listener uses most, to ask ProjectMind about by name.

    Only plain, recognisable names: transitive packages (`flutter_test`, `markdown-it-py`)
    and common words are noise to ProjectMind's relevance gate.
    """
    import re

    from voicebrief.pipeline.ranking import is_ambiguous_dependency

    if profile is None:
        return []
    ranked = sorted(
        profile.dependencies.items(), key=lambda kv: len(kv[1].get("repos", [])), reverse=True
    )
    plain = re.compile(r"^[a-z][a-z0-9]{3,}$")
    names = [n for n, _ in ranked if plain.match(n) and not is_ambiguous_dependency(n)]
    return names[:limit]


def resolve_personal(
    session: Session, user: User, *, refresh: bool = False, seeds: list[str] | None = None
):
    """Personal context from ProjectMind: cached for a day, never a reason to fail.

    ProjectMind is a separate, independently changing program. If it is not configured,
    slow, broken or returns nothing, whatever was cached last is used instead.
    """
    from voicebrief.personalization.profile_cache import load_personal, save_personal
    from voicebrief.personalization.projectmind import (
        CONTEXT_TTL_HOURS,
        configured_command,
        fetch_personal_context,
    )

    cached = load_personal(session, user.id)
    now = datetime.now(UTC)
    if (
        cached
        and not refresh
        and cached.fetched_at
        and (now - cached.fetched_at).total_seconds() < CONTEXT_TTL_HOURS * 3600
    ):
        return cached
    if configured_command() is None:
        return cached

    fetched = fetch_personal_context(seeds=seeds)
    if fetched is not None:
        save_personal(session, user.id, fetched)
        return fetched
    if cached:
        log.warning("daily.personal_kept_stale", reason="ProjectMind returned nothing")
    return cached


def attach_personal(profile, context, *, login: str | None):
    """Fold personal context into the profile the ranker and writer already consume."""
    if context is None or context.is_empty:
        return profile
    from voicebrief.personalization.github_profile import StackProfileData

    profile = profile or StackProfileData(login=login or "")
    profile.interests = list(context.interests)
    profile.avoid = list(context.avoid)
    profile.notes = list(context.statements)
    profile.projects = list(context.projects)
    return profile


def generate_brief(
    session: Session,
    request: BriefRequest,
    *,
    run_id: uuid.UUID | None = None,
    episode_id: uuid.UUID | None = None,
) -> BriefOutcome:
    """Everything between "fresh clusters exist" and "an episode is stored".

    Shared by `brief generate` and `daily`, so there is one definition of how a brief
    is made. Raises `NoFreshStories` when there is nothing new to say.
    """
    from voicebrief.pipeline.candidates import CandidateReport, build_candidates
    from voicebrief.pipeline.episodes import generate_episode

    user = resolve_user(session, request)
    profile = resolve_profile(session, user)
    try:
        personal = resolve_personal(session, user, seeds=seed_terms(profile))
        profile = attach_personal(profile, personal, login=user.github_login)
    except Exception as exc:  # personal context is an enhancement, never a precondition
        log.warning("daily.personal_failed", error=f"{type(exc).__name__}: {exc}"[:200])

    report = CandidateReport(window_hours=0)
    candidates = build_candidates(session, user_id=user.id, report=report)
    if not candidates:
        raise NoFreshStories(
            "no fresh stories to cover: run ingest and enrich, or every recent story "
            "has already been aired"
        )

    result = generate_episode(
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
        excluded=report.excluded,
        run_id=run_id,
        episode_id=episode_id,
    )
    return BriefOutcome(
        result=result,
        candidates_considered=report.candidates,
        excluded_aired=report.excluded_aired,
        window_hours=report.window_hours,
    )


def run_daily(
    request: BriefRequest,
    *,
    force: bool = False,
    skip_ingest: bool = False,
    now: datetime | None = None,
    trigger: str = "cli",
) -> DailyResult:
    """Ingest, enrich, generate and index today's brief, once."""
    now = now or datetime.now(UTC)
    timezone = get_settings().timezone

    with daily_lock() as acquired:
        if not acquired:
            log.warning("daily.skipped", reason="another daily run holds the lock")
            return DailyResult(outcome="locked")

        # Holding the lock means no other daily run is alive, so anything a previous
        # one left `running` was cut short and is closed out before this one starts.
        reap_interrupted()
        run_id = start_run("daily", trigger)
        try:
            with session_scope() as session:
                user = resolve_user(session, request)
                existing = todays_episode(session, user.id, now=now, timezone=timezone)
                if existing is not None and not force:
                    log.info("daily.already_done", episode=str(existing.id))
                    finish_run(
                        run_id,
                        "skipped",
                        summary={"reason": "today's brief already exists"},
                        episode_id=existing.id,
                    )
                    return DailyResult(
                        outcome="already_done",
                        episode_id=existing.id,
                        status=existing.status.value,
                        title=existing.title,
                    )
                topics = set(user.topics or DEFAULT_TOPICS)

            result = DailyResult(outcome="generated")
            summary: dict = {"forced": force}

            if not skip_ingest:
                from voicebrief.pipeline.ingest import run_ingest

                stats = asyncio.run(run_ingest())
                result.items_ingested = sum(s.inserted for s in stats)
                summary["ingest"] = {
                    "new_items": result.items_ingested,
                    "sources_polled": len(stats),
                    "sources_failed": sum(1 for s in stats if not s.ok),
                    "by_source": [asdict(s) for s in stats],
                }

            from voicebrief.pipeline.enrich import enrich

            with session_scope() as session:
                enriched = enrich(session, user_topics=topics)
            result.clusters = enriched.clusters
            summary["enrich"] = asdict(enriched)

            with session_scope() as session:
                outcome = generate_brief(session, request, run_id=run_id)
            generated = outcome.result
            summary["candidates"] = {
                "considered": outcome.candidates_considered,
                "already_aired": outcome.excluded_aired,
                "window_hours": round(outcome.window_hours, 1),
            }

            result.episode_id = generated.episode_id
            result.status = generated.status
            result.title = generated.title
            result.duration_seconds = generated.duration_seconds
            result.cost_inr = generated.cost_inr
            result.attribution_rate = generated.attribution_rate
            result.issues = list(generated.errors)
            summary.update(
                {
                    "status": generated.status,
                    "cost_inr": generated.cost_inr,
                    "attribution_rate": generated.attribution_rate,
                    "issues": result.issues[:10],
                }
            )
            if generated.status == "failed":
                summary["delivery"] = notify_failure(
                    "; ".join(result.issues[:3]) or "the episode could not be produced"
                )
            else:
                summary["delivery"] = deliver_episode(generated.episode_id)
            finish_run(
                run_id,
                "ok" if generated.status == "ready" else generated.status,
                summary=summary,
                episode_id=generated.episode_id,
            )
            _prune_history()
            return result

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.error("daily.failed", error=error)
            notify_failure(error)
            finish_run(run_id, "failed", error=error)
            return DailyResult(outcome="failed", status="failed", error=error[:500])


def _prune_history() -> None:
    """Housekeeping that must never cost a brief."""
    try:
        with session_scope() as session:
            pruned = prune_llm_bodies(session)
        if pruned:
            log.info("history.pruned", llm_bodies=pruned)
    except Exception as exc:
        log.warning("history.prune_failed", error=str(exc))
