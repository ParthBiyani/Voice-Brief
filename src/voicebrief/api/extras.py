"""Routes added for the web app: audio streaming, settings, and ranking explanations.

Audio is proxied rather than handed out as a presigned storage link. A presigned link
points at wherever the object store happens to be reachable from the server
(`localhost:59000`), which a phone or another computer cannot reach, and it expires after
an hour. Streaming it through the API works from any device that can reach the app, never
expires, and supports seeking.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.api import progress
from voicebrief.config import get_settings
from voicebrief.db import get_session
from voicebrief.db.models import (
    Episode,
    EpisodeStyle,
    Language,
    RankingSnapshot,
    StackProfile,
    User,
)
from voicebrief.history import record_interaction
from voicebrief.logging import get_logger
from voicebrief.pipeline.episodes import s3_client

log = get_logger(__name__)
router = APIRouter()

CHUNK = 64 * 1024
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────
def parse_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """Parse a single `Range: bytes=a-b` header.

    Returns (start, end) inclusive, None when there is no usable range (serve it all),
    or "invalid" when the range cannot be satisfied.
    """
    if not header:
        return None
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if size <= 0:
        return "invalid"
    if first == "":  # suffix range: the last N bytes
        length = int(last)
        if length == 0:
            return "invalid"
        return max(size - length, 0), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        return "invalid"
    return start, min(end, size - 1)


@router.get("/episodes/{episode_id}/audio", tags=["episodes"])
def stream_audio(
    episode_id: uuid.UUID,
    request: Request,
    download: bool = Query(False),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    episode = session.get(Episode, episode_id)
    if episode is None or not episode.audio_key:
        raise HTTPException(status_code=404, detail="no audio for this episode")

    key = episode.audio_key
    bucket = get_settings().s3_bucket
    client = s3_client()
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except (BotoCoreError, ClientError) as exc:
        log.warning("audio.unavailable", key=key, error=str(exc))
        raise HTTPException(status_code=502, detail="audio storage is unavailable") from exc

    size = int(head["ContentLength"])
    content_type = head.get("ContentType") or (
        "audio/mpeg" if key.endswith(".mp3") else "audio/wav"
    )
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if download:
        extension = "mp3" if key.endswith(".mp3") else "wav"
        headers["Content-Disposition"] = (
            f'attachment; filename="voicebrief-{episode_id}.{extension}"'
        )

    byte_range = parse_range(request.headers.get("range"), size)
    if byte_range == "invalid":
        raise HTTPException(
            status_code=416,
            detail="range not satisfiable",
            headers={"Content-Range": f"bytes */{size}"},
        )

    if byte_range is None:
        start, end, status = 0, size - 1, 200
        kwargs: dict = {}
    else:
        start, end = byte_range  # type: ignore[misc]
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        kwargs = {"Range": f"bytes={start}-{end}"}

    try:
        body = client.get_object(Bucket=bucket, Key=key, **kwargs)["Body"]
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail="audio storage is unavailable") from exc

    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        body.iter_chunks(CHUNK), status_code=status, media_type=content_type, headers=headers
    )


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")


class SettingsIn(BaseModel):
    github_login: str | None = None
    topics: list[str] | None = None
    language: str | None = None
    style: str | None = None

    @field_validator("github_login")
    @classmethod
    def _login(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lstrip("@")
        if value and not _LOGIN_RE.match(value):
            raise ValueError("not a valid GitHub username")
        return value

    @field_validator("topics")
    @classmethod
    def _topics(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for topic in value:
            topic = topic.strip().lower()
            if topic and len(topic) <= 40 and topic not in cleaned:
                cleaned.append(topic)
        if len(cleaned) > 25:
            raise ValueError("at most 25 topics")
        return cleaned

    @field_validator("language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        if value is not None and value not in {item.value for item in Language}:
            raise ValueError("language must be en or hi")
        return value

    @field_validator("style")
    @classmethod
    def _style(cls, value: str | None) -> str | None:
        if value is not None and value not in {item.value for item in EpisodeStyle}:
            raise ValueError("style must be solo_anchor or two_host")
        return value


class IntegrationStatus(BaseModel):
    configured: bool
    detail: str = ""
    fetched_at: datetime | None = None
    items: int = 0


class SettingsOut(BaseModel):
    email: str
    github_login: str | None = None
    topics: list[str] = Field(default_factory=list)
    language: str = "en"
    style: str = "solo_anchor"
    github: IntegrationStatus
    projectmind: IntegrationStatus
    llm_provider: str
    tts_engine: str
    daily_budget_inr: float


def _settings_out(session: Session, user: User) -> SettingsOut:
    from voicebrief.personalization.projectmind import configured_command

    settings = get_settings()
    profile = session.execute(
        select(StackProfile).where(StackProfile.user_id == user.id)
    ).scalar_one_or_none()

    dependencies = len(profile.dependencies or {}) if profile else 0
    github = IntegrationStatus(
        configured=bool(user.github_login),
        detail=(
            f"{dependencies} dependencies across {len(profile.repos or [])} repositories"
            if dependencies
            else (
                "no profile built yet; it is built on the next run"
                if user.github_login
                else "add your GitHub username to personalise by the code you write"
            )
        ),
        fetched_at=profile.built_at if profile else None,
        items=dependencies,
    )

    personal = (profile.personal or {}) if profile else {}
    command = configured_command()
    fetched = personal.get("fetched_at")
    projectmind = IntegrationStatus(
        configured=command is not None,
        detail=(
            f"{len(personal.get('interests', []))} interests, "
            f"{len(personal.get('statements', []))} statements"
            if personal
            else (
                "connected; nothing fetched yet"
                if command
                else "not connected (set VB_PROJECTMIND_CMD to use your ProjectMind memory)"
            )
        ),
        fetched_at=datetime.fromisoformat(fetched) if fetched else None,
        items=len(personal.get("interests", [])),
    )

    return SettingsOut(
        email=user.email,
        github_login=user.github_login,
        topics=list(user.topics or []),
        language=user.language.value,
        style=user.style.value,
        github=github,
        projectmind=projectmind,
        llm_provider=settings.llm_provider,
        tts_engine=settings.tts_engine,
        daily_budget_inr=settings.daily_budget_inr,
    )


@router.get("/settings", response_model=SettingsOut, tags=["settings"])
def get_user_settings(
    email: str = Query("me@example.com"), session: Session = Depends(get_session)
) -> SettingsOut:
    user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        user = User(email=email)
        session.add(user)
        session.flush()
    return _settings_out(session, user)


@router.put("/settings", response_model=SettingsOut, tags=["settings"])
def update_user_settings(
    payload: SettingsIn,
    email: str = Query("me@example.com"),
    session: Session = Depends(get_session),
) -> SettingsOut:
    user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        user = User(email=email)
        session.add(user)
        session.flush()

    changes: dict = {}
    if payload.github_login is not None and (payload.github_login or None) != user.github_login:
        changes["github_login"] = payload.github_login or None
        user.github_login = payload.github_login or None
    if payload.topics is not None and payload.topics != list(user.topics or []):
        changes["topics"] = payload.topics
        user.topics = payload.topics
    if payload.language is not None and payload.language != user.language.value:
        changes["language"] = payload.language
        user.language = Language(payload.language)
    if payload.style is not None and payload.style != user.style.value:
        changes["style"] = payload.style
        user.style = EpisodeStyle(payload.style)
    session.flush()

    if changes:
        record_interaction("settings", "web", user_id=user.id, request=changes)
    return _settings_out(session, user)


# ─────────────────────────────────────────────────────────────────────────────
# Why these stories
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/episodes/{episode_id}/ranking", tags=["episodes"])
def episode_ranking(episode_id: uuid.UUID, session: Session = Depends(get_session)) -> list[dict]:
    """Every story considered for an episode and why it was picked, skipped or already
    heard, with each ranking signal's contribution."""
    if session.get(Episode, episode_id) is None:
        raise HTTPException(status_code=404, detail="episode not found")
    rows = session.execute(
        select(RankingSnapshot)
        .where(RankingSnapshot.episode_id == episode_id)
        .order_by(RankingSnapshot.position.nulls_last(), RankingSnapshot.created_at)
    ).scalars()
    return [
        {
            "outcome": r.outcome,
            "title": r.title,
            "url": r.url,
            "source": r.source_slug,
            "position": r.position,
            "score": r.final_score,
            "llm_score": r.llm_score,
            "signals": r.signals,
            "stack_match": r.matched_dependencies,
            "reason": r.rationale,
        }
        for r in rows
    ]


@router.get("/episodes/active/now", tags=["episodes"])
def active_generations() -> list[str]:
    """Generations currently running, so a page opened mid-run can follow along."""
    return [str(i) for i in progress.active_ids()]
