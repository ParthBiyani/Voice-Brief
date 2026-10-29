"""Persist the stack profile so one bad build cannot erase personalisation.

Building a profile costs dozens of GitHub requests, which an unauthenticated client gets
sixty of per hour. The profile changes slowly, so it is stored per user and reused for a
week; a build that comes back incomplete falls back to the last good one instead of
replacing it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import StackProfile
from voicebrief.personalization.github_profile import StackProfileData

PROFILE_TTL = timedelta(days=7)


def save_profile(session: Session, user_id: uuid.UUID, profile: StackProfileData) -> None:
    row = session.execute(
        select(StackProfile).where(StackProfile.user_id == user_id)
    ).scalar_one_or_none()
    if row is None:
        row = StackProfile(user_id=user_id)
        session.add(row)
    row.languages = dict(profile.languages)
    row.dependencies = dict(profile.dependencies)
    row.repos = list(profile.repos)
    row.starred_topics = list(profile.starred_topics)
    row.recent_commit_terms = list(profile.recent_commit_terms)
    row.built_at = profile.built_at or datetime.now(UTC)
    session.flush()


def load_profile(
    session: Session, user_id: uuid.UUID, *, login: str
) -> tuple[StackProfileData, datetime] | None:
    """The stored profile and when it was built, if one exists."""
    row = session.execute(
        select(StackProfile).where(StackProfile.user_id == user_id)
    ).scalar_one_or_none()
    if row is None or row.built_at is None:
        return None
    profile = StackProfileData(
        login=login,
        languages=dict(row.languages or {}),
        dependencies=dict(row.dependencies or {}),
        repos=list(row.repos or []),
        starred_topics=list(row.starred_topics or []),
        recent_commit_terms=list(row.recent_commit_terms or []),
        built_at=row.built_at,
    )
    return profile, row.built_at


def save_personal(session: Session, user_id: uuid.UUID, context) -> None:
    """Cache ProjectMind-derived context alongside the stack profile."""
    row = session.execute(
        select(StackProfile).where(StackProfile.user_id == user_id)
    ).scalar_one_or_none()
    if row is None:
        row = StackProfile(user_id=user_id)
        session.add(row)
    row.personal = context.to_json()
    session.flush()


def load_personal(session: Session, user_id: uuid.UUID):
    from voicebrief.personalization.projectmind import PersonalContext

    row = session.execute(
        select(StackProfile).where(StackProfile.user_id == user_id)
    ).scalar_one_or_none()
    if row is None or not row.personal:
        return None
    try:
        return PersonalContext.from_json(row.personal)
    except Exception:
        return None


def is_fresh(built_at: datetime, *, now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    if built_at.tzinfo is None:
        built_at = built_at.replace(tzinfo=UTC)
    return now - built_at < PROFILE_TTL
