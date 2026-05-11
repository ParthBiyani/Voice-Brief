"""Engine and session lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from voicebrief.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.postgres_dsn,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


@contextmanager
def advisory_lock(session: Session, key: str) -> Iterator[bool]:
    """Postgres advisory lock — the whole scheduler story for v1.

    A cron hit to an endpoint plus one of these is sufficient concurrency control;
    Celery and Temporal are deferred (PRD §8).
    """
    lock_id = abs(hash(key)) % (2**31)
    acquired = session.execute(
        text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": lock_id}
    ).scalar_one()
    try:
        yield bool(acquired)
    finally:
        if acquired:
            session.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": lock_id})
