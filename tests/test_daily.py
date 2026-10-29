"""The morning loop: idempotent, exclusive, and honest about how it ended."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from voicebrief.db.models import Episode, EpisodeMode, EpisodeStatus, User
from voicebrief.db.session import get_engine
from voicebrief.pipeline.daily import (
    BriefRequest,
    DailyResult,
    daily_lock,
    local_day_start,
    run_daily,
)


def explode(*_args, **_kwargs):
    raise RuntimeError("enrich exploded")


class TestLocalDay:
    def test_a_morning_in_india_belongs_to_the_indian_date_not_the_utc_one(self):
        # 00:30 IST on 30 Oct is 19:00 UTC on 29 Oct: the UTC date is still "yesterday".
        now = datetime(2026, 10, 29, 19, 0, tzinfo=UTC)
        start = local_day_start(now, "Asia/Kolkata")
        assert start == datetime(2026, 10, 29, 18, 30, tzinfo=UTC)  # 00:00 IST on the 30th

    def test_a_brief_made_before_local_midnight_is_not_todays(self):
        now = datetime(2026, 10, 29, 19, 0, tzinfo=UTC)
        yesterday_evening = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)  # 17:30 IST on the 29th
        assert yesterday_evening < local_day_start(now, "Asia/Kolkata")

    def test_an_unknown_timezone_falls_back_instead_of_crashing(self):
        now = datetime(2026, 10, 29, 19, 0, tzinfo=UTC)
        assert local_day_start(now, "Not/AZone") == datetime(2026, 10, 29, 0, 0, tzinfo=UTC)


class TestExitCodes:
    @pytest.mark.parametrize(
        ("result", "code"),
        [
            (DailyResult(outcome="generated", status="ready"), 0),
            (DailyResult(outcome="already_done", status="ready"), 0),
            (DailyResult(outcome="locked"), 0),
            (DailyResult(outcome="generated", status="degraded"), 2),
            (DailyResult(outcome="generated", status="failed"), 1),
            (DailyResult(outcome="failed", status="failed", error="x"), 1),
        ],
    )
    def test_the_scheduler_can_tell_outcomes_apart(self, result, code):
        assert result.exit_code == code


@pytest.mark.integration
class TestDailyRun:
    @pytest.fixture
    def email(self):
        address = f"daily-{uuid.uuid4().hex[:8]}@example.com"
        yield address
        with Session(get_engine()) as cleanup:
            ids = [u.id for u in cleanup.query(User).filter(User.email == address)]
            cleanup.query(Episode).filter(Episode.user_id.in_(ids)).delete(
                synchronize_session=False
            )
            cleanup.query(User).filter(User.email == address).delete()
            cleanup.commit()

    def _make_episode(self, email, *, status):
        with Session(get_engine()) as s:
            user = s.query(User).filter(User.email == email).one_or_none() or User(email=email)
            s.add(user)
            s.flush()
            s.add(
                Episode(
                    user_id=user.id,
                    mode=EpisodeMode.brief,
                    status=status,
                    title="Earlier brief",
                    created_at=datetime.now(UTC) - timedelta(minutes=1),
                )
            )
            s.commit()

    def test_a_second_run_the_same_day_does_nothing(self, email):
        self._make_episode(email, status=EpisodeStatus.ready)
        result = run_daily(BriefRequest(email=email), skip_ingest=True)
        assert result.outcome == "already_done"
        assert result.exit_code == 0
        assert result.title == "Earlier brief"

    def test_a_degraded_brief_still_counts_as_todays(self, email):
        self._make_episode(email, status=EpisodeStatus.degraded)
        assert run_daily(BriefRequest(email=email), skip_ingest=True).outcome == "already_done"

    def test_a_failed_attempt_does_not_block_a_retry(self, email, monkeypatch):
        self._make_episode(email, status=EpisodeStatus.failed)
        monkeypatch.setattr("voicebrief.pipeline.enrich.enrich", explode)
        result = run_daily(BriefRequest(email=email), skip_ingest=True)
        assert result.outcome == "failed"
        assert "enrich exploded" in result.error
        assert result.exit_code == 1

    def test_the_lock_is_released_after_a_failure(self, email, monkeypatch):
        monkeypatch.setattr("voicebrief.pipeline.enrich.enrich", explode)
        run_daily(BriefRequest(email=email), skip_ingest=True)
        with daily_lock() as acquired:
            assert acquired

    def test_force_regenerates_even_when_todays_exists(self, email, monkeypatch):
        self._make_episode(email, status=EpisodeStatus.ready)
        monkeypatch.setattr("voicebrief.pipeline.enrich.enrich", explode)
        # Reaching the (sabotaged) enrich step proves the early return was bypassed.
        result = run_daily(BriefRequest(email=email), skip_ingest=True, force=True)
        assert result.outcome == "failed"

    def test_settings_given_once_are_remembered(self, email, monkeypatch):
        monkeypatch.setattr("voicebrief.pipeline.enrich.enrich", explode)
        run_daily(
            BriefRequest(email=email, github_login="someone", topics=["rust", "edge-ai"]),
            skip_ingest=True,
        )
        with Session(get_engine()) as s:
            user = s.query(User).filter(User.email == email).one()
            assert user.github_login == "someone"
            assert user.topics == ["rust", "edge-ai"]


@pytest.mark.integration
class TestLock:
    def test_two_runs_cannot_hold_the_lock_at_once(self):
        with daily_lock() as first:
            assert first
            with daily_lock() as second:
                assert second is False
        with daily_lock() as again:
            assert again
