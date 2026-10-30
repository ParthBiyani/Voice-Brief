"""What counts as a finished episode, and what a failed run leaves behind."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import CostEntry, Episode, EpisodeStatus, User
from voicebrief.db.session import get_engine
from voicebrief.llm.base import BudgetExceeded, LLMError, Stage
from voicebrief.pipeline.episodes import generate_episode, judge_episode
from voicebrief.pipeline.ranking import RankCandidate
from voicebrief.pipeline.summarize import SourceRef, summarize_cluster


def judge(**over):
    base = {
        "story_count": 6,
        "requested_stories": 6,
        "render_audio": True,
        "audio_key": "episodes/x.wav",
        "duration_seconds": 420.0,
        "engine_name": "kokoro",
        "configured_engine": "kokoro",
    }
    return judge_episode(**{**base, **over})


class TestJudgeEpisode:
    def test_a_complete_episode_is_ready(self):
        status, reasons = judge()
        assert status is EpisodeStatus.ready
        assert reasons == []

    def test_no_stories_is_a_failure(self):
        status, reasons = judge(story_count=0)
        assert status is EpisodeStatus.failed
        assert "no stories" in reasons[0]

    def test_a_silent_track_from_a_missing_engine_is_degraded_not_ready(self):
        status, reasons = judge(engine_name="null", configured_engine="kokoro")
        assert status is EpisodeStatus.degraded
        assert any("silent" in r for r in reasons)

    def test_deliberately_silent_runs_are_still_ready(self):
        status, _ = judge(engine_name="null", configured_engine="null")
        assert status is EpisodeStatus.ready

    def test_audio_that_never_reached_storage_is_degraded(self):
        status, reasons = judge(audio_key=None)
        assert status is EpisodeStatus.degraded
        assert any("not stored" in r for r in reasons)

    def test_zero_length_audio_is_degraded(self):
        status, _ = judge(duration_seconds=0.0)
        assert status is EpisodeStatus.degraded

    def test_half_the_stories_missing_is_degraded(self):
        status, reasons = judge(story_count=2, requested_stories=6)
        assert status is EpisodeStatus.degraded
        assert any("2 of 6" in r for r in reasons)

    def test_text_only_runs_do_not_need_audio(self):
        status, _ = judge(render_audio=False, audio_key=None, duration_seconds=0.0)
        assert status is EpisodeStatus.ready


def candidate() -> RankCandidate:
    item = uuid.uuid4()
    return RankCandidate(
        cluster_id=uuid.uuid4(),
        title="A release",
        summary="s",
        url="https://example.com/a",
        sources=[
            SourceRef(item_id=item, title="A release", url="https://example.com/a", source_slug="t")
        ],
        bodies={item: "Version 2 shipped today with faster startup."},
    )


class _Client:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def complete(self, **_):
        return self.behaviour()


class TestSummaryHandling:
    def test_budget_exhaustion_stops_the_run_instead_of_being_swallowed(self):
        def boom():
            raise BudgetExceeded("daily budget exhausted")

        c = candidate()
        with pytest.raises(BudgetExceeded):
            summarize_cluster(
                _Client(boom), cluster_id=c.cluster_id, sources=c.sources, bodies=c.bodies
            )

    def test_other_model_failures_drop_only_that_story(self):
        def boom():
            raise LLMError("model declined")

        c = candidate()
        assert (
            summarize_cluster(
                _Client(boom), cluster_id=c.cluster_id, sources=c.sources, bodies=c.bodies
            )
            is None
        )

    def test_a_summary_with_no_cited_facts_is_dropped(self):
        class Done:
            stage = Stage.cluster_summary

            def __init__(self):
                self.parsed = {
                    "headline": "h",
                    "what_changed": "w",
                    "why_it_matters": "y",
                    "key_facts": [{"fact": "invented", "source_index": 99}],
                }

        c = candidate()
        assert (
            summarize_cluster(
                _Client(lambda: Done()), cluster_id=c.cluster_id, sources=c.sources, bodies=c.bodies
            )
            is None
        )


@pytest.mark.integration
class TestFailedRunsLeaveEvidence:
    def test_a_crashed_generation_keeps_its_failed_row(self, monkeypatch):
        engine = get_engine()
        email = f"{uuid.uuid4().hex[:8]}@example.com"

        def crash(self, *a, **k):
            raise RuntimeError("graph exploded")

        monkeypatch.setattr("voicebrief.pipeline.episodes.BriefGraph.run", crash)
        episode_id = None
        try:
            with Session(engine) as work:
                user = User(email=email)
                work.add(user)
                work.commit()
                with pytest.raises(RuntimeError):
                    generate_episode(work, user_id=user.id, candidates=[], render_audio=False)
                work.rollback()  # what the caller's scope does on the re-raise

            with Session(engine) as check:
                row = check.execute(
                    select(Episode).join(User).where(User.email == email)
                ).scalar_one()
                episode_id = row.id
                assert row.status is EpisodeStatus.failed
                assert "graph exploded" in row.error
        finally:
            with Session(engine) as cleanup:
                if episode_id:
                    cleanup.query(CostEntry).filter(CostEntry.episode_id == episode_id).delete()
                cleanup.query(Episode).filter(
                    Episode.user_id.in_(select(User.id).where(User.email == email))
                ).delete(synchronize_session=False)
                cleanup.query(User).filter(User.email == email).delete()
                cleanup.commit()


@pytest.mark.integration
class TestFailureAfterGeneration:
    """A crash while voicing, storing or saving used to leave the episode `running`,
    because only the model step was guarded."""

    def run_with(self, monkeypatch, *, breaks):
        from voicebrief.graphs.brief import Segment

        email = f"late-{uuid.uuid4().hex[:8]}@example.com"
        engine = get_engine()
        written = {
            "title": "t",
            "segments": [Segment(position=0, kind="story", heading="h", script="Some words here.")],
            "grounding": {},
            "errors": [],
        }
        monkeypatch.setattr(*breaks)
        try:
            with Session(engine) as work:
                user = User(email=email)
                work.add(user)
                work.commit()
                with pytest.raises(RuntimeError):
                    generate_episode(
                        work,
                        user_id=user.id,
                        candidates=[],
                        produce=lambda client: written,
                        render_audio=True,
                    )
                work.rollback()  # what the caller's scope does on the re-raise
            with Session(engine) as check:
                return check.execute(
                    select(Episode).join(User).where(User.email == email)
                ).scalar_one()
        finally:
            with Session(engine) as cleanup:
                ids = [u.id for u in cleanup.query(User).filter(User.email == email)]
                cleanup.query(CostEntry).filter(
                    CostEntry.episode_id.in_(select(Episode.id).where(Episode.user_id.in_(ids)))
                ).delete(synchronize_session=False)
                cleanup.query(Episode).filter(Episode.user_id.in_(ids)).delete(
                    synchronize_session=False
                )
                cleanup.query(User).filter(User.email == email).delete()
                cleanup.commit()

    def test_a_crash_while_voicing_leaves_a_failed_episode_not_a_running_one(self, monkeypatch):
        def exploding_engine(*a, **k):
            raise RuntimeError("speech engine exploded")

        episode = self.run_with(
            monkeypatch, breaks=("voicebrief.pipeline.episodes.build_engine", exploding_engine)
        )
        assert episode.status is EpisodeStatus.failed
        assert "speech engine exploded" in episode.error

    def test_a_crash_while_saving_leaves_a_failed_episode_too(self, monkeypatch):
        def exploding_index(*a, **k):
            raise RuntimeError("could not record")

        # record_ranking is the last step before the result is built.
        episode = self.run_with(
            monkeypatch, breaks=("voicebrief.pipeline.episodes.record_ranking", exploding_index)
        )
        assert episode.status is EpisodeStatus.failed
        assert "could not record" in episode.error
