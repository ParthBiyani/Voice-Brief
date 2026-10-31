"""The audit trail: what was recorded, that recording cannot break a run, and retention."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief import history
from voicebrief.config import get_settings
from voicebrief.db.models import (
    CostEntry,
    Episode,
    EpisodeMode,
    EpisodeStatus,
    Interaction,
    RunLog,
    User,
)
from voicebrief.db.session import get_engine
from voicebrief.llm.base import Stage
from voicebrief.llm.client import LLMClient
from voicebrief.llm.echo_provider import EchoProvider
from voicebrief.pipeline.ranking import RankCandidate, heuristic_score, score_parts

NOW = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


def candidate(**over) -> RankCandidate:
    base = {
        "cluster_id": uuid.uuid4(),
        "title": "LangGraph adds durable execution",
        "summary": "Checkpointing for long running agents.",
        "url": "https://example.com/a",
        "topics": ["agentic-ai"],
        "trust_weight": 0.8,
        "engagement": 0.5,
        "cluster_size": 2,
        "freshness": NOW - timedelta(hours=3),
        "substance_chars": 900,
    }
    return RankCandidate(**{**base, **over})


class TestSignalBreakdown:
    def test_the_parts_add_up_to_the_score_the_ranker_uses(self):
        c = candidate()
        score, _matched, parts = score_parts(
            c, profile=None, declared_topics={"agentic-ai"}, now=NOW
        )
        assert score == pytest.approx(
            heuristic_score(c, profile=None, declared_topics={"agentic-ai"}, now=NOW)[0]
        )
        # Parts are rounded for storage, so allow for that.
        assert sum(parts.values()) == pytest.approx(score, abs=0.01)

    def test_each_signal_is_named_so_a_decision_can_be_explained(self):
        _, _, parts = score_parts(
            candidate(), profile=None, declared_topics={"agentic-ai"}, now=NOW
        )
        assert {"trust", "engagement", "recency", "coverage", "topics"} <= set(parts)

    def test_a_thin_story_shows_its_penalty(self):
        _, _, parts = score_parts(
            candidate(substance_chars=40), profile=None, declared_topics=set(), now=NOW
        )
        assert parts["thin_penalty"] < 0


class TestRecordingIsFailSafe:
    def test_a_broken_database_does_not_raise_from_record_interaction(self, monkeypatch):
        def broken():
            raise RuntimeError("database is down")

        monkeypatch.setattr(history, "session_scope", broken)
        history.record_interaction("qa", "web", request={"question": "x"})  # must not raise

    def test_a_broken_database_does_not_raise_from_run_bookkeeping(self, monkeypatch):
        def broken():
            raise RuntimeError("database is down")

        monkeypatch.setattr(history, "session_scope", broken)
        assert history.start_run("daily") is None
        history.finish_run(None, "ok")  # a missing run id is tolerated
        history.finish_run(uuid.uuid4(), "ok")  # and so is a broken database

    def test_payloads_are_made_json_safe(self):
        cleaned = history._jsonable({"id": uuid.uuid4(), "when": NOW, "set": {1}})
        assert isinstance(cleaned["id"], str)

    def test_long_text_is_clipped_with_a_marker(self):
        clipped = history._clip("x" * 9000)
        assert len(clipped) < 9000 and "more characters" in clipped


@pytest.mark.integration
class TestStoredHistory:
    @pytest.fixture
    def session(self):
        with Session(get_engine()) as s:
            s.begin()
            try:
                yield s
            finally:
                s.rollback()

    @pytest.fixture
    def episode(self, session):
        user = User(email=f"{uuid.uuid4().hex[:8]}@example.com")
        session.add(user)
        session.flush()
        ep = Episode(
            user_id=user.id,
            mode=EpisodeMode.brief,
            status=EpisodeStatus.ready,
            title="Test brief",
            grounding={"attribution_rate": 0.9, "warnings": []},
        )
        session.add(ep)
        session.flush()
        return ep

    def trace(self):
        return [
            {
                "cluster_id": uuid.uuid4(),
                "rank": 1,
                "picked": True,
                "title": "LangGraph adds durable execution",
                "url": "https://example.com/a",
                "source_slug": "simonwillison",
                "cluster_size": 2,
                "heuristic_score": 5.5,
                "llm_score": 8.0,
                "score": 7.0,
                "signals": {"stack": 3.0, "recency": 1.8},
                "matched_dependencies": ["langgraph"],
                "rationale": "touches a dependency you use",
            },
            {
                "cluster_id": uuid.uuid4(),
                "rank": 2,
                "picked": False,
                "title": "Funding round announced",
                "url": "https://example.com/b",
                "source_slug": "hackernews-top",
                "cluster_size": 1,
                "heuristic_score": 2.0,
                "llm_score": 1.0,
                "score": 1.4,
                "signals": {"trust": 0.6},
                "matched_dependencies": [],
                "rationale": "unrelated to the stack",
            },
        ]

    def test_every_candidate_and_its_fate_is_stored(self, session, episode):
        excluded = [{"cluster_id": uuid.uuid4(), "title": "Heard yesterday", "url": "https://x/y"}]
        stored = history.record_ranking(session, episode.id, self.trace(), excluded)
        assert stored == 3
        today = datetime.now(ZoneInfo(get_settings().timezone)).date()
        data = history.timeline(session, today)
        mine = next(e for e in data["episodes"] if e["id"] == str(episode.id))
        outcomes = {r["title"]: r["outcome"] for r in mine["ranking"]}
        assert outcomes["LangGraph adds durable execution"] == "picked"
        assert outcomes["Funding round announced"] == "not_picked"
        assert outcomes["Heard yesterday"] == "aired"

    def test_the_reason_a_story_was_picked_is_recoverable(self, session, episode):
        history.record_ranking(session, episode.id, self.trace())
        today = datetime.now(ZoneInfo(get_settings().timezone)).date()
        mine = next(
            e for e in history.timeline(session, today)["episodes"] if e["id"] == str(episode.id)
        )
        picked = next(r for r in mine["ranking"] if r["outcome"] == "picked")
        assert picked["stack_match"] == ["langgraph"]
        assert picked["signals"]["stack"] == 3.0
        assert "dependency" in picked["why"]

    def test_the_llm_client_keeps_the_full_exchange(self, session, episode):
        client = LLMClient(session, EchoProvider(), episode_id=episode.id, durable_ledger=False)
        client.complete(prompt="the prompt text", system="the system text", stage=Stage.ranking)
        session.flush()
        row = session.execute(
            select(CostEntry).where(CostEntry.episode_id == episode.id)
        ).scalar_one()
        assert row.prompt == "the prompt text"
        assert row.system_prompt == "the system text"
        assert row.response
        assert row.stop_reason is not None or row.response

    def test_old_prompt_bodies_are_pruned_but_cost_and_tokens_are_kept(self, session, episode):
        old = CostEntry(
            episode_id=episode.id,
            stage="script",
            provider="anthropic",
            model="claude-sonnet-5-5",
            input_tokens=1200,
            output_tokens=300,
            cost_inr=1.5,
            prompt="a long prompt",
            response="a long answer",
            system_prompt="rules",
            created_at=datetime.now(UTC) - timedelta(days=200),
        )
        recent = CostEntry(
            episode_id=episode.id,
            stage="script",
            provider="anthropic",
            model="claude-sonnet-5-5",
            cost_inr=1.0,
            prompt="recent prompt",
            response="recent answer",
        )
        session.add_all([old, recent])
        session.flush()

        assert history.prune_llm_bodies(session, days=90) >= 1
        session.refresh(old)
        session.refresh(recent)
        assert old.prompt is None and old.response is None and old.system_prompt is None
        assert old.input_tokens == 1200 and float(old.cost_inr) == 1.5
        assert recent.prompt == "recent prompt"

    def test_export_is_typed_flat_records(self, session, episode):
        history.record_ranking(session, episode.id, self.trace())
        session.add(
            Interaction(
                channel="web",
                kind="qa",
                request={"question": "why?"},
                response={"answer": "because"},
            )
        )
        session.add(
            RunLog(kind="daily", trigger="schedule", status="ok", started_at=datetime.now(UTC))
        )
        session.flush()

        records = list(history.export_records(session, datetime.now(UTC) - timedelta(hours=1)))
        kinds = {r["type"] for r in records}
        assert {"ranking", "interaction", "run"} <= kinds
        # The database is shared with real use, so pick this test's row rather than the
        # first question anyone asked in the last hour.
        qa = next(
            r
            for r in records
            if r["type"] == "interaction" and (r.get("request") or {}).get("question") == "why?"
        )
        assert qa["kind"] == "qa"
        assert all("type" in r for r in records)


@pytest.mark.integration
class TestInteractionsPersist:
    def test_a_question_and_its_answer_survive_in_their_own_transaction(self):
        marker = f"marker-{uuid.uuid4().hex[:8]}"
        engine = get_engine()
        try:
            history.record_interaction(
                "qa",
                "web",
                request={"question": marker},
                response={"answer": "an answer"},
                latency_ms=12,
            )
            with Session(engine) as check:
                row = check.execute(
                    select(Interaction).where(Interaction.request["question"].astext == marker)
                ).scalar_one()
                assert row.response["answer"] == "an answer"
                assert row.channel == "web" and row.latency_ms == 12
        finally:
            with Session(engine) as cleanup:
                cleanup.query(Interaction).filter(
                    Interaction.request["question"].astext == marker
                ).delete(synchronize_session=False)
                cleanup.commit()

    def test_a_run_is_recorded_from_start_to_finish(self):
        engine = get_engine()
        run_id = history.start_run("daily", "schedule")
        try:
            history.finish_run(run_id, "ok", summary={"new_items": 5})
            with Session(engine) as check:
                run = check.get(RunLog, run_id)
                assert run.status == "ok"
                assert run.trigger == "schedule"
                assert run.summary["new_items"] == 5
                assert run.finished_at is not None
        finally:
            with Session(engine) as cleanup:
                cleanup.query(RunLog).filter(RunLog.id == run_id).delete()
                cleanup.commit()


class TestHistoryEndpoint:
    def test_the_history_route_returns_the_days_structure(self):
        from fastapi.testclient import TestClient

        from voicebrief.api.main import app

        response = TestClient(app).get("/history")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"date", "runs", "episodes", "interactions"}

    def test_an_unreadable_date_is_rejected(self):
        from fastapi.testclient import TestClient

        from voicebrief.api.main import app

        assert TestClient(app).get("/history", params={"date": "not-a-date"}).status_code == 422


@pytest.mark.integration
class TestInterruptedRuns:
    """A laptop that freezes mid-brief leaves its run and episode 'running' forever."""

    @pytest.fixture
    def cleanup(self):
        created: dict[str, list] = {"runs": [], "episodes": [], "users": []}
        yield created
        with Session(get_engine()) as s:
            if created["episodes"]:
                s.query(Episode).filter(Episode.id.in_(created["episodes"])).delete(
                    synchronize_session=False
                )
            if created["runs"]:
                s.query(RunLog).filter(RunLog.id.in_(created["runs"])).delete(
                    synchronize_session=False
                )
            if created["users"]:
                s.query(User).filter(User.id.in_(created["users"])).delete(
                    synchronize_session=False
                )
            s.commit()

    def test_stale_running_rows_are_closed_out_and_fresh_ones_left_alone(self, cleanup):
        with Session(get_engine()) as s:
            user = User(email=f"{uuid.uuid4().hex[:8]}@example.com")
            s.add(user)
            s.flush()
            old = datetime.now(UTC) - timedelta(hours=5)
            stale_run = RunLog(kind="daily", trigger="schedule", status="running", started_at=old)
            fresh_run = RunLog(
                kind="daily", trigger="cli", status="running", started_at=datetime.now(UTC)
            )
            stale_ep = Episode(
                user_id=user.id,
                mode=EpisodeMode.brief,
                status=EpisodeStatus.running,
                created_at=old,
            )
            fresh_ep = Episode(
                user_id=user.id, mode=EpisodeMode.brief, status=EpisodeStatus.running
            )
            s.add_all([stale_run, fresh_run, stale_ep, fresh_ep])
            s.commit()
            cleanup["users"].append(user.id)
            cleanup["runs"] += [stale_run.id, fresh_run.id]
            cleanup["episodes"] += [stale_ep.id, fresh_ep.id]

        runs, episodes = history.reap_interrupted()
        assert runs >= 1 and episodes >= 1

        with Session(get_engine()) as check:
            assert check.get(RunLog, cleanup["runs"][0]).status == "interrupted"
            assert "interrupted" in check.get(RunLog, cleanup["runs"][0]).error
            assert check.get(RunLog, cleanup["runs"][1]).status == "running"
            assert check.get(Episode, cleanup["episodes"][0]).status is EpisodeStatus.failed
            assert check.get(Episode, cleanup["episodes"][1]).status is EpisodeStatus.running

    def test_it_never_raises(self, monkeypatch):
        def broken():
            raise RuntimeError("database is down")

        monkeypatch.setattr(history, "session_scope", broken)
        assert history.reap_interrupted() == (0, 0)
