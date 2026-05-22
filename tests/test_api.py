"""API tests against the real database.

Uses FastAPI's TestClient with the docker-compose Postgres, because the interesting
behaviour is the interaction between routes and persisted state — 404s, ordering,
feedback upserts — and none of that is exercised by mocking the session.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from voicebrief.api.main import app
from voicebrief.db import session_scope
from voicebrief.db.models import (
    Episode,
    EpisodeMode,
    EpisodeStatus,
    Feedback,
    Language,
    Segment,
    User,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def episode():
    """A persisted episode with segments, cleaned up afterwards."""
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    with session_scope() as session:
        user = User(email=email, language=Language.en)
        session.add(user)
        session.flush()

        ep = Episode(
            user_id=user.id,
            mode=EpisodeMode.brief,
            status=EpisodeStatus.ready,
            title="Test episode",
            duration_seconds=120.0,
            cost_inr=3.5,
        )
        session.add(ep)
        session.flush()

        for i, (kind, start, end) in enumerate(
            [("cold_open", 0.0, 10.0), ("story", 10.5, 75.0), ("sign_off", 75.5, 120.0)]
        ):
            session.add(
                Segment(
                    episode_id=ep.id,
                    position=i,
                    kind=kind,
                    heading=f"Heading {i}",
                    script=f"Script for segment {i}.",
                    start_seconds=start,
                    end_seconds=end,
                    citations=[{"url": "https://example.com/a", "title": "Source A"}]
                    if kind == "story"
                    else [],
                )
            )
        session.flush()
        ids = (ep.id, user.id, email, [s.id for s in ep.segments])

    yield ids

    with session_scope() as session:
        ep = session.get(Episode, ids[0])
        if ep:
            session.delete(ep)
        user = session.get(User, ids[1])
        if user:
            session.delete(user)


class TestHealth:
    def test_reports_dependencies(self, client):
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert body["postgres"] is True
        assert "llm_provider" in body and "tts_engine" in body


class TestEpisodes:
    def test_list_returns_newest_first(self, client, episode):
        rows = client.get("/episodes?limit=5").json()
        assert rows
        timestamps = [r["created_at"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_detail_includes_segments_and_citations(self, client, episode):
        episode_id, *_ = episode
        body = client.get(f"/episodes/{episode_id}").json()
        assert body["title"] == "Test episode"
        assert len(body["segments"]) == 3
        story = next(s for s in body["segments"] if s["kind"] == "story")
        assert story["citations"][0]["url"] == "https://example.com/a"

    def test_segments_are_ordered_by_position(self, client, episode):
        episode_id, *_ = episode
        positions = [s["position"] for s in client.get(f"/episodes/{episode_id}").json()["segments"]]
        assert positions == sorted(positions)

    def test_missing_episode_is_404(self, client):
        assert client.get(f"/episodes/{uuid.uuid4()}").status_code == 404

    def test_malformed_id_is_422(self, client):
        assert client.get("/episodes/not-a-uuid").status_code == 422


class TestTranscript:
    def test_timestamps_are_formatted_from_seconds(self, client, episode):
        episode_id, *_ = episode
        lines = client.get(f"/episodes/{episode_id}/transcript").json()
        assert lines[0]["timestamp"] == "00:00"
        assert lines[1]["timestamp"] == "00:10"
        assert lines[2]["timestamp"] == "01:15"

    def test_lines_carry_citations(self, client, episode):
        episode_id, *_ = episode
        lines = client.get(f"/episodes/{episode_id}/transcript").json()
        story = next(line for line in lines if line["citations"])
        assert story["citations"][0]["url"] == "https://example.com/a"

    def test_missing_episode_is_404(self, client):
        assert client.get(f"/episodes/{uuid.uuid4()}/transcript").status_code == 404


class TestFeedback:
    def test_records_a_vote(self, client, episode):
        _, _, email, segment_ids = episode
        response = client.post(
            "/feedback",
            params={"email": email},
            json={"segment_id": str(segment_ids[1]), "vote": 1},
        )
        assert response.status_code == 204

        with session_scope() as session:
            row = session.query(Feedback).filter_by(segment_id=segment_ids[1]).one()
            assert row.vote == 1

    def test_second_vote_updates_rather_than_duplicates(self, client, episode):
        _, _, email, segment_ids = episode
        for vote in (1, -1):
            client.post(
                "/feedback",
                params={"email": email},
                json={"segment_id": str(segment_ids[1]), "vote": vote},
            )
        with session_scope() as session:
            rows = session.query(Feedback).filter_by(segment_id=segment_ids[1]).all()
            assert len(rows) == 1
            assert rows[0].vote == -1

    def test_invalid_vote_is_rejected(self, client, episode):
        _, _, email, segment_ids = episode
        response = client.post(
            "/feedback",
            params={"email": email},
            json={"segment_id": str(segment_ids[1]), "vote": 5},
        )
        assert response.status_code == 422

    def test_unknown_segment_is_404(self, client, episode):
        _, _, email, _ = episode
        response = client.post(
            "/feedback",
            params={"email": email},
            json={"segment_id": str(uuid.uuid4()), "vote": 1},
        )
        assert response.status_code == 404


class TestGenerate:
    def test_accepts_and_returns_a_stream_url(self, client, monkeypatch):
        """The route must return immediately — generation takes minutes and a
        blocking request dies to a proxy timeout."""
        import voicebrief.api.main as api

        monkeypatch.setattr(api, "_run_generation", lambda *a, **k: None)
        response = client.post(
            "/episodes/generate",
            json={"email": f"gen-{uuid.uuid4().hex[:6]}@example.com", "render_audio": False},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"
        assert body["stream_url"].endswith("/stream")

    def test_rejects_out_of_range_story_count(self, client):
        response = client.post("/episodes/generate", json={"max_stories": 99})
        assert response.status_code == 422


class TestSearch:
    def test_empty_query_is_rejected(self, client):
        assert client.get("/search?q=a").status_code == 422

    def test_returns_a_response_shape_even_with_no_matches(self, client):
        body = client.get("/search?q=zzzzz unlikely query zzzzz").json()
        assert body["query"]
        assert isinstance(body["hits"], list)


class TestSegmentChat:
    def test_unknown_segment_is_404(self, client):
        response = client.post(
            f"/segments/{uuid.uuid4()}/chat", json={"question": "what?"}
        )
        assert response.status_code == 404

    def test_empty_question_without_preset_is_rejected(self, client, episode):
        _, _, _, segment_ids = episode
        response = client.post(f"/segments/{segment_ids[1]}/chat", json={"question": "  "})
        assert response.status_code == 422
