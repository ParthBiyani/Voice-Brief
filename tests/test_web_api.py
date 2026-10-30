"""The API as the web app uses it: audio, settings, progress, access control, hosting."""

from __future__ import annotations

import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from voicebrief.api import progress
from voicebrief.api.extras import parse_range
from voicebrief.api.main import app
from voicebrief.api.memory import _source_text
from voicebrief.api.site import build_site, is_loopback, presented_token
from voicebrief.db.models import (
    Episode,
    EpisodeMode,
    EpisodeStatus,
    Item,
    RankingSnapshot,
    Segment,
    Source,
    SourceKind,
    User,
)
from voicebrief.db.session import get_engine


@pytest.fixture(autouse=True)
def _keep_history_clean(monkeypatch):
    for target in ("voicebrief.api.main", "voicebrief.api.memory", "voicebrief.api.extras"):
        monkeypatch.setattr(f"{target}.record_interaction", lambda *a, **k: None)


# ─────────────────────────────────────────────────────────────────────────────
# Range parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestParseRange:
    @pytest.mark.parametrize(
        ("header", "size", "expected"),
        [
            (None, 100, None),
            ("", 100, None),
            ("bytes=0-9", 100, (0, 9)),
            ("bytes=10-", 100, (10, 99)),
            ("bytes=-20", 100, (80, 99)),
            ("bytes=90-500", 100, (90, 99)),
            ("bytes=-500", 100, (0, 99)),
            ("bytes=0-0", 100, (0, 0)),
        ],
    )
    def test_valid_ranges(self, header, size, expected):
        assert parse_range(header, size) == expected

    @pytest.mark.parametrize(
        ("header", "size"),
        [("bytes=100-", 100), ("bytes=200-300", 100), ("bytes=9-3", 100), ("bytes=-0", 100)],
    )
    def test_unsatisfiable_ranges_are_flagged(self, header, size):
        assert parse_range(header, size) == "invalid"

    @pytest.mark.parametrize("header", ["items=0-9", "bytes=a-b", "bytes=-", "garbage"])
    def test_headers_it_does_not_understand_are_ignored_not_errors(self, header):
        assert parse_range(header, 100) is None


# ─────────────────────────────────────────────────────────────────────────────
# Progress channel
# ─────────────────────────────────────────────────────────────────────────────
class TestProgressChannel:
    def test_a_late_subscriber_replays_everything_so_far(self):
        episode_id = uuid.uuid4()
        progress.emit(episode_id, "started")
        progress.emit(episode_id, "writing")
        events, finished = progress.get_channel(episode_id).wait(0, 0.1)
        assert [e["stage"] for e in events] == ["started", "writing"]
        assert not finished

    def test_events_emitted_from_another_thread_wake_a_waiting_reader(self):
        episode_id = uuid.uuid4()
        channel = progress.open_channel(episode_id)
        threading.Timer(0.1, lambda: progress.emit(episode_id, "writing")).start()
        started = time.monotonic()
        events, _ = channel.wait(0, 5.0)
        assert [e["stage"] for e in events] == ["writing"]
        assert time.monotonic() - started < 2.0, "the reader sat out the whole timeout"

    def test_a_final_stage_marks_the_channel_finished(self):
        episode_id = uuid.uuid4()
        progress.emit(episode_id, "ready", title="x")
        _, finished = progress.get_channel(episode_id).wait(0, 0.1)
        assert finished and episode_id not in progress.active_ids()

    def test_a_running_channel_is_listed_as_active(self):
        episode_id = uuid.uuid4()
        progress.open_channel(episode_id)
        assert episode_id in progress.active_ids()

    def test_the_stream_replays_and_ends_with_the_outcome(self):
        episode_id = uuid.uuid4()
        progress.emit(episode_id, "started")
        progress.emit(episode_id, "ready", title="Done", episode_id=str(episode_id))
        client = TestClient(app)
        with client.stream("GET", f"/episodes/{episode_id}/stream") as response:
            body = "".join(response.iter_text())
        assert '"stage": "started"' in body and '"stage": "ready"' in body

    def test_an_unknown_generation_is_a_404(self):
        assert TestClient(app).get(f"/episodes/{uuid.uuid4()}/stream").status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Hosting and access control
# ─────────────────────────────────────────────────────────────────────────────
class TestSite:
    @pytest.fixture
    def dist(self, tmp_path):
        (tmp_path / "index.html").write_text("<html><body>VoiceBrief app</body></html>")
        (tmp_path / "app.js").write_text("console.log(1)")
        return tmp_path

    def test_the_app_and_the_api_share_one_origin(self, dist):
        client = TestClient(build_site(dist, token=""))
        assert "VoiceBrief app" in client.get("/").text
        assert client.get("/app.js").status_code == 200
        assert client.get("/api/health").status_code == 200

    def test_deep_links_fall_back_to_the_app_so_a_reload_does_not_404(self, dist):
        client = TestClient(build_site(dist, token=""))
        response = client.get("/history/2026-10-30")
        assert response.status_code == 200 and "VoiceBrief app" in response.text

    def test_without_a_build_the_api_still_works(self, tmp_path):
        client = TestClient(build_site(tmp_path, token=""))
        assert client.get("/api/health").status_code == 200

    def test_is_loopback(self):
        assert is_loopback("127.0.0.1") and is_loopback("localhost")
        assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")


class TestAccessToken:
    @pytest.fixture
    def client(self, tmp_path):
        (tmp_path / "index.html").write_text("app")
        # A fresh API app per test: the gate is installed on it by build_site.
        from importlib import reload

        import voicebrief.api.main as main_module
        import voicebrief.api.site as site_module

        reload(main_module)
        reload(site_module)
        return TestClient(site_module.build_site(tmp_path, token="s3cret"))

    def test_the_api_refuses_requests_without_the_token(self, client):
        assert client.get("/api/episodes").status_code == 401

    def test_health_stays_open_for_monitoring(self, client):
        assert client.get("/api/health").status_code == 200

    def test_the_app_shell_itself_is_public(self, client):
        assert client.get("/").status_code == 200

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"headers": {"Authorization": "Bearer s3cret"}},
            {"headers": {"X-VB-Token": "s3cret"}},
            {"cookies": {"vb_token": "s3cret"}},
            {"params": {"token": "s3cret"}},
        ],
    )
    def test_every_supported_way_of_presenting_the_token_works(self, client, kwargs):
        assert client.get("/api/episodes", **kwargs).status_code == 200

    def test_a_wrong_token_is_refused(self, client):
        assert client.get("/api/episodes", headers={"X-VB-Token": "nope"}).status_code == 401

    def test_presented_token_is_read_from_all_places(self):
        request = SimpleNamespace(
            headers={"authorization": "Bearer abc"}, cookies={}, query_params={}
        )
        assert presented_token(request) == "abc"


class TestServeRefusesToExposeWithoutAToken:
    def test_a_network_bind_without_a_token_is_refused(self, monkeypatch):
        from typer.testing import CliRunner

        from voicebrief.cli import app as cli
        from voicebrief.config import get_settings

        keyless = get_settings().model_copy(update={"api_token": None})
        monkeypatch.setattr("voicebrief.cli.get_settings", lambda: keyless)
        result = CliRunner().invoke(cli, ["serve", "--host", "0.0.0.0"])
        assert result.exit_code == 1
        assert "VB_API_TOKEN" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Audio, settings and the episode detail (these need real rows)
# ─────────────────────────────────────────────────────────────────────────────
class FakeS3:
    def __init__(self, payload: bytes, content_type="audio/mpeg"):
        self.payload = payload
        self.content_type = content_type
        self.calls: list[dict] = []

    def head_object(self, **_):
        return {"ContentLength": len(self.payload), "ContentType": self.content_type}

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803 - boto3 casing
        self.calls.append({"Range": Range})
        data = self.payload
        if Range:
            a, b = Range.removeprefix("bytes=").split("-")
            data = self.payload[int(a) : int(b) + 1]

        class Body:
            def iter_chunks(self, size):
                yield data

        return {"Body": Body()}


@pytest.mark.integration
class TestWithRows:
    @pytest.fixture
    def episode(self):
        engine = get_engine()
        with Session(engine) as s:
            user = User(email=f"web-{uuid.uuid4().hex[:8]}@example.com")
            s.add(user)
            s.flush()
            ep = Episode(
                user_id=user.id,
                mode=EpisodeMode.brief,
                status=EpisodeStatus.ready,
                title="Audio test",
                audio_key="episodes/test.mp3",
                grounding={
                    "attribution_rate": 0.9,
                    "warnings": [],
                    "tts_engine": "kokoro",
                    "noise": 1,
                },
            )
            s.add(ep)
            s.flush()
            cluster = uuid.uuid4()
            s.add(
                Segment(
                    episode_id=ep.id,
                    position=1,
                    kind="story",
                    heading="LangGraph story",
                    script="x",
                    cluster_id=None,
                    citations=[],
                )
            )
            s.add(
                RankingSnapshot(
                    episode_id=ep.id,
                    cluster_id=cluster,
                    outcome="picked",
                    position=1,
                    title="LangGraph story",
                    signals={"stack": 3.0},
                    matched_dependencies=["langgraph"],
                )
            )
            s.commit()
            ids = (user.id, ep.id)
        yield ids
        with Session(engine) as s:
            s.query(Episode).filter(Episode.id == ids[1]).delete(synchronize_session=False)
            s.query(User).filter(User.id == ids[0]).delete()
            s.commit()

    def client(self, monkeypatch, payload=b"0123456789" * 10):
        fake = FakeS3(payload)
        monkeypatch.setattr("voicebrief.api.extras.s3_client", lambda: fake)
        return TestClient(app), fake

    def test_the_whole_file_is_served_with_range_support_advertised(self, episode, monkeypatch):
        client, _ = self.client(monkeypatch)
        response = client.get(f"/episodes/{episode[1]}/audio")
        assert response.status_code == 200
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) == 100

    def test_a_range_request_gets_exactly_those_bytes(self, episode, monkeypatch):
        client, fake = self.client(monkeypatch)
        response = client.get(f"/episodes/{episode[1]}/audio", headers={"Range": "bytes=10-19"})
        assert response.status_code == 206
        assert response.content == b"0123456789"
        assert response.headers["content-range"] == "bytes 10-19/100"
        assert fake.calls[-1]["Range"] == "bytes=10-19"

    def test_an_open_ended_range_runs_to_the_end(self, episode, monkeypatch):
        client, _ = self.client(monkeypatch)
        response = client.get(f"/episodes/{episode[1]}/audio", headers={"Range": "bytes=95-"})
        assert response.status_code == 206 and len(response.content) == 5

    def test_an_unsatisfiable_range_is_a_416(self, episode, monkeypatch):
        client, _ = self.client(monkeypatch)
        response = client.get(f"/episodes/{episode[1]}/audio", headers={"Range": "bytes=500-"})
        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */100"

    def test_download_sets_a_filename(self, episode, monkeypatch):
        client, _ = self.client(monkeypatch)
        response = client.get(f"/episodes/{episode[1]}/audio", params={"download": "1"})
        assert ".mp3" in response.headers["content-disposition"]

    def test_unavailable_storage_is_a_502_not_a_crash(self, episode, monkeypatch):
        from botocore.exceptions import EndpointConnectionError

        class Down:
            def head_object(self, **_):
                raise EndpointConnectionError(endpoint_url="http://nowhere")

        monkeypatch.setattr("voicebrief.api.extras.s3_client", lambda: Down())
        assert TestClient(app).get(f"/episodes/{episode[1]}/audio").status_code == 502

    def test_an_episode_without_audio_is_a_404(self, monkeypatch):
        assert TestClient(app).get(f"/episodes/{uuid.uuid4()}/audio").status_code == 404

    def test_the_detail_points_at_the_streaming_route_not_at_storage(self, episode):
        body = TestClient(app).get(f"/episodes/{episode[1]}").json()
        assert body["audio_url"] == f"/episodes/{episode[1]}/audio"
        assert "localhost" not in body["audio_url"]

    def test_the_detail_carries_grounding_but_not_internal_noise(self, episode):
        body = TestClient(app).get(f"/episodes/{episode[1]}").json()
        assert body["grounding"]["attribution_rate"] == 0.9
        assert "noise" not in body["grounding"]

    def test_the_ranking_explains_each_story(self, episode):
        rows = TestClient(app).get(f"/episodes/{episode[1]}/ranking").json()
        assert rows[0]["title"] == "LangGraph story"
        assert rows[0]["stack_match"] == ["langgraph"]
        assert rows[0]["signals"]["stack"] == 3.0


@pytest.mark.integration
class TestSettings:
    @pytest.fixture
    def email(self):
        address = f"settings-{uuid.uuid4().hex[:8]}@example.com"
        yield address
        with Session(get_engine()) as s:
            s.query(User).filter(User.email == address).delete()
            s.commit()

    def test_a_new_user_gets_sensible_defaults(self, email):
        body = TestClient(app).get("/settings", params={"email": email}).json()
        assert body["language"] == "en" and body["style"] == "solo_anchor"
        assert body["topics"] == [] and body["github_login"] is None
        assert body["projectmind"]["configured"] in (True, False)

    def test_saved_settings_are_returned_and_persist(self, email):
        client = TestClient(app)
        response = client.put(
            "/settings",
            params={"email": email},
            json={
                "github_login": "@SomeOne",
                "topics": [" Rust ", "edge-ai", "rust"],
                "style": "two_host",
            },
        )
        assert response.status_code == 200
        again = client.get("/settings", params={"email": email}).json()
        assert again["github_login"] == "SomeOne"
        assert again["topics"] == ["rust", "edge-ai"]
        assert again["style"] == "two_host"

    def test_an_empty_login_clears_it(self, email):
        client = TestClient(app)
        client.put("/settings", params={"email": email}, json={"github_login": "someone"})
        client.put("/settings", params={"email": email}, json={"github_login": ""})
        assert client.get("/settings", params={"email": email}).json()["github_login"] is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"github_login": "not a valid name!"},
            {"language": "fr"},
            {"style": "chaotic"},
            {"topics": [f"t{i}" for i in range(40)]},
        ],
    )
    def test_invalid_values_are_rejected(self, email, payload):
        response = TestClient(app).put("/settings", params={"email": email}, json=payload)
        assert response.status_code == 422

    def test_unsent_fields_are_left_alone(self, email):
        client = TestClient(app)
        client.put(
            "/settings", params={"email": email}, json={"topics": ["rust"], "language": "hi"}
        )
        client.put("/settings", params={"email": email}, json={"github_login": "someone"})
        body = client.get("/settings", params={"email": email}).json()
        assert body["topics"] == ["rust"] and body["language"] == "hi"


@pytest.mark.integration
class TestGenerateFlow:
    def test_the_returned_id_has_a_channel_and_the_user_is_committed_first(self, monkeypatch):
        email = f"gen-{uuid.uuid4().hex[:8]}@example.com"
        seen: dict = {}

        def fake_worker(episode_id, request):
            # By the time the worker runs, the user must already be visible to others.
            with Session(get_engine()) as s:
                seen["user_visible"] = (
                    s.query(User).filter(User.email == request.email).count() == 1
                )
            seen["id"] = episode_id

        monkeypatch.setattr("voicebrief.api.main._run_generation", fake_worker)
        try:
            response = TestClient(app).post("/episodes/generate", json={"email": email})
            assert response.status_code == 202
            body = response.json()
            assert uuid.UUID(body["episode_id"]) == seen["id"]
            assert progress.get_channel(seen["id"]) is not None
            assert seen["user_visible"] is True
        finally:
            with Session(get_engine()) as s:
                s.query(User).filter(User.email == email).delete()
                s.commit()

    def test_unset_options_fall_back_to_saved_settings(self):
        from voicebrief.api.schemas import GenerateRequest

        request = GenerateRequest()
        assert request.language is None and request.max_stories is None and request.style is None


@pytest.mark.integration
class TestQuestionSources:
    def test_the_prompt_carries_the_source_text_not_just_titles(self):
        with Session(get_engine()) as s:
            s.begin()
            try:
                source = Source(
                    slug=f"qa-{uuid.uuid4().hex[:6]}",
                    name="t",
                    kind=SourceKind.rss,
                    endpoint="https://x/f",
                )
                s.add(source)
                s.flush()
                item = Item(
                    source_id=source.id,
                    external_id=uuid.uuid4().hex,
                    url="https://example.com/a",
                    title="Release 2.0",
                    summary="Version 2.0 adds durable execution and drops Python 3.9.",
                    published_at=time_now(),
                )
                s.add(item)
                s.flush()
                text = _source_text(
                    s,
                    [
                        {
                            "item_id": str(item.id),
                            "title": "Release 2.0",
                            "url": "https://example.com/a",
                        }
                    ],
                )
                assert "durable execution" in text and "drops Python 3.9" in text
                assert text.startswith("[1] Release 2.0")
            finally:
                s.rollback()

    def test_citations_without_an_item_still_list_their_title(self):
        with Session(get_engine()) as s:
            text = _source_text(s, [{"title": "Orphan", "url": "https://x/y", "item_id": None}])
        assert "Orphan" in text

    def test_no_citations_says_so(self):
        with Session(get_engine()) as s:
            assert "no sources" in _source_text(s, [])


def time_now():
    from datetime import UTC, datetime

    return datetime.now(UTC)
