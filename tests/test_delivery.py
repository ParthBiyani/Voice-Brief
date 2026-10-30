"""Delivery: rendering, the Slack and SMTP protocols, and the guarantees around them
(never raises, delivers once, records failures, silent until configured)."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db.models import Episode, EpisodeMode, EpisodeStatus, RankingSnapshot, Segment, User
from voicebrief.db.session import get_engine
from voicebrief.delivery import deliver, mail
from voicebrief.delivery.digest import (
    Digest,
    DigestStory,
    build_digest,
    first_sentences,
    to_html,
    to_slack_blocks,
    to_text,
)
from voicebrief.delivery.slack import API, SlackClient, SlackError


def digest(**over) -> Digest:
    base = {
        "title": "LangGraph adds durable execution",
        "date_label": "Fri 30 Oct",
        "status": "ready",
        "duration_minutes": 6.2,
        "cost_inr": 5.69,
        "sourced": 0.9,
        "stories": [
            DigestStory(
                heading="LangGraph 1.3 adds durable execution",
                summary="Checkpoints now survive restarts. It replaces the workaround.",
                links=[
                    ("Release notes", "https://github.com/langchain-ai/langgraph/releases/v1.3")
                ],
                matched=["langgraph"],
                personal=True,
            ),
            DigestStory(heading="A generic story", summary="Something happened."),
        ],
        "web_url": "http://192.168.1.20:8000/",
    }
    return Digest(**{**base, **over})


class TestSummaries:
    def test_short_text_is_left_alone(self):
        assert first_sentences("One. Two.") == "One. Two."

    def test_long_text_is_cut_on_a_sentence_boundary(self):
        text = " ".join(f"This is sentence number {i} of the script." for i in range(30))
        out = first_sentences(text, limit=120)
        assert out.endswith(".") and len(out) <= 160
        assert "sentence number 0" in out

    def test_one_enormous_sentence_is_truncated_with_an_ellipsis(self):
        out = first_sentences("word " * 400, limit=100)
        assert out.endswith("…") and len(out) <= 110

    def test_empty_text(self):
        assert first_sentences("") == ""


class TestRendering:
    def test_slack_blocks_lead_with_a_header_and_the_stats(self):
        blocks = to_slack_blocks(digest())
        assert blocks[0]["type"] == "header"
        assert "VoiceBrief" in blocks[0]["text"]["text"]
        assert "6 min" in blocks[1]["elements"][0]["text"]
        assert "90% of claims traced" in blocks[1]["elements"][0]["text"]

    def test_each_story_is_a_section_with_its_link_and_a_personal_note(self):
        blocks = to_slack_blocks(digest())
        text = blocks[2]["text"]["text"]
        assert "*LangGraph 1.3 adds durable execution*" in text
        assert ":dart:" in text and "your stack: langgraph" in text
        assert "<https://github.com/langchain-ai/langgraph/releases/v1.3|Release notes>" in text
        assert ":dart:" not in blocks[3]["text"]["text"]

    def test_the_app_link_is_the_last_block(self):
        blocks = to_slack_blocks(digest())
        assert "192.168.1.20:8000" in blocks[-1]["elements"][0]["text"]

    def test_no_link_block_without_a_public_url(self):
        blocks = to_slack_blocks(digest(web_url=None))
        assert all("Open the transcript" not in json.dumps(b) for b in blocks)

    def test_a_degraded_brief_says_so_up_front(self):
        blocks = to_slack_blocks(digest(status="degraded", issues=["audio was not stored"]))
        assert ":warning:" in blocks[2]["text"]["text"]
        assert "audio was not stored" in blocks[2]["text"]["text"]

    def test_slack_markup_in_a_title_cannot_break_the_message(self):
        evil = digest(stories=[DigestStory(heading="A <b> & C", summary="x < y > z")])
        text = to_slack_blocks(evil)[2]["text"]["text"]
        assert "&lt;b&gt;" in text and "&amp;" in text and "x &lt; y &gt; z" in text

    def test_block_count_stays_within_slacks_limit(self):
        many = digest(stories=[DigestStory(heading=f"s{i}", summary="x") for i in range(80)])
        assert len(to_slack_blocks(many)) <= 50

    def test_plain_text_carries_everything(self):
        text = to_text(digest())
        assert "LangGraph 1.3 adds durable execution" in text
        assert "your stack: langgraph" in text and "192.168.1.20" in text

    def test_html_escapes_content(self):
        evil = digest(stories=[DigestStory(heading="<script>x</script>", summary="a & b")])
        out = to_html(evil)
        assert "<script>" not in out and "&lt;script&gt;" in out and "a &amp; b" in out


# ─────────────────────────────────────────────────────────────────────────────
# Slack protocol
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def slack():
    return SlackClient("xoxb-test-token", "C123")


class TestSlack:
    @respx.mock
    def test_a_message_goes_to_the_channel_with_the_bot_token(self, slack):
        route = respx.post(f"{API}/chat.postMessage").mock(
            return_value=httpx.Response(200, json={"ok": True, "ts": "111.222"})
        )
        ts = slack.post_message("hello", [{"type": "divider"}])
        assert ts == "111.222"
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer xoxb-test-token"
        body = json.loads(request.content)
        assert body["channel"] == "C123" and body["blocks"] == [{"type": "divider"}]
        assert body["unfurl_links"] is False

    @respx.mock
    def test_slacks_no_becomes_a_clear_error(self, slack):
        respx.post(f"{API}/chat.postMessage").mock(
            return_value=httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
        )
        with pytest.raises(SlackError, match="channel_not_found"):
            slack.post_message("hello")

    @respx.mock
    def test_a_rate_limit_is_waited_out_once(self, slack, monkeypatch):
        slept = []
        monkeypatch.setattr("voicebrief.delivery.slack.time.sleep", slept.append)
        respx.post(f"{API}/chat.postMessage").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "3"}),
                httpx.Response(200, json={"ok": True, "ts": "1.1"}),
            ]
        )
        assert slack.post_message("hello") == "1.1"
        assert slept == [3]

    @respx.mock
    def test_files_use_the_current_three_step_upload_into_the_thread(self, slack):
        start = respx.post(f"{API}/files.getUploadURLExternal").mock(
            return_value=httpx.Response(
                200,
                json={
                    "ok": True,
                    "upload_url": "https://files.slack.com/upload/abc",
                    "file_id": "F9",
                },
            )
        )
        put = respx.post("https://files.slack.com/upload/abc").mock(
            return_value=httpx.Response(200)
        )
        done = respx.post(f"{API}/files.completeUploadExternal").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )

        file_id = slack.upload_file(b"audio-bytes", "brief.mp3", thread_ts="111.222", title="Brief")

        assert file_id == "F9"
        assert b"filename=brief.mp3" in start.calls[0].request.content
        assert b"length=11" in start.calls[0].request.content
        assert put.calls[0].request.content == b"audio-bytes"
        finish = json.loads(done.calls[0].request.content)
        assert finish["channel_id"] == "C123" and finish["thread_ts"] == "111.222"
        assert finish["files"] == [{"id": "F9", "title": "Brief"}]

    @respx.mock
    def test_the_digest_is_posted_then_the_audio_lands_in_its_thread(self, slack):
        respx.post(f"{API}/chat.postMessage").mock(
            return_value=httpx.Response(200, json={"ok": True, "ts": "5.5"})
        )
        respx.post(f"{API}/files.getUploadURLExternal").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "upload_url": "https://files.slack.com/u", "file_id": "F1"}
            )
        )
        respx.post("https://files.slack.com/u").mock(return_value=httpx.Response(200))
        finish = respx.post(f"{API}/files.completeUploadExternal").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        result = slack.send_digest(digest(), b"mp3", "brief.mp3")
        assert result == {"ts": "5.5", "audio": True}
        assert json.loads(finish.calls[0].request.content)["thread_ts"] == "5.5"

    @respx.mock
    def test_a_failed_audio_upload_still_delivers_the_digest_and_says_so(self, slack):
        post = respx.post(f"{API}/chat.postMessage").mock(
            return_value=httpx.Response(200, json={"ok": True, "ts": "5.5"})
        )
        respx.post(f"{API}/files.getUploadURLExternal").mock(
            return_value=httpx.Response(200, json={"ok": False, "error": "missing_scope"})
        )
        result = slack.send_digest(digest(), b"mp3", "brief.mp3")
        assert result["ts"] == "5.5" and result["audio"] is False
        assert "missing_scope" in result["audio_error"]
        assert post.call_count == 2  # the digest, then a note that the audio is missing

    @respx.mock
    def test_without_audio_only_the_message_is_sent(self, slack):
        respx.post(f"{API}/chat.postMessage").mock(
            return_value=httpx.Response(200, json={"ok": True, "ts": "5.5"})
        )
        assert slack.send_digest(digest(), None, "x.mp3") == {"ts": "5.5", "audio": False}

    @respx.mock
    def test_a_failure_notice_says_what_went_wrong(self, slack):
        route = respx.post(f"{API}/chat.postMessage").mock(
            return_value=httpx.Response(200, json={"ok": True, "ts": "9.9"})
        )
        slack.send_failure("no fresh stories to cover")
        assert "no fresh stories" in route.calls[0].request.content.decode()


# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────
class FakeSMTP:
    instances: ClassVar[list[FakeSMTP]] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.ops, self.sent = host, port, [], None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        self.ops.append("ehlo")

    def starttls(self):
        self.ops.append("starttls")

    def login(self, user, password):
        self.ops.append(("login", user, password))

    def send_message(self, message):
        self.sent = message
        self.ops.append("send")


class TestEmail:
    def message(self, audio, name="b.mp3"):
        return mail.build_message(
            digest(),
            sender="me@example.com",
            recipient="you@example.com",
            audio=audio,
            filename=name,
        )

    def test_it_carries_text_html_and_the_audio(self):
        message = self.message(b"x" * 1000)
        kinds = [part.get_content_type() for part in message.walk()]
        assert "text/plain" in kinds and "text/html" in kinds and "audio/mpeg" in kinds
        assert message["Subject"].startswith("VoiceBrief, Fri 30 Oct")
        assert message["To"] == "you@example.com"

    def test_audio_too_big_for_a_mailbox_is_left_off_with_a_note(self):
        message = self.message(b"x" * (mail.MAX_ATTACHMENT_BYTES + 1))
        assert "audio/mpeg" not in [part.get_content_type() for part in message.walk()]
        assert "too large to attach" in message.get_body(("plain",)).get_content()

    def test_no_audio_is_fine(self):
        assert self.message(None)["Subject"]

    @pytest.fixture(autouse=True)
    def _smtp(self, monkeypatch):
        FakeSMTP.instances.clear()
        monkeypatch.setattr("voicebrief.delivery.mail.smtplib.SMTP", FakeSMTP)
        monkeypatch.setattr("voicebrief.delivery.mail.smtplib.SMTP_SSL", FakeSMTP)

    def test_port_587_uses_starttls_and_logs_in(self):
        mail.send(self.message(None), host="smtp.example.com", port=587, user="u", password="p")
        server = FakeSMTP.instances[0]
        assert server.ops == ["ehlo", "starttls", "ehlo", ("login", "u", "p"), "send"]

    def test_port_465_is_implicit_tls_without_starttls(self):
        mail.send(self.message(None), host="smtp.example.com", port=465, user="u", password="p")
        assert "starttls" not in FakeSMTP.instances[0].ops
        assert FakeSMTP.instances[0].sent is not None

    def test_no_credentials_means_no_login(self):
        mail.send(self.message(None), host="relay", port=25, user=None, password=None)
        assert not any(isinstance(op, tuple) for op in FakeSMTP.instances[0].ops)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def configure(monkeypatch, **update):
    settings = get_settings().model_copy(update=update)
    monkeypatch.setattr("voicebrief.delivery.deliver.get_settings", lambda: settings)
    return settings


class RecordingSlack:
    def __init__(self, fail=False):
        self.fail, self.sent, self.failures = fail, [], []

    def send_digest(self, digest, audio, filename):
        if self.fail:
            raise SlackError("channel_not_found")
        self.sent.append((digest.title, audio, filename))
        return {"ts": "1.1", "audio": audio is not None}

    def send_failure(self, message):
        if self.fail:
            raise SlackError("invalid_auth")
        self.failures.append(message)


class TestNothingConfigured:
    def test_it_is_silent_and_does_nothing(self, monkeypatch):
        configure(monkeypatch, slack_bot_token=None, slack_channel_id=None, smtp_host=None)
        assert deliver.configured_channels() == []
        assert deliver.deliver_episode(uuid.uuid4()) == {}
        assert deliver.notify_failure("x") == {}

    def test_both_slack_settings_are_needed(self, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id=None, smtp_host=None)
        assert deliver.configured_channels() == []

    def test_email_needs_host_sender_and_recipient(self, monkeypatch):
        configure(monkeypatch, smtp_host="h", mail_from="a@b", email_to=None, slack_bot_token=None)
        assert not deliver.email_configured()
        configure(monkeypatch, smtp_host="h", mail_from="a@b", email_to="c@d", slack_bot_token=None)
        assert deliver.email_configured()


class TestNeverRaises:
    def test_a_crash_while_building_the_digest_is_contained(self, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id="C1")
        monkeypatch.setattr(deliver, "slack_client", lambda: RecordingSlack())

        def boom(*a, **k):
            raise RuntimeError("database exploded")

        monkeypatch.setattr(deliver, "build_digest", boom)
        assert deliver.deliver_episode(uuid.uuid4()) == {}

    def test_a_failing_failure_notice_is_contained_too(self, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id="C1")
        monkeypatch.setattr(deliver, "slack_client", lambda: RecordingSlack(fail=True))
        result = deliver.notify_failure("something broke")
        assert result["slack"]["ok"] is False and "invalid_auth" in result["slack"]["error"]


@pytest.mark.integration
class TestWithAnEpisode:
    @pytest.fixture
    def episode_id(self):
        engine = get_engine()
        with Session(engine) as s:
            user = User(email=f"deliver-{uuid.uuid4().hex[:8]}@example.com")
            s.add(user)
            s.flush()
            cluster = uuid.uuid4()
            ep = Episode(
                user_id=user.id,
                mode=EpisodeMode.brief,
                status=EpisodeStatus.ready,
                title="Delivery test brief",
                duration_seconds=372.0,
                cost_inr=5.5,
                audio_key="episodes/test.mp3",
                grounding={"attribution_rate": 0.9},
            )
            s.add(ep)
            s.flush()
            s.add(
                Segment(
                    episode_id=ep.id,
                    position=1,
                    kind="story",
                    heading="LangGraph story",
                    script="LangGraph adds durable execution. It replaces the checkpoint workaround.",
                    cluster_id=None,
                    citations=[
                        {"url": "https://example.com/a", "title": "Release notes"},
                        {"url": "https://example.com/b", "title": "Blog"},
                        {"url": "https://example.com/c", "title": "Third"},
                    ],
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
        yield ids[1]
        with Session(engine) as s:
            s.query(Episode).filter(Episode.id == ids[1]).delete(synchronize_session=False)
            s.query(User).filter(User.id == ids[0]).delete()
            s.commit()

    def stored(self, episode_id):
        with Session(get_engine()) as s:
            return (s.get(Episode, episode_id).grounding or {}).get("delivery", {})

    def test_the_digest_is_built_from_the_stored_episode(self, episode_id):
        with Session(get_engine()) as s:
            built = build_digest(s, episode_id, base_url="http://phone.local:8000")
        assert built.title == "Delivery test brief"
        assert built.sourced == 0.9 and round(built.duration_minutes) == 6
        assert built.web_url == "http://phone.local:8000/"
        assert len(built.stories) == 1 and len(built.stories[0].links) == 2  # capped
        assert built.stories[0].summary.startswith("LangGraph adds durable execution")

    def test_delivery_is_sent_and_recorded_on_the_episode(self, episode_id, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id="C1", smtp_host=None)
        fake = RecordingSlack()
        monkeypatch.setattr(deliver, "slack_client", lambda: fake)
        monkeypatch.setattr(deliver, "_load_audio", lambda key: b"MP3DATA")

        result = deliver.deliver_episode(episode_id)

        assert result["slack"]["ok"] is True
        assert fake.sent == [("Delivery test brief", b"MP3DATA", fake.sent[0][2])]
        assert fake.sent[0][2].endswith(".mp3")
        assert self.stored(episode_id)["slack"]["ok"] is True

    def test_a_brief_is_delivered_once_unless_forced(self, episode_id, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id="C1", smtp_host=None)
        fake = RecordingSlack()
        monkeypatch.setattr(deliver, "slack_client", lambda: fake)
        monkeypatch.setattr(deliver, "_load_audio", lambda key: None)

        deliver.deliver_episode(episode_id)
        deliver.deliver_episode(episode_id)
        assert len(fake.sent) == 1
        deliver.deliver_episode(episode_id, force=True)
        assert len(fake.sent) == 2

    def test_a_failed_channel_is_recorded_and_retried_next_time(self, episode_id, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id="C1", smtp_host=None)
        monkeypatch.setattr(deliver, "_load_audio", lambda key: None)

        monkeypatch.setattr(deliver, "slack_client", lambda: RecordingSlack(fail=True))
        first = deliver.deliver_episode(episode_id)
        assert first["slack"]["ok"] is False and "channel_not_found" in first["slack"]["error"]
        assert self.stored(episode_id)["slack"]["ok"] is False

        healthy = RecordingSlack()
        monkeypatch.setattr(deliver, "slack_client", lambda: healthy)
        deliver.deliver_episode(episode_id)
        assert len(healthy.sent) == 1, "a failed delivery must be retried"

    def test_one_channel_failing_does_not_stop_the_other(self, episode_id, monkeypatch):
        configure(
            monkeypatch,
            slack_bot_token="xoxb-1",
            slack_channel_id="C1",
            smtp_host="smtp.example.com",
            mail_from="a@b.c",
            email_to="d@e.f",
            smtp_password=None,
        )
        monkeypatch.setattr(deliver, "slack_client", lambda: RecordingSlack(fail=True))
        monkeypatch.setattr(deliver, "_load_audio", lambda key: None)
        sent = []
        monkeypatch.setattr(
            deliver.email_channel, "send", lambda message, **kw: sent.append(message)
        )

        result = deliver.deliver_episode(episode_id)
        assert result["slack"]["ok"] is False and result["email"]["ok"] is True
        assert len(sent) == 1 and sent[0]["To"] == "d@e.f"

    def test_missing_audio_still_delivers_the_digest(self, episode_id, monkeypatch):
        configure(monkeypatch, slack_bot_token="xoxb-1", slack_channel_id="C1", smtp_host=None)
        fake = RecordingSlack()
        monkeypatch.setattr(deliver, "slack_client", lambda: fake)
        monkeypatch.setattr(deliver, "_load_audio", lambda key: None)
        deliver.deliver_episode(episode_id)
        assert fake.sent[0][1] is None


def test_loading_audio_that_is_not_there_returns_nothing(monkeypatch):
    class Boom:
        def get_object(self, **_):
            raise OSError("no such key")

    monkeypatch.setattr("voicebrief.pipeline.episodes.s3_client", lambda: Boom())
    assert deliver._load_audio("episodes/missing.mp3") is None
    assert deliver._load_audio(None) is None


@pytest.mark.integration
class TestDailyRunWiring:
    """run_daily delivers a good brief, announces a failed one, and survives delivery."""

    @pytest.fixture
    def email(self):
        address = f"wiring-{uuid.uuid4().hex[:8]}@example.com"
        yield address
        with Session(get_engine()) as s:
            s.query(User).filter(User.email == address).delete()
            s.commit()

    def run(self, email, monkeypatch, status, *, delivery_raises=False):
        from voicebrief.pipeline import daily

        calls = {"deliver": [], "notify": []}
        episode_id = uuid.uuid4()

        def fake_generate(session, request, **kwargs):
            result = SimpleNamespace(
                episode_id=episode_id,
                status=status,
                title="T",
                duration_seconds=60.0,
                cost_inr=1.0,
                attribution_rate=0.9,
                errors=["the episode could not be produced"] if status == "failed" else [],
            )
            return daily.BriefOutcome(result, 3, 1, 36.0)

        def fake_deliver(eid):
            if delivery_raises:
                raise RuntimeError("slack is down")
            calls["deliver"].append(eid)
            return {"slack": {"ok": True}}

        monkeypatch.setattr(daily, "generate_brief", fake_generate)
        monkeypatch.setattr(daily, "deliver_episode", fake_deliver)
        monkeypatch.setattr(daily, "notify_failure", lambda m: calls["notify"].append(m) or {})
        from voicebrief.pipeline.enrich import EnrichResult

        monkeypatch.setattr(
            "voicebrief.pipeline.enrich.enrich", lambda *a, **k: EnrichResult(10, 8, 8, 1, 1, 5, 1)
        )
        # The conftest-level isolation patches these for other tests; this one needs them real.
        monkeypatch.setattr(daily, "start_run", lambda *a, **k: None)
        monkeypatch.setattr(daily, "finish_run", lambda *a, **k: None)
        monkeypatch.setattr(daily, "reap_interrupted", lambda: (0, 0))
        monkeypatch.setattr(daily, "record_interaction", lambda *a, **k: None)
        return daily.run_daily(daily.BriefRequest(email=email), skip_ingest=True), calls, episode_id

    def test_a_good_brief_is_delivered(self, email, monkeypatch):
        result, calls, episode_id = self.run(email, monkeypatch, "ready")
        assert calls["deliver"] == [episode_id] and calls["notify"] == []
        assert result.status == "ready"

    def test_a_degraded_brief_is_still_delivered_with_its_warning(self, email, monkeypatch):
        _, calls, episode_id = self.run(email, monkeypatch, "degraded")
        assert calls["deliver"] == [episode_id]

    def test_a_failed_episode_sends_the_failure_notice_instead(self, email, monkeypatch):
        result, calls, _ = self.run(email, monkeypatch, "failed")
        assert calls["deliver"] == [] and "could not be produced" in calls["notify"][0]
        assert result.exit_code == 1
