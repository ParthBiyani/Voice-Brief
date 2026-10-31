"""Answering questions in Slack: which messages count, which brief they are about, and
that the listener survives a bad question."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from voicebrief.db.models import Episode, EpisodeMode, EpisodeStatus, Segment, User
from voicebrief.db.session import get_engine
from voicebrief.delivery import slack_listener
from voicebrief.delivery.slack_listener import Listener, handle_message, is_question
from voicebrief.qa import Answer

CHANNEL = "C1"
BOT = "UBOT"


def message(**over) -> dict:
    return {
        "type": "message",
        "channel": CHANNEL,
        "user": "UME",
        "text": "why?",
        "ts": "1.0",
    } | over


class TestWhatCountsAsAQuestion:
    def test_a_person_writing_in_the_channel(self):
        assert is_question(message(), channel=CHANNEL, bot_user=BOT)

    @pytest.mark.parametrize(
        "over",
        [
            {"bot_id": "B1"},  # the bot's own digest or answer
            {"user": BOT},
            {"subtype": "channel_join"},
            {"subtype": "message_changed"},
            {"channel": "C-other"},
            {"text": "   "},
        ],
    )
    def test_not_bots_edits_joins_or_other_channels(self, over):
        assert not is_question(message(**over), channel=CHANNEL, bot_user=BOT)


class RecordingClient:
    def __init__(self):
        self.posts = []

    def post_message(self, text, blocks=None, *, thread_ts=None):
        self.posts.append((text, thread_ts))
        return "9.9"


class TestHandling:
    def test_a_thread_reply_is_answered_in_that_thread_with_sources(self, monkeypatch):
        episode = uuid.uuid4()
        seen = {}
        monkeypatch.setattr(slack_listener, "episode_for_thread", lambda ts: episode)

        def answer(session, episode_id, question, *, channel):
            seen.update(episode=episode_id, question=question, channel=channel)
            return Answer("Windows and Linux.", [{"url": "https://x.dev/a"}], 0.1)

        monkeypatch.setattr("voicebrief.qa.answer_about_episode", answer)
        monkeypatch.setattr(slack_listener, "session_scope", _null_scope)
        client = RecordingClient()

        handle_message(message(thread_ts="5.5", ts="6.6", text="Which OS?"), client)

        assert seen == {"episode": episode, "question": "Which OS?", "channel": "slack"}
        ((text, thread),) = client.posts
        assert thread == "5.5" and text.startswith("Windows and Linux.")
        assert "<https://x.dev/a|source 1>" in text

    def test_a_new_message_is_answered_under_itself(self, monkeypatch):
        monkeypatch.setattr(slack_listener, "episode_for_thread", lambda ts: uuid.uuid4())
        monkeypatch.setattr(
            "voicebrief.qa.answer_about_episode", lambda *a, **k: Answer("ok", [], 0.0)
        )
        monkeypatch.setattr(slack_listener, "session_scope", _null_scope)
        client = RecordingClient()
        handle_message(message(ts="7.7"), client)
        assert client.posts == [("ok", "7.7")]

    def test_a_failure_becomes_an_apology_not_a_crash(self, monkeypatch):
        def broken(ts):
            raise RuntimeError("db down")

        monkeypatch.setattr(slack_listener, "episode_for_thread", broken)
        client = RecordingClient()
        handle_message(message(), client)
        assert "could not answer" in client.posts[0][0]

    def test_no_brief_yet(self, monkeypatch):
        monkeypatch.setattr(slack_listener, "episode_for_thread", lambda ts: None)
        client = RecordingClient()
        handle_message(message(), client)
        assert "no brief yet" in client.posts[0][0]


class TestEnvelopes:
    def test_only_questions_in_event_envelopes_are_dispatched(self, monkeypatch):
        handled = []
        monkeypatch.setattr(slack_listener, "handle_message", lambda e, c: handled.append(e))
        listener = Listener("xapp-1", "xoxb-1", CHANNEL)
        listener.bot_user = BOT
        listener.dispatch({"type": "hello"})
        listener.dispatch({"type": "events_api", "payload": {"event": message(bot_id="B")}})
        listener.dispatch({"type": "events_api", "payload": {"event": message(text="real")}})
        import time

        for _ in range(50):
            if handled:
                break
            time.sleep(0.01)
        assert [e["text"] for e in handled] == ["real"]

    def test_off_without_all_three_settings(self, monkeypatch):
        from voicebrief.config import get_settings

        settings = get_settings().model_copy(update={"slack_app_token": None})
        monkeypatch.setattr(slack_listener, "get_settings", lambda: settings)
        assert slack_listener.listener_from_settings() is None


class _null_scope:  # noqa: N801  (stands in for the session_scope function)
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.mark.integration
class TestWhichBrief:
    @pytest.fixture
    def episodes(self):
        engine = get_engine()
        with Session(engine) as s:
            user = User(email=f"slackqa-{uuid.uuid4().hex[:8]}@example.com")
            s.add(user)
            s.flush()
            thread = f"{uuid.uuid4().int % 10**10}.000100"
            ep = Episode(
                user_id=user.id,
                mode=EpisodeMode.brief,
                status=EpisodeStatus.ready,
                title="Threaded brief",
                grounding={"delivery": {"slack": {"ok": True, "ts": thread}}},
            )
            s.add(ep)
            s.flush()
            s.add(Segment(episode_id=ep.id, position=1, kind="story", heading="H", script="S."))
            s.commit()
            ids = (user.id, ep.id, thread)
        yield ids
        with Session(engine) as s:
            s.query(Episode).filter(Episode.id == ids[1]).delete(synchronize_session=False)
            s.query(User).filter(User.id == ids[0]).delete()
            s.commit()

    def test_a_thread_maps_to_the_brief_that_started_it(self, episodes):
        _, episode_id, thread = episodes
        assert slack_listener.episode_for_thread(thread) == episode_id

    def test_an_unknown_thread_falls_back_to_a_brief(self, episodes):
        assert slack_listener.episode_for_thread("0.0") is not None

    def test_an_episode_question_is_answered_and_recorded(self, episodes, monkeypatch):
        from voicebrief import qa

        recorded = []
        monkeypatch.setattr(qa, "record_interaction", lambda *a, **k: recorded.append((a, k)))
        with Session(get_engine()) as s:
            answer = qa.answer_about_episode(s, episodes[1], "What happened?", channel="slack")
        assert answer.text  # the echo provider answers deterministically
        ((args, kwargs),) = recorded
        assert args[:2] == ("qa", "slack") and kwargs["request"]["scope"] == "episode"
