"""Answer questions about the brief in Slack.

Socket Mode: the bot opens an outbound WebSocket to Slack (`apps.connections.open` with
the app-level token), so a laptop behind a home router receives messages without a public
URL. Every envelope is acknowledged at once, as Slack requires within three seconds, and
the answer is worked out afterwards.

- A reply in a brief's thread is a question about that brief.
- A new message in the channel is a question about the latest brief; the answer goes in
  a thread under it.

Answers come from `voicebrief.qa`, the same rule as the web player: only the episode and
its sources, and "the sources don't say" when they don't.
"""

from __future__ import annotations

import contextlib
import json
import threading
import uuid
from typing import Any

import httpx
from sqlalchemy import select

from voicebrief.config import get_settings
from voicebrief.db import session_scope
from voicebrief.db.models import Episode, EpisodeMode, EpisodeStatus
from voicebrief.delivery.slack import API, SlackClient
from voicebrief.logging import get_logger

log = get_logger(__name__)

RECONNECT_SECONDS = (2, 5, 15, 30, 60)
MAX_QUESTION_CHARS = 1500


def episode_for_thread(thread_ts: str | None) -> uuid.UUID | None:
    """The episode whose Slack digest started this thread, else the latest brief."""
    with session_scope() as session:
        usable = Episode.status.in_([EpisodeStatus.ready, EpisodeStatus.degraded])
        if thread_ts:
            match = session.execute(
                select(Episode.id).where(
                    Episode.grounding["delivery"]["slack"]["ts"].astext == thread_ts
                )
            ).scalar_one_or_none()
            if match is not None:
                return match
        return session.execute(
            select(Episode.id)
            .where(usable, Episode.mode == EpisodeMode.brief)
            .order_by(Episode.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()


def is_question(event: dict, *, channel: str, bot_user: str | None) -> bool:
    """A person's new message in our channel; not the bot, not an edit or a join."""
    return (
        event.get("type") == "message"
        and event.get("channel") == channel
        and not event.get("subtype")
        and not event.get("bot_id")
        and event.get("user") not in (None, bot_user)
        and bool((event.get("text") or "").strip())
    )


def handle_message(event: dict, client: SlackClient) -> None:
    """Answer one message. Never raises: a bad question must not stop the listener."""
    # A top-level message gets its answer in a thread under itself.
    thread = event.get("thread_ts") or event["ts"]
    question = (event.get("text") or "").strip()[:MAX_QUESTION_CHARS]
    try:
        from voicebrief.qa import answer_about_episode

        episode_id = episode_for_thread(event.get("thread_ts"))
        if episode_id is None:
            client.post_message("There is no brief yet to answer from.", thread_ts=thread)
            return
        with session_scope() as session:
            answer = answer_about_episode(session, episode_id, question, channel="slack")
        links = list(dict.fromkeys(c["url"] for c in answer.citations if c.get("url")))[:3]
        text = answer.text
        if links:
            text += "\n\n" + "  ".join(f"<{url}|source {i}>" for i, url in enumerate(links, 1))
        client.post_message(text, thread_ts=thread)
    except Exception as exc:
        log.error("slack.answer_failed", error=f"{type(exc).__name__}: {exc}"[:300])
        with contextlib.suppress(Exception):
            client.post_message(
                "Sorry, I could not answer that just now. The web app's Ask button may work.",
                thread_ts=thread,
            )


class Listener:
    def __init__(self, app_token: str, bot_token: str, channel: str) -> None:
        self.app_token = app_token
        self.client = SlackClient(bot_token, channel)
        self.channel = channel
        self.bot_user: str | None = None
        self._stop = threading.Event()

    def _bot_user(self) -> str | None:
        try:
            return self.client._call("auth.test", data={}).get("user_id")
        except Exception:
            return None

    def _socket_url(self) -> str:
        response = httpx.post(
            f"{API}/apps.connections.open",
            headers={"Authorization": f"Bearer {self.app_token}"},
            timeout=30,
        )
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"apps.connections.open: {payload.get('error')}")
        return payload["url"]

    def dispatch(self, envelope: dict[str, Any]) -> None:
        if envelope.get("type") != "events_api":
            return
        event = (envelope.get("payload") or {}).get("event") or {}
        if is_question(event, channel=self.channel, bot_user=self.bot_user):
            threading.Thread(target=handle_message, args=(event, self.client), daemon=True).start()

    def run_once(self) -> None:
        """One connection, until Slack asks us to reconnect or it drops."""
        from websockets.sync.client import connect

        with connect(self._socket_url(), open_timeout=30) as socket:
            log.info("slack.listening", channel=self.channel)
            while not self._stop.is_set():
                try:
                    raw = socket.recv(timeout=30)
                except TimeoutError:
                    continue
                envelope = json.loads(raw)
                if envelope_id := envelope.get("envelope_id"):
                    socket.send(json.dumps({"envelope_id": envelope_id}))
                if envelope.get("type") == "disconnect":
                    return  # Slack rotates connections; open a fresh one
                self.dispatch(envelope)

    def run_forever(self) -> None:
        self.bot_user = self._bot_user()
        failures = 0
        while not self._stop.is_set():
            try:
                self.run_once()
                failures = 0
            except Exception as exc:
                wait = RECONNECT_SECONDS[min(failures, len(RECONNECT_SECONDS) - 1)]
                failures += 1
                log.warning("slack.listener_dropped", error=str(exc)[:200], retry_in=wait)
                self._stop.wait(wait)

    def stop(self) -> None:
        self._stop.set()


def listener_from_settings() -> Listener | None:
    settings = get_settings()
    if not (settings.slack_app_token and settings.slack_bot_token and settings.slack_channel_id):
        return None
    return Listener(
        settings.slack_app_token.get_secret_value(),
        settings.slack_bot_token.get_secret_value(),
        settings.slack_channel_id,
    )


def start_in_background() -> Listener | None:
    """Start answering in Slack on a daemon thread, if it is configured."""
    listener = listener_from_settings()
    if listener is not None:
        threading.Thread(target=listener.run_forever, name="slack-listener", daemon=True).start()
    return listener
