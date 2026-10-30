"""Deliver a finished brief to wherever it is configured to go.

Three rules, all learned the hard way by other morning-brief systems:

* **Delivery never breaks the brief.** Every channel is wrapped; a Slack outage or a bad
  SMTP password costs a message, not the episode, and the failure is recorded.
* **A brief is delivered once.** Results are stored on the episode, and a second call
  (a re-run, a retry after a crash) skips channels that already succeeded.
* **Silence is never ambiguous.** When a run fails, a short message says so, so a missing
  brief is never confused with a slow one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db import session_scope
from voicebrief.db.models import Episode
from voicebrief.delivery import mail as email_channel
from voicebrief.delivery.digest import build_digest
from voicebrief.delivery.slack import SlackClient
from voicebrief.logging import get_logger

log = get_logger(__name__)


def slack_client() -> SlackClient | None:
    settings = get_settings()
    if settings.slack_bot_token and settings.slack_channel_id:
        return SlackClient(settings.slack_bot_token.get_secret_value(), settings.slack_channel_id)
    return None


def email_configured() -> bool:
    settings = get_settings()
    return bool(settings.smtp_host and settings.mail_from and settings.email_to)


def configured_channels() -> list[str]:
    return [
        name
        for name, ok in (("slack", slack_client() is not None), ("email", email_configured()))
        if ok
    ]


def _load_audio(key: str | None) -> bytes | None:
    if not key:
        return None
    from voicebrief.pipeline.episodes import s3_client

    try:
        return s3_client().get_object(Bucket=get_settings().s3_bucket, Key=key)["Body"].read()
    except Exception as exc:
        log.warning(
            "delivery.audio_unavailable", key=key, error=f"{type(exc).__name__}: {exc}"[:200]
        )
        return None


def _record(session: Session, episode: Episode, channel: str, outcome: dict) -> None:
    grounding = dict(episode.grounding or {})
    delivery = dict(grounding.get("delivery", {}))
    delivery[channel] = {**outcome, "at": datetime.now(UTC).isoformat()}
    grounding["delivery"] = delivery
    episode.grounding = grounding  # reassigned: JSONB changes are not tracked in place
    session.flush()


def deliver_episode(episode_id: uuid.UUID, *, force: bool = False) -> dict[str, dict]:
    """Send the brief to every configured channel. Never raises.

    Returns the outcome per channel (also stored on the episode).
    """
    results: dict[str, dict] = {}
    try:
        channels = configured_channels()
        if not channels:
            return results

        settings = get_settings()
        with session_scope() as session:
            episode = session.get(Episode, episode_id)
            if episode is None:
                return results
            done = (episode.grounding or {}).get("delivery", {})
            digest = build_digest(session, episode_id, base_url=settings.public_url)
            key = episode.audio_key
            stem = f"voicebrief-{episode.created_at:%Y-%m-%d}"
            filename = f"{stem}.mp3" if (key or "").endswith(".mp3") else f"{stem}.wav"

            pending = [c for c in channels if force or not done.get(c, {}).get("ok")]
            if not pending:
                return {c: done[c] for c in channels if c in done}
            audio = _load_audio(key)

            for channel in pending:
                try:
                    if channel == "slack":
                        client = slack_client()
                        assert client is not None
                        detail = client.send_digest(digest, audio, filename)
                    else:
                        message = email_channel.build_message(
                            digest,
                            sender=settings.mail_from or "",
                            recipient=settings.email_to or "",
                            audio=audio,
                            filename=filename,
                        )
                        email_channel.send(
                            message,
                            host=settings.smtp_host or "",
                            port=settings.smtp_port,
                            user=settings.smtp_user,
                            password=(
                                settings.smtp_password.get_secret_value()
                                if settings.smtp_password
                                else None
                            ),
                        )
                        detail = {"audio": audio is not None}
                    outcome = {"ok": True, **detail}
                except Exception as exc:
                    outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
                    log.warning("delivery.failed", channel=channel, error=outcome["error"])
                results[channel] = outcome
                _record(session, episode, channel, outcome)
        return results
    except Exception as exc:  # delivery must never take the brief down with it
        log.error("delivery.crashed", error=f"{type(exc).__name__}: {exc}"[:300])
        return results


def notify_failure(message: str) -> dict[str, dict]:
    """Tell the configured channels that today's brief did not happen. Never raises."""
    results: dict[str, dict] = {}
    try:
        settings = get_settings()
        if client := slack_client():
            try:
                client.send_failure(message)
                results["slack"] = {"ok": True}
            except Exception as exc:
                results["slack"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        if email_configured():
            try:
                from email.message import EmailMessage

                note = EmailMessage()
                note["From"] = settings.mail_from or ""
                note["To"] = settings.email_to or ""
                note["Subject"] = "VoiceBrief could not make today's brief"
                note.set_content(message[:2000])
                email_channel.send(
                    note,
                    host=settings.smtp_host or "",
                    port=settings.smtp_port,
                    user=settings.smtp_user,
                    password=(
                        settings.smtp_password.get_secret_value()
                        if settings.smtp_password
                        else None
                    ),
                )
                results["email"] = {"ok": True}
            except Exception as exc:
                results["email"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    except Exception as exc:
        log.error("delivery.failure_notice_crashed", error=str(exc)[:300])
    return results
