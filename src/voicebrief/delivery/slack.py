"""Slack delivery: a digest message with the audio attached in its thread.

Plain HTTPS against the Web API, no SDK. Files use Slack's current external-upload flow
(`files.getUploadURLExternal`, then the bytes, then `files.completeUploadExternal`); the
older `files.upload` is retired. The audio lands in the digest's own thread, which Slack's
mobile app plays inline.
"""

from __future__ import annotations

import time

import httpx

from voicebrief.delivery.digest import Digest, to_slack_blocks
from voicebrief.logging import get_logger

log = get_logger(__name__)

API = "https://slack.com/api"
TIMEOUT = httpx.Timeout(30.0, read=120.0)


class SlackError(RuntimeError):
    """Slack answered, and the answer was no."""


class SlackClient:
    def __init__(self, token: str, channel: str, *, client: httpx.Client | None = None) -> None:
        self.channel = channel
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = client or httpx.Client(timeout=TIMEOUT)

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _call(self, method: str, *, json: dict | None = None, data: dict | None = None) -> dict:
        for attempt in (1, 2):
            if json is not None:
                response = self._client.post(f"{API}/{method}", headers=self._headers, json=json)
            else:
                response = self._client.post(
                    f"{API}/{method}", headers=self._headers, data=data or {}
                )
            if response.status_code == 429 and attempt == 1:
                wait = min(int(response.headers.get("Retry-After", "2")), 30)
                log.warning("slack.rate_limited", method=method, wait=wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise SlackError(f"{method}: {payload.get('error', 'unknown error')}")
            return payload
        raise SlackError(f"{method}: rate limited")  # pragma: no cover

    # ── messages ──────────────────────────────────────────────────────────────
    def post_message(self, text: str, blocks: list[dict] | None = None) -> str:
        """Post to the channel. Returns the message timestamp (the thread id)."""
        body: dict = {"channel": self.channel, "text": text, "unfurl_links": False}
        if blocks:
            body["blocks"] = blocks
        return self._call("chat.postMessage", json=body)["ts"]

    def upload_file(
        self,
        content: bytes,
        filename: str,
        *,
        thread_ts: str | None = None,
        title: str | None = None,
    ) -> str:
        """Attach a file to the channel (in a thread when `thread_ts` is given)."""
        started = self._call(
            "files.getUploadURLExternal", data={"filename": filename, "length": str(len(content))}
        )
        put = self._client.post(started["upload_url"], content=content)
        put.raise_for_status()

        body: dict = {
            "files": [{"id": started["file_id"], "title": title or filename}],
            "channel_id": self.channel,
        }
        if thread_ts:
            body["thread_ts"] = thread_ts
        self._call("files.completeUploadExternal", json=body)
        return started["file_id"]

    # ── the morning brief ─────────────────────────────────────────────────────
    def send_digest(self, digest: Digest, audio: bytes | None, filename: str) -> dict:
        ts = self.post_message(f"{digest.headline}: {digest.title}", to_slack_blocks(digest))
        result = {"ts": ts, "audio": False}
        if audio:
            try:
                self.upload_file(audio, filename, thread_ts=ts, title=f"{digest.headline} (audio)")
                result["audio"] = True
            except Exception as exc:
                # The digest is already delivered; say plainly that the audio is not.
                log.warning("slack.audio_failed", error=f"{type(exc).__name__}: {exc}"[:200])
                result["audio_error"] = f"{type(exc).__name__}: {exc}"[:200]
                self.post_message(
                    ":warning: The audio could not be attached; the transcript is in the app.",
                )
        return result

    def send_failure(self, message: str) -> str:
        return self.post_message(
            f":x: VoiceBrief could not make today's brief: {message[:600]}",
            [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":x: *VoiceBrief could not make today's brief.*\n{message[:600]}",
                    },
                }
            ],
        )
