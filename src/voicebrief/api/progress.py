"""Generation progress, shared between the worker thread and the SSE endpoint.

The worker runs in a thread and the stream is served by the event loop, so the channel
has to be thread-safe: the previous version pushed straight into an `asyncio.Queue` from
the worker thread, which is not allowed and let events sit unseen until the next
keep-alive.

Events are kept (not consumed), so a page reload, a second tab or a phone opened mid-run
replays everything so far and then follows live. Finished channels linger for a while so
a late reload still sees the outcome.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime

KEEP_FINISHED_SECONDS = 30 * 60
FINAL_STAGES = ("ready", "failed")


class Channel:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.finished = False
        self.finished_at: float | None = None
        self._cond = threading.Condition()

    def emit(self, stage: str, /, **fields) -> None:
        event = {"stage": stage, "at": datetime.now(UTC).isoformat(), **fields}
        with self._cond:
            self.events.append(event)
            if stage in FINAL_STAGES:
                self.finished = True
                self.finished_at = time.monotonic()
            self._cond.notify_all()

    def wait(self, cursor: int, timeout: float) -> tuple[list[dict], bool]:
        """Block until there are events past `cursor`, the run ends, or the timeout."""
        with self._cond:
            if len(self.events) <= cursor and not self.finished:
                self._cond.wait(timeout)
            return self.events[cursor:], self.finished


_channels: dict[uuid.UUID, Channel] = {}
_lock = threading.Lock()


def open_channel(episode_id: uuid.UUID) -> Channel:
    with _lock:
        _purge()
        return _channels.setdefault(episode_id, Channel())


def get_channel(episode_id: uuid.UUID) -> Channel | None:
    with _lock:
        return _channels.get(episode_id)


def emit(channel_id: uuid.UUID, stage: str, /, **fields) -> None:
    """Positional-only, so a field may itself be called `episode_id`."""
    open_channel(channel_id).emit(stage, **fields)


def active_ids() -> list[uuid.UUID]:
    with _lock:
        return [i for i, c in _channels.items() if not c.finished]


def _purge() -> None:
    now = time.monotonic()
    stale = [
        i
        for i, c in _channels.items()
        if c.finished and c.finished_at is not None and now - c.finished_at > KEEP_FINISHED_SECONDS
    ]
    for episode_id in stale:
        del _channels[episode_id]
