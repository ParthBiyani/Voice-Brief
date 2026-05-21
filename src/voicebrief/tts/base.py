"""Text-to-speech contract.

Per-segment synthesis with a content-addressed cache, not one call for the whole
episode. Three reasons, all of which came out of how the rest of the system works:

* **Caching.** A regenerated episode usually reuses the cold open and several
  segments verbatim. Hashing the text means unchanged segments are never re-synthesised.
* **Timestamps.** The transcript needs clickable timestamps per segment, and the only
  way to know where a segment starts is to know how long the previous ones ran.
* **Failure isolation.** One segment that trips the engine costs that segment, not
  the episode.

Engines are swappable because the PRD reserves the right to pick on a blind listen
test in week 4 rather than committing up front.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SpeechClip:
    """One synthesised segment."""

    path: Path
    duration_seconds: float
    sample_rate: int
    voice: str
    engine: str
    cached: bool = False
    meta: dict = field(default_factory=dict)


class TTSError(RuntimeError):
    """Synthesis failed for a segment."""


def voice_key(text: str, voice: str, engine: str, speed: float = 1.0) -> str:
    """Content address for a clip.

    Includes the engine and voice, not just the text: the same words spoken by a
    different voice are a different artefact, and a cache keyed on text alone would
    serve the wrong audio after a voice change.
    """
    basis = f"{engine}|{voice}|{speed:.2f}|{text.strip()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:40]


class TTSEngine(ABC):
    name: str
    sample_rate: int = 24_000

    @abstractmethod
    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        """Render `text` to a WAV file at `out_path`."""

    @abstractmethod
    def available(self) -> bool:
        """Whether this engine can actually run right now.

        Checked before use so a missing model degrades to a documented fallback
        instead of failing halfway through an episode.
        """

    def voices(self) -> dict[str, str]:  # pragma: no cover - metadata only
        return {}
