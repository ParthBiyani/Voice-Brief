"""TTS engines.

Three, in descending order of quality and ascending order of availability:

* **Kokoro** — 82M parameters, Apache-licensed, runs on CPU. The PRD's default.
* **Piper** — the fallback when Kokoro's weights are not present. Widely packaged.
* **Silent** — generates correctly-timed silence. Not a joke: it makes the whole
  pipeline, including episode assembly and transcript timestamps, runnable and
  testable on a machine with no audio stack at all.

The silent engine is what keeps CI honest. Every timing calculation downstream is
exercised by it, so a bug in assembly surfaces in tests rather than on the first day
someone plays a real episode.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from voicebrief.logging import get_logger
from voicebrief.tts.base import SpeechClip, TTSEngine, TTSError

log = get_logger(__name__)

# Speaking rate used to estimate duration when no audio is rendered. Matches the
# 155 wpm the script generator targets.
WORDS_PER_SECOND = 155 / 60


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24_000

    # Kokoro voice ids. The Hindi voices are what make the PRD's multilingual claim
    # more than a translation layer.
    VOICES = {
        "af_heart": "en-US female, warm",
        "af_bella": "en-US female, brisk",
        "am_adam": "en-US male, news register",
        "bf_emma": "en-GB female",
        "hf_alpha": "hi female",
        "hf_beta": "hi female, alternate",
        "hm_omega": "hi male",
    }

    def __init__(self, lang_code: str = "a") -> None:
        self.lang_code = lang_code
        self._pipeline = None

    def available(self) -> bool:
        try:
            import kokoro  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            log.info("tts.loading", engine=self.name, lang=self.lang_code)
            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def voices(self) -> dict[str, str]:
        return dict(self.VOICES)

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        if not text.strip():
            raise TTSError("refusing to synthesize empty text")

        try:
            import numpy as np
            import soundfile as sf

            pipeline = self._load()
            chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
            if not chunks:
                raise TTSError("Kokoro produced no audio")

            samples = np.concatenate(chunks)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, samples, self.sample_rate)
            duration = len(samples) / self.sample_rate
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001 — engine internals vary by version
            raise TTSError(f"Kokoro synthesis failed: {exc}") from exc

        return SpeechClip(
            path=out_path,
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            voice=voice,
            engine=self.name,
        )


class PiperEngine(TTSEngine):
    """Fallback engine. Invoked as a subprocess because Piper ships as a binary."""

    name = "piper"
    sample_rate = 22_050

    def __init__(self, model_path: str | None = None, binary: str = "piper") -> None:
        self.model_path = model_path
        self.binary = binary

    def available(self) -> bool:
        import shutil

        return shutil.which(self.binary) is not None and bool(self.model_path)

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        import subprocess

        if not text.strip():
            raise TTSError("refusing to synthesize empty text")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        command = [self.binary, "--model", str(self.model_path), "--output_file", str(out_path)]
        if speed != 1.0:
            command += ["--length_scale", f"{1 / speed:.3f}"]

        try:
            subprocess.run(
                command, input=text.encode("utf-8"), check=True, capture_output=True, timeout=180
            )
        except subprocess.CalledProcessError as exc:
            raise TTSError(f"piper failed: {exc.stderr.decode()[:200]}") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TTSError(f"piper unavailable: {exc}") from exc

        return SpeechClip(
            path=out_path,
            duration_seconds=_wav_duration(out_path),
            sample_rate=self.sample_rate,
            voice=voice,
            engine=self.name,
        )


class SilentEngine(TTSEngine):
    """Correctly-timed silence.

    Exists so that episode assembly, transcript timestamping and the API can all be
    exercised without an audio stack. Duration is estimated from word count at the
    same rate the script generator targets, so downstream timing arithmetic sees
    realistic values rather than zeros.
    """

    name = "null"
    sample_rate = 24_000

    def available(self) -> bool:
        return True

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        words = max(len(text.split()), 1)
        duration = max(words / (WORDS_PER_SECOND * speed), 0.5)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames = int(duration * self.sample_rate)
        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))

        return SpeechClip(
            path=out_path,
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            voice=voice,
            engine=self.name,
            meta={"synthetic": True},
        )


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def build_engine(name: str, *, language: str = "en") -> TTSEngine:
    """Construct the requested engine, degrading rather than raising.

    A missing model must not stop an episode from being produced — a silent episode
    with a correct transcript is still useful, and the log says exactly why.
    """
    if name == "kokoro":
        engine = KokoroEngine(lang_code="h" if language == "hi" else "a")
        if engine.available():
            return engine
        log.warning("tts.unavailable", engine="kokoro", fallback="null",
                    hint="pip install 'voicebrief[tts]' to enable")
    elif name == "piper":
        engine = PiperEngine()
        if engine.available():
            return engine
        log.warning("tts.unavailable", engine="piper", fallback="null")

    return SilentEngine()
