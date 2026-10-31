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

import os
import struct
import wave
from pathlib import Path
from typing import ClassVar

from voicebrief.config import get_settings
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
    VOICES: ClassVar[dict[str, str]] = {
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
        except Exception as exc:
            raise TTSError(f"Kokoro synthesis failed: {exc}") from exc

        return SpeechClip(
            path=out_path,
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            voice=voice,
            engine=self.name,
        )


ONNX_MODEL_FILE = "kokoro-v1.0.onnx"
ONNX_VOICES_FILE = "voices-v1.0.bin"
ONNX_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


def model_dir() -> Path:
    """Per-user model folder: models are hundreds of MB and do not belong in the repo."""
    configured = get_settings().tts_model_dir
    if configured:
        return Path(configured)
    base = (
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("XDG_DATA_HOME")
        or str(Path.home() / ".local" / "share")
    )
    return Path(base) / "VoiceBrief" / "models"


class KokoroOnnxEngine(TTSEngine):
    """The Kokoro voice model run through onnxruntime.

    The reference `kokoro` package pins Python below 3.13, which ruled it out on current
    interpreters. This is the same 82M-parameter model and the same voices, loaded from
    two files (see `voicebrief tts setup`), with no torch or spaCy dependency.
    """

    name = "kokoro"
    sample_rate = 24_000
    VOICES: ClassVar[dict[str, str]] = KokoroEngine.VOICES

    def __init__(self, lang: str = "en-us") -> None:
        self.lang = lang
        self._kokoro = None

    def paths(self) -> tuple[Path, Path]:
        folder = model_dir()
        return folder / ONNX_MODEL_FILE, folder / ONNX_VOICES_FILE

    def available(self) -> bool:
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            return False
        return all(path.is_file() for path in self.paths())

    def voices(self) -> dict[str, str]:
        return dict(self.VOICES)

    def _load(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            model, voices = self.paths()
            log.info("tts.loading", engine="kokoro-onnx", lang=self.lang)
            self._kokoro = Kokoro(str(model), str(voices))
        return self._kokoro

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        if not text.strip():
            raise TTSError("refusing to synthesize empty text")
        try:
            import soundfile as sf

            samples, rate = self._load().create(text, voice=voice, speed=speed, lang=self.lang)
            if samples is None or len(samples) == 0:
                raise TTSError("Kokoro (onnx) produced no audio")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, samples, rate)
            duration = len(samples) / float(rate)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Kokoro (onnx) synthesis failed: {exc}") from exc

        return SpeechClip(
            path=out_path,
            duration_seconds=duration,
            sample_rate=int(rate),
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
        onnx = KokoroOnnxEngine(lang="hi" if language == "hi" else "en-us")
        if onnx.available():
            return onnx
        log.warning(
            "tts.unavailable",
            engine="kokoro",
            fallback="null",
            hint="run `voicebrief tts setup` to download the voice model",
        )
    elif name == "piper":
        piper = PiperEngine()
        if piper.available():
            return piper
        log.warning("tts.unavailable", engine="piper", fallback="null")

    return SilentEngine()
