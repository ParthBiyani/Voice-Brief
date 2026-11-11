"""TTS engines.

In descending order of how human they sound, and ascending order of how local they are:

* **ElevenLabs**, **OpenAI** and **Edge** voices: cloud speech. Edge is free and needs no
  key; the other two are paid and keyed. Text leaves the machine, so a listener who wants
  none of that picks Kokoro.
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
import re
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


def _save_wav(samples, rate: int, out_path: Path, target_rate: int) -> float:
    """Write mono float samples as 16-bit WAV at the episode's sample rate.

    Every clip in an episode must share one rate or timestamps drift (see assemble), so
    audio that arrives at another rate is resampled here, once, at the edge.
    """
    import numpy as np
    import soundfile as sf

    samples = np.asarray(samples, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != target_rate and len(samples):
        count = round(len(samples) * target_rate / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples
        ).astype("float32")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), samples, target_rate, subtype="PCM_16")
    return len(samples) / float(target_rate)


def _with_retries(action, *, attempts: int = 3, what: str):
    """Run a network call, retrying transient failures with a short growing pause."""
    import time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
    raise TTSError(f"{what} failed after {attempts} attempts: {last}") from last


class EdgeEngine(TTSEngine):
    """Microsoft's neural voices, through the free endpoint the Edge browser reads aloud with.

    No key and no cost, and far more human than a small local model: these are the same
    voices most AI-made short videos use. The catch is that the text of each sentence goes
    to Microsoft, so a listener who wants everything on their own machine sets
    VB_TTS_ENGINE=kokoro. Needs a network; `build_engine` falls back to Kokoro without one.
    """

    name = "edge"
    sample_rate = 24_000
    concurrency = 4

    VOICES: ClassVar[dict[str, str]] = {
        "en-US-AvaMultilingualNeural": "en-US female, warm and conversational",
        "en-US-EmmaMultilingualNeural": "en-US female, clear",
        "en-US-AndrewMultilingualNeural": "en-US male, relaxed",
        "en-US-BrianMultilingualNeural": "en-US male, steady",
        "en-IN-NeerjaExpressiveNeural": "en-IN female, expressive",
        "en-IN-PrabhatNeural": "en-IN male",
        "hi-IN-SwaraNeural": "hi female",
        "hi-IN-MadhurNeural": "hi male",
    }

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False
        return True

    def voices(self) -> dict[str, str]:
        return dict(self.VOICES)

    @staticmethod
    def reachable() -> bool:
        import socket

        try:
            with socket.create_connection(("speech.platform.bing.com", 443), timeout=4):
                return True
        except OSError:
            return False

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        if not text.strip():
            raise TTSError("refusing to synthesize empty text")
        import asyncio
        import io

        import edge_tts
        import soundfile as sf

        rate = f"{round((speed - 1.0) * 100):+d}%"

        async def fetch() -> bytes:
            chunks = bytearray()
            async for part in edge_tts.Communicate(text, voice, rate=rate).stream():
                if part["type"] == "audio":
                    chunks.extend(part["data"])
            return bytes(chunks)

        def attempt() -> bytes:
            data = asyncio.run(fetch())
            if not data:
                raise TTSError("Edge returned no audio")
            return data

        data = _with_retries(attempt, what="Edge speech")
        try:
            samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
        except Exception as exc:
            raise TTSError(f"Edge audio could not be decoded: {exc}") from exc
        duration = _save_wav(samples, sample_rate, out_path, self.sample_rate)
        return SpeechClip(out_path, duration, self.sample_rate, voice, self.name)


class OpenAIEngine(TTSEngine):
    """OpenAI's speech model, which takes spoken-style instructions with the text.

    Paid (about a cent and a half a minute) and needs OPENAI_API_KEY. Its edge over the
    others is that tone is directed in words: the same sentence can be told warmly,
    quickly, like a colleague, which is what keeps a long brief from sounding flat.
    """

    name = "openai"
    sample_rate = 24_000
    concurrency = 4
    MODEL = "gpt-4o-mini-tts"
    VOICES: ClassVar[dict[str, str]] = {
        "coral": "female, warm",
        "nova": "female, bright",
        "sage": "female, calm",
        "ash": "male, relaxed",
        "onyx": "male, deep",
        "echo": "male, clear",
    }
    STYLE = (
        "Speak like a friendly, well-informed colleague telling a friend the news over coffee. "
        "Natural, unhurried and conversational, with real variation in pitch and pace. Never "
        "announcer-like and never flat. Let questions rise and short replies come in quickly."
    )

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._key = api_key or (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )

    def available(self) -> bool:
        return bool(self._key)

    def voices(self) -> dict[str, str]:
        return dict(self.VOICES)

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        if not text.strip():
            raise TTSError("refusing to synthesize empty text")
        import io

        import httpx
        import soundfile as sf

        def attempt() -> bytes:
            response = httpx.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self.MODEL,
                    "voice": voice,
                    "input": text,
                    "instructions": self.STYLE,
                    "response_format": "wav",
                    "speed": max(0.25, min(4.0, speed)),
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content

        data = _with_retries(attempt, what="OpenAI speech")
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
        duration = _save_wav(samples, sample_rate, out_path, self.sample_rate)
        return SpeechClip(out_path, duration, self.sample_rate, voice, self.name)


class ElevenLabsEngine(TTSEngine):
    """ElevenLabs: the most human voices commercially available, and what most polished
    reels use. Paid per character, needs ELEVENLABS_API_KEY. Voices are named by id."""

    name = "elevenlabs"
    sample_rate = 24_000
    concurrency = 4
    MODEL = "eleven_multilingual_v2"
    VOICES: ClassVar[dict[str, str]] = {
        "21m00Tcm4TlvDq8ikWAM": "Rachel, female, calm",
        "EXAVITQu4vr4xnSDxMaL": "Sarah, female, soft",
        "pNInz6obpgDQGcFmaJgB": "Adam, male, deep",
        "TX3LPaxmHKxFdv7VOQHJ": "Liam, male, young",
    }

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._key = api_key or (
            settings.elevenlabs_api_key.get_secret_value() if settings.elevenlabs_api_key else None
        )

    def available(self) -> bool:
        return bool(self._key)

    def voices(self) -> dict[str, str]:
        return dict(self.VOICES)

    def synthesize(
        self, text: str, *, voice: str, out_path: Path, speed: float = 1.0
    ) -> SpeechClip:
        if not text.strip():
            raise TTSError("refusing to synthesize empty text")
        import httpx
        import numpy as np

        def attempt() -> bytes:
            response = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                params={"output_format": "pcm_24000"},
                headers={"xi-api-key": str(self._key)},
                json={
                    "text": text,
                    "model_id": self.MODEL,
                    "voice_settings": {
                        # Lower stability lets delivery vary the way a person's does.
                        "stability": 0.4,
                        "similarity_boost": 0.8,
                        "style": 0.3,
                        "speed": max(0.7, min(1.2, speed)),
                    },
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content

        data = _with_retries(attempt, what="ElevenLabs speech")
        samples = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
        duration = _save_wav(samples, 24_000, out_path, self.sample_rate)
        return SpeechClip(out_path, duration, self.sample_rate, voice, self.name)


# Who says what, by engine and language: the narrator, host A (asks) and host B (answers).
# Hinglish is read by the Hindi voices, which pronounce English words the Indian way, as
# the listener's own colleagues do.
_EDGE_EN = (
    "en-US-AvaMultilingualNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
)
_EDGE_HI = ("hi-IN-SwaraNeural", "hi-IN-SwaraNeural", "hi-IN-MadhurNeural")
_OPENAI = ("coral", "coral", "ash")
_ELEVEN = ("21m00Tcm4TlvDq8ikWAM", "21m00Tcm4TlvDq8ikWAM", "pNInz6obpgDQGcFmaJgB")
DEFAULT_VOICES: dict[str, dict[str, tuple[str, str, str]]] = {
    "edge": {"en": _EDGE_EN, "hi": _EDGE_HI, "hinglish": _EDGE_HI},
    "openai": {"en": _OPENAI, "hi": _OPENAI, "hinglish": _OPENAI},
    "elevenlabs": {"en": _ELEVEN, "hi": _ELEVEN, "hinglish": _ELEVEN},
    "kokoro": {
        "en": ("af_heart", "af_heart", "am_michael"),
        "hi": ("hf_alpha", "hf_alpha", "hm_omega"),
        "hinglish": ("hf_alpha", "hf_alpha", "hm_omega"),
    },
}

# A voice named for one engine means nothing to another, and an old .env may still name a
# Kokoro voice. An override is used only when it looks like it belongs to the engine.
_VOICE_SHAPE = {
    "edge": re.compile(r"^[a-z]{2}-[A-Z]{2}-\w+Neural$"),
    "kokoro": re.compile(r"^[a-z][fm]_\w+$"),
    "openai": re.compile(r"^[a-z]+$"),
    "elevenlabs": re.compile(r"^[A-Za-z0-9]{16,}$"),
}


def voices_for(engine_name: str, language: str) -> dict[str, str]:
    """The voice map ("" narrator, "A" asks, "B" answers) for an engine and language."""
    settings = get_settings()
    table = DEFAULT_VOICES.get(engine_name) or DEFAULT_VOICES["kokoro"]
    solo, host_a, host_b = table.get(language) or table["en"]
    indian = language in {"hi", "hinglish"}
    overrides = {
        "": settings.tts_voice_hi if indian else settings.tts_voice_en,
        "A": settings.tts_voice_host_a_hi if indian else settings.tts_voice_host_a_en,
        "B": settings.tts_voice_host_b_hi if indian else settings.tts_voice_host_b_en,
    }
    shape = _VOICE_SHAPE.get(engine_name)
    chosen = {"": solo, "A": host_a, "B": host_b}
    for who, voice in overrides.items():
        if voice and (shape is None or shape.match(voice)):
            chosen[who] = voice
    return chosen


def build_engine(name: str, *, language: str = "en") -> TTSEngine:
    """Construct the requested engine, degrading rather than raising.

    A missing model must not stop an episode from being produced — a silent episode
    with a correct transcript is still useful, and the log says exactly why. The order
    of retreat is the requested engine, then local Kokoro, then silence.
    """
    if name in {"edge", "openai", "elevenlabs"}:
        engine: TTSEngine = {
            "edge": EdgeEngine,
            "openai": OpenAIEngine,
            "elevenlabs": ElevenLabsEngine,
        }[name]()
        if engine.available() and (name != "edge" or EdgeEngine.reachable()):
            return engine
        log.warning(
            "tts.unavailable",
            engine=name,
            fallback="kokoro",
            hint="needs a network (edge) or an API key (openai, elevenlabs)",
        )
        name = "kokoro"

    if name == "kokoro":
        kokoro = KokoroEngine(lang_code="h" if language in {"hi", "hinglish"} else "a")
        if kokoro.available():
            return kokoro
        onnx = KokoroOnnxEngine(lang="hi" if language in {"hi", "hinglish"} else "en-us")
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
