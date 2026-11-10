"""Spoken delivery: who says what, and how it is paced.

A two-host script is a list of turns, each marked "HOST A:" or "HOST B:". Those labels are
instructions to this module, never words to say. Each turn is voiced by its host's voice
(a woman asks, a man answers), and every turn, and every solo segment too, is spoken
sentence by sentence rather than as one block. Rendering a long passage in a single call is
what made the voice sound flat and robotic: the model reads it with one steady rhythm, with
no breath between sentences. Here each sentence is rendered on its own, trimmed of the
engine's own padding, levelled to one loudness so the two hosts match, and put back with
pauses that depend on what was said: a question leaves more room than a statement, a
change of speaker more than a continuation, and a one-word reply comes in quickly.
"""

from __future__ import annotations

import hashlib
import re
import wave
from dataclasses import dataclass
from pathlib import Path

from voicebrief.logging import get_logger
from voicebrief.tts.base import SpeechClip, TTSEngine, TTSError

log = get_logger(__name__)

SPEAKER_LABEL = re.compile(
    r"^\s*(?:\*\*)?(HOST|SPEAKER)\s*([AB12])(?:\*\*)?\s*[:\-]\s*", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[]?[A-Z0-9])")

# Pause after a sentence, in seconds, by how it ends.
PAUSE_STATEMENT = 0.20
PAUSE_QUESTION = 0.30
PAUSE_EXCLAIM = 0.24
# Between speakers: someone finishing, someone else starting.
PAUSE_TURN = 0.38
PAUSE_QUICK_REPLY = 0.16  # after a very short turn ("Right."), the reply comes in fast
QUICK_REPLY_WORDS = 3

# Speaking rate varies a little by sentence kind, so no two sentences share a tempo.
RATE_STATEMENT = 1.0
RATE_QUESTION = 1.04  # questions are lighter and quicker
RATE_SHORT = 0.97  # a short line is given a touch more weight

# Every clip is brought to this RMS (of full scale) so two voices sit at the same level.
TARGET_RMS = 0.085
MAX_GAIN = 4.0
FADE_SECONDS = 0.01
# Padding the engine puts around speech is trimmed below this amplitude.
TRIM_THRESHOLD = 0.012
KEEP_EDGE_SECONDS = 0.04


_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_VERSION = re.compile(r"(?<![\w.$])v?(\d+(?:\.\d+)+)(?![\w.]*\w)")
_IDENT_DOT = re.compile(r"(?<=[A-Za-z])\.(?=[a-z_]{2,})")
_FILE_EXT = frozenset(
    {"py", "js", "ts", "md", "json", "yaml", "yml", "txt", "toml", "io", "com", "org", "ai", "dev"}
)


def _say_version(match: re.Match) -> str:
    """1.4.9 -> "one point four point nine" (each part read as a number)."""
    spoken = []
    for part in match.group(1).split("."):
        spoken.append(" ".join(_DIGIT_WORDS[int(c)] for c in part) if len(part) > 2 else part)
    return " point ".join(spoken)


def speakable(text: str) -> str:
    """What a voice should be given: versions and code names turned into words.

    The writer is told to spell numbers out, but it does not always ("version 1.4.9",
    "threads.get_state"), and engines read those unpredictably or letter by letter.
    """
    text = _VERSION.sub(_say_version, text)

    def dotted(match: re.Match) -> str:
        after = text[match.end() : match.end() + 4].split()[:1]
        return "." if after and after[0].rstrip(".,") in _FILE_EXT else " dot "

    text = _IDENT_DOT.sub(dotted, text)
    return text.replace("_", " ")


@dataclass(frozen=True, slots=True)
class Turn:
    speaker: str  # "A" | "B" | "" for a single narrator
    text: str


def speaker_key(label: str) -> str:
    return {"A": "A", "1": "A", "B": "B", "2": "B"}.get(label.upper(), "")


def has_speakers(script: str) -> bool:
    return any(turn.speaker for turn in parse_turns(script))


def parse_turns(script: str) -> list[Turn]:
    """Split a script into turns. Text with no speaker labels is one unlabelled turn."""
    turns: list[Turn] = []
    current: list[str] = []
    speaker = ""

    def flush() -> None:
        text = " ".join(" ".join(current).split())
        if text:
            turns.append(Turn(speaker, text))
        current.clear()

    for raw in (script or "").splitlines():
        match = SPEAKER_LABEL.match(raw)
        if match:
            flush()
            speaker = speaker_key(match.group(2))
            raw = raw[match.end() :]
        current.append(raw.strip())
    flush()
    return turns


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(" ".join(text.split())) if s.strip()]


def sentence_rate(sentence: str) -> float:
    if sentence.rstrip().endswith("?"):
        return RATE_QUESTION
    if len(sentence.split()) <= 4:
        return RATE_SHORT
    return RATE_STATEMENT


def sentence_pause(sentence: str) -> float:
    end = sentence.rstrip()[-1:]
    if end == "?":
        return PAUSE_QUESTION
    if end == "!":
        return PAUSE_EXCLAIM
    return PAUSE_STATEMENT


def _read_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
    if width != 2:
        raise TTSError(f"{path.name}: expected 16-bit audio, got {8 * width}-bit")
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0, rate


def shape_samples(samples, rate: int):
    """Trim the engine's padding, level the clip, and soften its edges."""
    import numpy as np

    if len(samples) == 0:
        return samples
    loud = np.flatnonzero(np.abs(samples) > TRIM_THRESHOLD)
    if loud.size:
        keep = int(KEEP_EDGE_SECONDS * rate)
        samples = samples[max(int(loud[0]) - keep, 0) : int(loud[-1]) + keep + 1]

    rms = float(np.sqrt(np.mean(np.square(samples)))) or 1e-9
    gain = min(TARGET_RMS / rms, MAX_GAIN)
    samples = np.clip(samples * gain, -0.98, 0.98)

    fade = min(int(FADE_SECONDS * rate), len(samples) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    return samples


def _write_wav(path: Path, samples, rate: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _sentence_key(text: str, voice: str, engine: str, speed: float) -> str:
    basis = f"v2|{engine}|{voice}|{speed:.3f}|{text.strip()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:40]


def render_script(
    script: str,
    engine: TTSEngine,
    *,
    voices: dict[str, str],
    out_path: Path,
    cache_dir: Path,
    base_speed: float = 1.0,
) -> SpeechClip:
    """Voice one segment: turn by turn, sentence by sentence, as one WAV.

    `voices` maps "A", "B" and "" (the narrator) to voice names; anything missing falls
    back to the narrator's. Raises TTSError when nothing could be rendered, so the
    caller's failure isolation still applies.
    """
    import numpy as np

    turns = parse_turns(script)
    if not turns:
        raise TTSError("nothing to say")

    narrator = voices.get("", "") or next(iter(voices.values()), "")
    pieces: list = []
    rate = engine.sample_rate
    previous_speaker: str | None = None
    previous_short = False
    used: list[str] = []

    for turn in turns:
        voice = voices.get(turn.speaker) or narrator
        if voice not in used:
            used.append(voice)

        if pieces:
            if previous_speaker is not None and turn.speaker != previous_speaker:
                gap = PAUSE_QUICK_REPLY if previous_short else PAUSE_TURN
            else:
                gap = 0.0  # a continuing turn is paced by its own sentences
            pieces.append(np.zeros(int(gap * rate), dtype="float32"))

        sentences = split_sentences(turn.text) or [turn.text]
        for index, sentence in enumerate(sentences):
            speed = base_speed * sentence_rate(sentence)
            sentence = speakable(sentence)
            key = _sentence_key(sentence, voice, engine.name, speed)
            cached = cache_dir / f"{key}.wav"
            if not cached.exists():
                engine.synthesize(sentence, voice=voice, out_path=cached, speed=speed)
            samples, clip_rate = _read_wav(cached)
            if clip_rate != rate:
                raise TTSError(f"{cached.name} is {clip_rate}Hz, expected {rate}Hz")
            pieces.append(shape_samples(samples, rate))
            if index < len(sentences) - 1:
                pieces.append(np.zeros(int(sentence_pause(sentence) * rate), dtype="float32"))

        previous_speaker = turn.speaker
        previous_short = len(turn.text.split()) <= QUICK_REPLY_WORDS

    audio = np.concatenate(pieces)
    _write_wav(out_path, audio, rate)
    return SpeechClip(
        path=out_path,
        duration_seconds=len(audio) / float(rate),
        sample_rate=rate,
        voice="+".join(used),
        engine=engine.name,
        meta={"turns": len(turns)},
    )
