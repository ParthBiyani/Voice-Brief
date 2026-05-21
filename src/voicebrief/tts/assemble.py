"""Episode assembly.

Turns a list of written segments into one audio file plus a transcript whose
timestamps actually line up with it.

The timestamp guarantee is the point. A transcript with drifting timestamps is worse
than none — the user taps a line, hears the wrong thing, and stops trusting the
feature. So offsets are accumulated from measured clip durations rather than
estimated from word counts, and the concatenation writes frames in the same order the
offsets were computed.

Segments are cached by content hash. Regenerating an episode after a prompt tweak
re-synthesises only the segments whose text actually changed.
"""

from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass, field
from pathlib import Path

from voicebrief.logging import get_logger
from voicebrief.tts.base import SpeechClip, TTSEngine, TTSError, voice_key

log = get_logger(__name__)

# Silence inserted between segments, in seconds. Long enough to feel like a beat,
# short enough not to feel like a gap.
SEGMENT_GAP = 0.45


@dataclass(slots=True)
class TimedSegment:
    position: int
    kind: str
    heading: str
    script: str
    start_seconds: float
    end_seconds: float
    audio_path: Path | None = None
    citations: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def timestamp(self) -> str:
        minutes, seconds = divmod(int(self.start_seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass(slots=True)
class AssembledEpisode:
    audio_path: Path
    duration_seconds: float
    segments: list[TimedSegment]
    engine: str
    voice: str
    cache_hits: int = 0
    failures: list[str] = field(default_factory=list)

    def transcript(self) -> str:
        lines = []
        for segment in self.segments:
            lines.append(f"[{segment.timestamp()}] {segment.heading}")
            lines.append(segment.script)
            if segment.citations:
                lines.append(
                    "Sources: " + ", ".join(c.get("url", "") for c in segment.citations)
                )
            lines.append("")
        return "\n".join(lines).strip()


def synthesize_segments(
    segments: list,
    engine: TTSEngine,
    *,
    voice: str,
    cache_dir: Path,
    speed: float = 1.0,
) -> tuple[list[tuple[object, SpeechClip | None]], int]:
    """Render each segment, reusing cached audio where the text is unchanged."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[object, SpeechClip | None]] = []
    hits = 0

    for segment in segments:
        text = (segment.script or "").strip()
        if not text:
            results.append((segment, None))
            continue

        key = voice_key(text, voice, engine.name, speed)
        cached_path = cache_dir / f"{key}.wav"

        if cached_path.exists():
            hits += 1
            results.append(
                (
                    segment,
                    SpeechClip(
                        path=cached_path,
                        duration_seconds=_wav_duration(cached_path),
                        sample_rate=engine.sample_rate,
                        voice=voice,
                        engine=engine.name,
                        cached=True,
                    ),
                )
            )
            continue

        try:
            clip = engine.synthesize(text, voice=voice, out_path=cached_path, speed=speed)
            results.append((segment, clip))
        except TTSError as exc:
            # One segment failing costs that segment, not the episode.
            log.warning("tts.segment_failed", position=segment.position, error=str(exc))
            results.append((segment, None))

    log.info("tts.synthesized", segments=len(segments), cache_hits=hits)
    return results, hits


def assemble(
    segments: list,
    engine: TTSEngine,
    *,
    voice: str,
    out_path: Path,
    cache_dir: Path,
    speed: float = 1.0,
) -> AssembledEpisode:
    """Synthesize, concatenate, and produce timestamps that match the audio."""
    rendered, hits = synthesize_segments(
        segments, engine, voice=voice, cache_dir=cache_dir, speed=speed
    )

    timed: list[TimedSegment] = []
    failures: list[str] = []
    cursor = 0.0

    for segment, clip in rendered:
        duration = clip.duration_seconds if clip else 0.0
        if clip is None and (segment.script or "").strip():
            failures.append(f"segment {segment.position}: synthesis failed")

        timed.append(
            TimedSegment(
                position=segment.position,
                kind=segment.kind,
                heading=segment.heading,
                script=segment.script,
                start_seconds=round(cursor, 3),
                end_seconds=round(cursor + duration, 3),
                audio_path=clip.path if clip else None,
                citations=list(getattr(segment, "citations", []) or []),
            )
        )
        # The gap is added to the cursor only when a clip follows, so the last
        # segment's end time equals the episode duration exactly.
        cursor += duration
        if clip is not None:
            cursor += SEGMENT_GAP

    total = _concatenate(
        [clip for _, clip in rendered if clip is not None],
        out_path=out_path,
        sample_rate=engine.sample_rate,
        gap_seconds=SEGMENT_GAP,
    )

    return AssembledEpisode(
        audio_path=out_path,
        duration_seconds=round(total, 3),
        segments=timed,
        engine=engine.name,
        voice=voice,
        cache_hits=hits,
        failures=failures,
    )


def _concatenate(
    clips: list[SpeechClip], *, out_path: Path, sample_rate: int, gap_seconds: float
) -> float:
    """Join clips into one WAV, inserting a gap between them."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not clips:
        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
        return 0.0

    if len(clips) == 1:
        shutil.copyfile(clips[0].path, out_path)
        return clips[0].duration_seconds

    gap_frames = int(gap_seconds * sample_rate)
    silence = b"\x00\x00" * gap_frames
    total_frames = 0

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)

        for index, clip in enumerate(clips):
            with wave.open(str(clip.path), "rb") as src:
                if src.getframerate() != sample_rate:
                    # Mixing sample rates would desynchronise every timestamp after
                    # this point, so it is a hard error rather than a silent resample.
                    raise TTSError(
                        f"{clip.path.name} is {src.getframerate()}Hz but the episode "
                        f"is {sample_rate}Hz; timestamps would drift"
                    )
                frames = src.readframes(src.getnframes())
            out.writeframes(frames)
            total_frames += len(frames) // 2

            if index < len(clips) - 1:
                out.writeframes(silence)
                total_frames += gap_frames

    return total_frames / sample_rate


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())
