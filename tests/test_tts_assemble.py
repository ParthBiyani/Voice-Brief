"""Episode assembly tests.

The property under test throughout is that transcript timestamps match the audio. A
transcript that drifts is worse than no transcript — the listener taps a line, hears
the wrong thing, and stops trusting the feature.

Run against the silent engine, which is exactly why that engine exists: every timing
calculation is exercised with no audio stack present.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from voicebrief.tts.assemble import SEGMENT_GAP, assemble, synthesize_segments
from voicebrief.tts.base import TTSError, voice_key
from voicebrief.tts.engines import SilentEngine, build_engine


@dataclass
class FakeSegment:
    position: int
    kind: str
    heading: str
    script: str
    citations: list = field(default_factory=list)


def segments(*scripts: str) -> list[FakeSegment]:
    return [
        FakeSegment(position=i, kind="story", heading=f"Story {i}", script=s)
        for i, s in enumerate(scripts)
    ]


@pytest.fixture
def engine() -> SilentEngine:
    return SilentEngine()


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "episode.wav", tmp_path / "cache"


class TestVoiceKey:
    def test_same_inputs_give_the_same_key(self):
        assert voice_key("hello", "af_heart", "kokoro") == voice_key(
            "hello", "af_heart", "kokoro"
        )

    def test_different_voice_gives_a_different_key(self):
        """A cache keyed on text alone would serve the wrong voice after a change."""
        assert voice_key("hello", "af_heart", "kokoro") != voice_key(
            "hello", "am_adam", "kokoro"
        )

    def test_different_engine_gives_a_different_key(self):
        assert voice_key("hello", "v", "kokoro") != voice_key("hello", "v", "piper")

    def test_whitespace_is_normalised(self):
        assert voice_key("  hello  ", "v", "e") == voice_key("hello", "v", "e")


class TestSynthesis:
    def test_produces_a_clip_per_segment(self, engine, workspace):
        _, cache = workspace
        results, _ = synthesize_segments(
            segments("one two three", "four five six"), engine, voice="v", cache_dir=cache
        )
        assert len(results) == 2
        assert all(clip is not None for _, clip in results)

    def test_empty_segment_yields_no_clip(self, engine, workspace):
        _, cache = workspace
        results, _ = synthesize_segments(segments("", "real text here"), engine,
                                         voice="v", cache_dir=cache)
        assert results[0][1] is None
        assert results[1][1] is not None

    def test_second_run_hits_the_cache(self, engine, workspace):
        _, cache = workspace
        text = segments("the same words every time")
        _, first = synthesize_segments(text, engine, voice="v", cache_dir=cache)
        _, second = synthesize_segments(text, engine, voice="v", cache_dir=cache)
        assert first == 0
        assert second == 1

    def test_changed_text_misses_the_cache(self, engine, workspace):
        _, cache = workspace
        synthesize_segments(segments("original wording"), engine, voice="v", cache_dir=cache)
        _, hits = synthesize_segments(segments("revised wording"), engine,
                                      voice="v", cache_dir=cache)
        assert hits == 0

    def test_longer_text_takes_longer(self, engine, workspace):
        _, cache = workspace
        results, _ = synthesize_segments(
            segments("short", " ".join(["word"] * 200)), engine, voice="v", cache_dir=cache
        )
        assert results[1][1].duration_seconds > results[0][1].duration_seconds


class TestAssembly:
    def test_timestamps_are_contiguous_and_ordered(self, engine, workspace):
        out, cache = workspace
        episode = assemble(
            segments("first segment words", "second segment words", "third segment words"),
            engine, voice="v", out_path=out, cache_dir=cache,
        )
        starts = [s.start_seconds for s in episode.segments]
        assert starts == sorted(starts)
        for earlier, later in zip(episode.segments, episode.segments[1:], strict=False):
            assert later.start_seconds >= earlier.end_seconds

    def test_segment_offsets_account_for_the_inter_segment_gap(self, engine, workspace):
        out, cache = workspace
        episode = assemble(
            segments("first segment words", "second segment words"),
            engine, voice="v", out_path=out, cache_dir=cache,
        )
        first, second = episode.segments
        assert second.start_seconds == pytest.approx(first.end_seconds + SEGMENT_GAP, abs=0.01)

    def test_reported_duration_matches_the_rendered_file(self, engine, workspace):
        """The guarantee that makes timestamps trustworthy."""
        out, cache = workspace
        episode = assemble(
            segments("one two three four", "five six seven eight", "nine ten"),
            engine, voice="v", out_path=out, cache_dir=cache,
        )
        with wave.open(str(out), "rb") as handle:
            actual = handle.getnframes() / handle.getframerate()
        assert episode.duration_seconds == pytest.approx(actual, abs=0.05)

    def test_last_segment_ends_at_the_episode_duration(self, engine, workspace):
        out, cache = workspace
        episode = assemble(
            segments("alpha beta gamma", "delta epsilon zeta"),
            engine, voice="v", out_path=out, cache_dir=cache,
        )
        assert episode.segments[-1].end_seconds == pytest.approx(
            episode.duration_seconds, abs=0.05
        )

    def test_single_segment_episode(self, engine, workspace):
        out, cache = workspace
        episode = assemble(segments("only one here"), engine, voice="v",
                           out_path=out, cache_dir=cache)
        assert episode.duration_seconds > 0
        assert out.exists()

    def test_empty_episode_writes_a_valid_file(self, engine, workspace):
        out, cache = workspace
        episode = assemble([], engine, voice="v", out_path=out, cache_dir=cache)
        assert episode.duration_seconds == 0.0
        assert out.exists()

    def test_failed_segment_is_reported_and_skipped(self, workspace):
        out, cache = workspace

        class BrokenEngine(SilentEngine):
            def synthesize(self, text, *, voice, out_path, speed=1.0):
                if "poison" in text:
                    raise TTSError("engine blew up")
                return super().synthesize(text, voice=voice, out_path=out_path, speed=speed)

        episode = assemble(
            segments("good segment here", "poison segment here", "another good one"),
            BrokenEngine(), voice="v", out_path=out, cache_dir=cache,
        )
        assert len(episode.failures) == 1
        assert episode.duration_seconds > 0, "the episode survives one bad segment"

    def test_mismatched_sample_rates_are_rejected(self, workspace, tmp_path):
        """Silently resampling would desynchronise every later timestamp."""
        out, cache = workspace
        cache.mkdir(parents=True, exist_ok=True)

        engine = SilentEngine()
        clips, _ = synthesize_segments(segments("first one here", "second one here"),
                                       engine, voice="v", cache_dir=cache)
        # Rewrite one clip at a different rate.
        odd = clips[1][1].path
        with wave.open(str(odd), "rb") as src:
            frames = src.readframes(src.getnframes())
        with wave.open(str(odd), "wb") as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(16_000)
            dst.writeframes(frames)

        with pytest.raises(TTSError, match="timestamps would drift"):
            assemble(segments("first one here", "second one here"), engine,
                     voice="v", out_path=out, cache_dir=cache)


class TestTranscript:
    def test_includes_timestamps_and_headings(self, engine, workspace):
        out, cache = workspace
        episode = assemble(segments("alpha beta gamma", "delta epsilon"), engine,
                           voice="v", out_path=out, cache_dir=cache)
        transcript = episode.transcript()
        assert "[00:00]" in transcript
        assert "Story 0" in transcript

    def test_includes_citations_when_present(self, engine, workspace):
        out, cache = workspace
        segs = segments("alpha beta gamma")
        segs[0].citations = [{"url": "https://example.com/a"}]
        episode = assemble(segs, engine, voice="v", out_path=out, cache_dir=cache)
        assert "https://example.com/a" in episode.transcript()


class TestEngineSelection:
    def test_missing_engine_degrades_to_silence(self):
        """A missing model must not stop an episode being produced."""
        assert build_engine("piper").name == "null"

    def test_unknown_engine_name_degrades(self):
        assert build_engine("does-not-exist").name == "null"

    def test_silent_engine_is_always_available(self):
        assert SilentEngine().available()
