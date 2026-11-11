"""Two hosts, two voices, and speech paced like speech."""

from __future__ import annotations

import itertools
import wave
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from voicebrief.graphs.brief import DIALOGUE_RULES, BriefGraph, Segment
from voicebrief.tts.assemble import assemble
from voicebrief.tts.base import SpeechClip, TTSEngine
from voicebrief.tts.dialogue import (
    PAUSE_QUESTION,
    PAUSE_QUICK_REPLY,
    PAUSE_STATEMENT,
    PAUSE_TURN,
    has_speakers,
    parse_turns,
    render_script,
    sentence_pause,
    sentence_rate,
    shape_samples,
    split_sentences,
)

RATE = 24_000


class ToneEngine(TTSEngine):
    """A tone per voice, 0.5 s per sentence, with silence padding: records every call."""

    name = "tone"
    sample_rate = RATE
    FREQ: ClassVar[dict[str, float]] = {"fem": 220.0, "masc": 120.0}

    def __init__(self):
        self.calls: list[tuple[str, str, float]] = []

    def available(self):
        return True

    def synthesize(self, text, *, voice, out_path, speed=1.0):
        self.calls.append((text, voice, speed))
        t = np.arange(int(0.5 * RATE)) / RATE
        tone = 0.3 * np.sin(2 * np.pi * self.FREQ.get(voice, 150.0) * t)
        pad = np.zeros(int(0.2 * RATE))
        pcm = (np.concatenate([pad, tone, pad]) * 32767).astype("<i2")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes(pcm.tobytes())
        return SpeechClip(out_path, 0.9, RATE, voice, self.name)


VOICES = {"": "fem", "A": "fem", "B": "masc"}
DIALOGUE = (
    "HOST A: What changed in this release?\n"
    "HOST B: The run object gained a cancel reason. Both state calls take a select option.\n"
    "HOST A: Right.\n"
    "HOST B: And the stream example now shows the run status."
)


class TestParsing:
    def test_labels_become_speakers_and_are_not_part_of_the_text(self):
        turns = parse_turns(DIALOGUE)
        assert [t.speaker for t in turns] == ["A", "B", "A", "B"]
        assert all("HOST" not in t.text for t in turns)
        assert turns[0].text == "What changed in this release?"

    def test_text_without_labels_is_one_narrator_turn(self):
        turns = parse_turns("Just one voice here. Two sentences.")
        assert [(t.speaker, t.text) for t in turns] == [("", "Just one voice here. Two sentences.")]
        assert not has_speakers("Just one voice here.")
        assert has_speakers(DIALOGUE)

    def test_a_turn_can_run_over_several_lines(self):
        turns = parse_turns("HOST A: First part\nsecond part.\nHOST B: Reply.")
        assert turns[0].text == "First part second part."

    def test_markdown_bold_labels_and_speaker_numbers_are_understood(self):
        turns = parse_turns("**HOST A:** Hello?\nSPEAKER 2 - Hi.")
        assert [t.speaker for t in turns] == ["A", "B"]

    def test_sentences_split_on_end_marks_but_not_inside_versions(self):
        assert split_sentences("Version 1.4.9 is out. Is it safe? Yes!") == [
            "Version 1.4.9 is out.",
            "Is it safe?",
            "Yes!",
        ]

    def test_hindi_sentences_split_on_the_danda(self):
        assert split_sentences("यह पहला वाक्य है। Streaming अब default है। क्या यह सही है?") == [
            "यह पहला वाक्य है।",
            "Streaming अब default है।",
            "क्या यह सही है?",
        ]


class TestPacing:
    def test_a_question_leaves_more_room_than_a_statement(self):
        assert sentence_pause("Is it safe?") > sentence_pause("It is safe.")

    def test_questions_are_lighter_and_short_lines_get_weight(self):
        assert sentence_rate("What changed in the release?") > sentence_rate(
            "The release changed three things for people who use the library."
        )
        assert sentence_rate("Okay, right.") < 1.0

    def test_samples_are_levelled_trimmed_and_faded(self):
        quiet = np.concatenate(
            [np.zeros(RATE // 5), 0.02 * np.ones(RATE // 2), np.zeros(RATE // 5)]
        )
        shaped = shape_samples(quiet.astype("float32"), RATE)
        assert len(shaped) < len(quiet)  # padding gone
        assert abs(float(np.sqrt(np.mean(shaped**2))) - 0.085) < 0.02  # brought up to level
        assert shaped[0] == 0.0 and abs(shaped[-1]) < 1e-6  # no clicks


class TestRendering:
    @pytest.fixture
    def rendered(self, tmp_path):
        engine = ToneEngine()
        clip = render_script(
            DIALOGUE,
            engine,
            voices=VOICES,
            out_path=tmp_path / "seg.wav",
            cache_dir=tmp_path / "cache",
        )
        return engine, clip

    def test_each_host_speaks_in_their_own_voice(self, rendered):
        engine, _ = rendered
        spoken = {text: voice for text, voice, _ in engine.calls}
        assert spoken["What changed in this release?"] == "fem"
        assert spoken["The run object gained a cancel reason."] == "masc"
        assert spoken["Right."] == "fem"
        assert spoken["And the stream example now shows the run status."] == "masc"

    def test_the_labels_are_never_spoken(self, rendered):
        engine, _ = rendered
        assert not any("HOST" in text for text, _, _ in engine.calls)

    def test_it_is_spoken_sentence_by_sentence_at_varied_pace(self, rendered):
        engine, _ = rendered
        assert len(engine.calls) == 5
        speeds = {round(speed, 2) for _, _, speed in engine.calls}
        assert len(speeds) >= 2

    def test_the_gaps_follow_who_speaks_and_how(self, rendered):
        _, clip = rendered
        # Five sentences of 0.5 s once trimmed (0.04 s edge each side on the tone, which
        # starts 0.2 s in), plus the pauses between them.
        per_sentence = 0.5 + 0.08
        pauses = (
            PAUSE_TURN  # A -> B
            + PAUSE_STATEMENT  # inside B's two-sentence turn
            + PAUSE_TURN  # B -> A (a long turn, so a full turn gap)
            + PAUSE_QUICK_REPLY  # "Right." is short: B comes in fast
        )
        assert clip.duration_seconds == pytest.approx(5 * per_sentence + pauses, abs=0.15)
        assert clip.voice == "fem+masc"
        assert PAUSE_QUESTION > PAUSE_STATEMENT

    def test_a_cached_sentence_is_not_rendered_twice(self, tmp_path):
        engine = ToneEngine()
        for _ in range(2):
            render_script(
                DIALOGUE,
                engine,
                voices=VOICES,
                out_path=tmp_path / "seg.wav",
                cache_dir=tmp_path / "cache",
            )
        assert len(engine.calls) == 5

    def test_a_solo_script_uses_the_narrator_voice(self, tmp_path):
        engine = ToneEngine()
        render_script(
            "One narrator. Three short sentences. Nothing else.",
            engine,
            voices=VOICES,
            out_path=tmp_path / "s.wav",
            cache_dir=tmp_path / "c",
        )
        assert {voice for _, voice, _ in engine.calls} == {"fem"}


class TestInAnEpisode:
    def test_assembly_voices_each_segment_and_keeps_timestamps_honest(self, tmp_path):
        engine = ToneEngine()
        segs = [
            SimpleNamespace(position=0, kind="agenda", heading="Agenda", script=DIALOGUE),
            SimpleNamespace(position=1, kind="story", heading="Story", script="Solo line one."),
        ]
        episode = assemble(
            segs,
            engine,
            voice="fem",
            voices=VOICES,
            out_path=tmp_path / "ep.wav",
            cache_dir=tmp_path / "cache",
        )
        first, second = episode.segments
        assert first.start_seconds == 0.0
        assert second.start_seconds == pytest.approx(first.end_seconds + 0.45, abs=0.01)
        with wave.open(str(episode.audio_path), "rb") as handle:
            actual = handle.getnframes() / handle.getframerate()
        assert actual == pytest.approx(episode.duration_seconds, abs=0.05)


class TestScriptingTheConversation:
    @pytest.fixture
    def graph(self, monkeypatch):
        monkeypatch.setattr(
            BriefGraph,
            "_simple_segment",
            lambda self, prompt, kind, heading, max_tokens: Segment(0, kind, heading, "x"),
        )
        monkeypatch.setattr(
            BriefGraph,
            "_write_story",
            lambda self, *a: Segment(1, "story", "h", "HOST A: q?"),
        )
        graph = BriefGraph(SimpleNamespace())
        return graph

    def write(self, graph, style):
        summaries = [(None, SimpleNamespace(headline="A story"))]
        graph.write_node({"summaries": summaries, "style": style})
        return graph.active_system

    def test_two_hosts_get_the_conversation_rules(self, graph):
        system = self.write(graph, "two_host")
        assert DIALOGUE_RULES in system
        assert "HOST A is the curious one" in system

    def test_one_voice_does_not(self, graph):
        assert DIALOGUE_RULES not in self.write(graph, "solo_anchor")

    def test_the_rules_keep_grounding_and_ban_verdicts(self):
        assert "FACTS only" in DIALOGUE_RULES
        assert "not a verdict" in DIALOGUE_RULES
        assert "asks at least" in DIALOGUE_RULES.replace("\n", " ")


def test_default_hosts_are_one_woman_and_one_man(monkeypatch):
    from voicebrief.config import get_settings
    from voicebrief.tts.engines import voices_for

    settings = get_settings()
    for field in ("tts_voice_en", "tts_voice_host_a_en", "tts_voice_host_b_en"):
        monkeypatch.setattr(settings, field, None)
    edge = voices_for("edge", "en")
    assert edge["A"].startswith("en-US-Ava")  # female
    assert edge["B"].startswith("en-US-Andrew")  # male
    kokoro = voices_for("kokoro", "en")
    assert kokoro["A"].startswith("af_") and kokoro["B"].startswith("am_")
    assert voices_for("kokoro", "hinglish")["B"].startswith("hm_")


class TestSpeakable:
    @pytest.mark.parametrize(
        ("written", "spoken"),
        [
            (
                "requires langchain-core 1.7.5 or later",
                "requires langchain-core 1 point 7 point 5 or later",
            ),
            ("threads.get_state takes select", "threads dot get state takes select"),
            ("a cancel_reason field", "a cancel reason field"),
            ("see setup.py for details", "see setup.py for details"),
            ("it costs $67.16 today", "it costs $67.16 today"),
        ],
    )
    def test_versions_and_code_names_become_words(self, written, spoken):
        from voicebrief.tts.dialogue import speakable

        assert speakable(written) == spoken

    def test_what_the_engine_is_given_is_the_spoken_form(self, tmp_path):
        engine = ToneEngine()
        render_script(
            "HOST B: It needs langchain-core 1.7.5.",
            engine,
            voices=VOICES,
            out_path=tmp_path / "s.wav",
            cache_dir=tmp_path / "c",
        )
        assert engine.calls[0][0] == "It needs langchain-core 1 point 7 point 5."


class TestSentenceTimings:
    """The web player seeks by these, so they must match the audio that was laid down."""

    def test_every_sentence_has_a_cue_that_matches_where_it_sits(self, tmp_path):
        engine = ToneEngine()
        clip = render_script(
            DIALOGUE, engine, voices=VOICES, out_path=tmp_path / "s.wav", cache_dir=tmp_path / "c"
        )
        cues = clip.meta["cues"]
        assert [c["speaker"] for c in cues] == ["A", "B", "B", "A", "B"]
        assert cues[0]["text"] == "What changed in this release?"
        assert cues[0]["start"] == 0.0
        for earlier, later in itertools.pairwise(cues):
            assert earlier["end"] <= later["start"]  # a pause lies between sentences
        assert cues[-1]["end"] == pytest.approx(clip.duration_seconds, abs=0.01)

    def test_the_cue_shows_the_written_text_not_the_spoken_form(self, tmp_path):
        engine = ToneEngine()
        clip = render_script(
            "HOST B: It needs langchain-core 1.7.5.",
            engine,
            voices=VOICES,
            out_path=tmp_path / "s.wav",
            cache_dir=tmp_path / "c",
        )
        assert clip.meta["cues"][0]["text"] == "It needs langchain-core 1.7.5."

    def test_assembly_puts_cues_on_the_episode_clock(self, tmp_path):
        engine = ToneEngine()
        segs = [
            SimpleNamespace(position=0, kind="agenda", heading="A", script="First one."),
            SimpleNamespace(position=1, kind="story", heading="S", script="Second one."),
        ]
        episode = assemble(
            segs,
            engine,
            voice="fem",
            voices=VOICES,
            out_path=tmp_path / "e.wav",
            cache_dir=tmp_path / "c",
        )
        first, second = episode.segments
        assert first.cues[0]["start"] == 0.0
        assert second.cues[0]["start"] == pytest.approx(second.start_seconds, abs=0.01)
        assert second.cues[-1]["end"] == pytest.approx(second.end_seconds, abs=0.01)
