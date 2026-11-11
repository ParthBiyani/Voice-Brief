"""Which voice speaks, for which engine and language; and what Hinglish asks of the writer."""

from __future__ import annotations

import pytest

from voicebrief.config import get_settings
from voicebrief.languages import HINGLISH, is_indian, writing_instruction
from voicebrief.tts.engines import (
    ElevenLabsEngine,
    OpenAIEngine,
    build_engine,
    voices_for,
)


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch):
    settings = get_settings()
    for field in (
        "tts_voice_en",
        "tts_voice_hi",
        "tts_voice_host_a_en",
        "tts_voice_host_b_en",
        "tts_voice_host_a_hi",
        "tts_voice_host_b_hi",
    ):
        monkeypatch.setattr(settings, field, None)


class TestVoiceChoice:
    def test_hinglish_is_read_by_the_hindi_voices(self):
        voices = voices_for("edge", "hinglish")
        assert voices["A"] == "hi-IN-SwaraNeural"
        assert voices["B"] == "hi-IN-MadhurNeural"

    def test_a_kokoro_voice_left_in_an_old_env_is_not_forced_onto_another_engine(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "tts_voice_en", "af_heart")
        assert voices_for("edge", "en")[""].endswith("Neural")
        assert voices_for("kokoro", "en")[""] == "af_heart"

    def test_an_override_that_fits_the_engine_is_used(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "tts_voice_host_b_en", "en-US-BrianMultilingualNeural")
        assert voices_for("edge", "en")["B"] == "en-US-BrianMultilingualNeural"

    def test_every_engine_has_a_woman_and_a_man_for_every_language(self):
        for engine in ("edge", "openai", "elevenlabs", "kokoro"):
            for language in ("en", "hi", "hinglish"):
                voices = voices_for(engine, language)
                assert voices["A"] and voices["B"] and voices["A"] != voices["B"]


class TestEngineChoice:
    def test_keyed_engines_are_unavailable_without_a_key(self):
        assert not OpenAIEngine(api_key=None).available() or get_settings().openai_api_key
        assert OpenAIEngine(api_key="k").available()
        assert ElevenLabsEngine(api_key="k").available()

    def test_without_a_network_edge_falls_back_to_local_speech(self, monkeypatch):
        from voicebrief.tts import engines

        monkeypatch.setattr(engines.EdgeEngine, "reachable", staticmethod(lambda: False))
        assert build_engine("edge").name != "edge"

    def test_a_missing_key_falls_back_rather_than_failing(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "openai_api_key", None)
        assert build_engine("openai").name != "openai"


class TestHinglish:
    def test_it_asks_for_hindi_grammar_with_english_terms(self):
        text = writing_instruction("hinglish")
        assert text == HINGLISH
        assert "Devanagari" in text and "English" in text

    def test_english_adds_nothing_and_hindi_asks_for_hindi(self):
        assert writing_instruction("en") == ""
        assert "Hindi" in writing_instruction("hi")

    def test_both_indian_options_use_the_indian_voices(self):
        assert is_indian("hi") and is_indian("hinglish") and not is_indian("en")

    def test_the_verifier_still_checks_the_english_in_a_mixed_sentence(self):
        from voicebrief.pipeline.grounding import content_words

        words = content_words("ये library अब streaming support करती है")
        assert {"library", "streaming", "support"} <= words
