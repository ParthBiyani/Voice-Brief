"""Multilingual behaviour.

Hindi is generated natively rather than translated (PRD §6): translated-then-spoken
audio carries English sentence rhythm and sounds wrong. These tests check the wiring
that makes that possible — language selects the voice, the engine's language model,
and the prompt instruction — without requiring a live model.
"""

from __future__ import annotations

import pytest

from voicebrief.config import get_settings
from voicebrief.pipeline.grounding import split_sentences
from voicebrief.tts.engines import KokoroEngine, build_engine


class TestVoiceSelection:
    def test_hindi_selects_the_hindi_language_model(self):
        engine = KokoroEngine(lang_code="h")
        assert engine.lang_code == "h"

    def test_english_is_the_default(self):
        assert KokoroEngine().lang_code == "a"

    def test_build_engine_threads_language_through(self):
        engine = build_engine("kokoro", language="hi")
        # Degrades to the silent engine when weights are absent; either way the
        # language must not be silently dropped.
        assert engine.name in ("kokoro", "null")
        if engine.name == "kokoro":
            assert engine.lang_code == "h"

    def test_hindi_voices_are_registered(self):
        voices = KokoroEngine().voices()
        assert any(v.startswith("hf_") or v.startswith("hm_") for v in voices)

    def test_settings_expose_a_voice_per_language(self):
        settings = get_settings()
        assert settings.tts_voice_en
        assert settings.tts_voice_hi
        assert settings.tts_voice_en != settings.tts_voice_hi


class TestDevanagariHandling:
    def test_sentence_splitting_handles_devanagari(self):
        """The splitter must recognise a Devanagari capital as a sentence start, or
        the whole Hindi transcript collapses into one line."""
        text = "यह पहला वाक्य है। दूसरा वाक्य यहाँ है। तीसरा भी।"
        assert len(split_sentences(text)) >= 2

    def test_mixed_hinglish_splits_correctly(self):
        """Real output is Hinglish: English technical terms, Hindi connective tissue."""
        text = (
            "LangGraph version zero point four जारी हुआ है। "
            "अब graph checkpoints process restart के बाद भी बने रहते हैं। "
            "This matters for durable execution."
        )
        assert len(split_sentences(text)) == 3

    @pytest.mark.parametrize("language", ["en", "hi"])
    def test_both_languages_are_accepted_by_the_engine_builder(self, language):
        assert build_engine("kokoro", language=language) is not None
