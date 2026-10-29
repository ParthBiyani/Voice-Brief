"""Engine selection and the Kokoro ONNX fallback (no model download needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voicebrief.config import get_settings
from voicebrief.tts.base import TTSError
from voicebrief.tts.engines import (
    ONNX_MODEL_FILE,
    ONNX_VOICES_FILE,
    KokoroOnnxEngine,
    SilentEngine,
    build_engine,
    model_dir,
)


@pytest.fixture
def empty_models(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"tts_model_dir": str(tmp_path)})
    monkeypatch.setattr("voicebrief.tts.engines.get_settings", lambda: settings)
    return tmp_path


def put_models(folder):
    (folder / ONNX_MODEL_FILE).write_bytes(b"x")
    (folder / ONNX_VOICES_FILE).write_bytes(b"x")


class TestModelFolder:
    def test_a_configured_folder_wins(self, empty_models):
        assert model_dir() == empty_models

    def test_the_default_is_per_user_not_the_repo(self, monkeypatch):
        settings = get_settings().model_copy(update={"tts_model_dir": None})
        monkeypatch.setattr("voicebrief.tts.engines.get_settings", lambda: settings)
        monkeypatch.setenv("LOCALAPPDATA", "C:/Users/someone/AppData/Local")
        assert model_dir() == Path("C:/Users/someone/AppData/Local/VoiceBrief/models")


class TestOnnxEngine:
    def test_unavailable_until_both_model_files_exist(self, empty_models):
        engine = KokoroOnnxEngine()
        assert not engine.available()
        (empty_models / ONNX_MODEL_FILE).write_bytes(b"x")
        assert not engine.available()
        (empty_models / ONNX_VOICES_FILE).write_bytes(b"x")
        assert engine.available()

    def test_empty_text_is_refused(self, empty_models):
        with pytest.raises(TTSError):
            KokoroOnnxEngine().synthesize("  ", voice="af_heart", out_path=empty_models / "a.wav")


class TestBuildEngine:
    def test_without_a_model_it_degrades_to_silence_rather_than_failing(self, empty_models):
        assert isinstance(build_engine("kokoro"), SilentEngine)

    def test_with_the_model_files_present_the_onnx_engine_is_chosen(
        self, empty_models, monkeypatch
    ):
        put_models(empty_models)
        monkeypatch.setattr("voicebrief.tts.engines.KokoroEngine.available", lambda self: False)
        engine = build_engine("kokoro")
        assert isinstance(engine, KokoroOnnxEngine)
        assert engine.lang == "en-us"

    def test_hindi_selects_the_hindi_language(self, empty_models, monkeypatch):
        put_models(empty_models)
        monkeypatch.setattr("voicebrief.tts.engines.KokoroEngine.available", lambda self: False)
        assert build_engine("kokoro", language="hi").lang == "hi"
