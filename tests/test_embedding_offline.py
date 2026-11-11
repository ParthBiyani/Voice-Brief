"""The embedding model loads from disk when it can, and only goes online when it must."""

from __future__ import annotations

import pytest

from voicebrief.pipeline.embedding import _load_model


def test_a_cached_model_is_loaded_without_asking_the_hub():
    seen = []

    def factory(name, **kwargs):
        seen.append(kwargs)
        return "model"

    assert _load_model(factory, "m") == "model"
    assert seen == [{"device": "cpu", "local_files_only": True}]


def test_a_missing_model_falls_back_to_downloading():
    seen = []

    def factory(name, **kwargs):
        seen.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return "downloaded"

    assert _load_model(factory, "m") == "downloaded"
    assert seen[-1] == {"device": "cpu"}


def test_a_real_failure_while_downloading_is_not_hidden():
    def factory(name, **kwargs):
        raise OSError("no network")

    with pytest.raises(OSError):
        _load_model(factory, "m")
