"""The setup doctor: one clear line per problem, and a useful exit code."""

from __future__ import annotations

from voicebrief.config import get_settings
from voicebrief.doctor import Check, check_api_key, check_python, exit_code, run_checks


def test_a_required_failure_fails_the_run_and_an_optional_one_does_not():
    ok = Check("a", True, "fine")
    advice = Check("b", False, "not set", "do this", required=False)
    broken = Check("c", False, "down", "docker compose up -d --wait")
    assert exit_code([ok, advice]) == 0
    assert exit_code([ok, advice, broken]) == 1


def test_a_check_that_crashes_is_reported_not_raised():
    def boom() -> Check:
        raise RuntimeError("no network")

    [result] = run_checks([boom])
    assert not result.ok and "no network" in result.detail


def test_the_key_check_names_the_fix(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    check = check_api_key()
    assert not check.ok and "ANTHROPIC_API_KEY" in check.fix


def test_no_key_is_needed_for_a_local_model(monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_provider", "ollama")
    assert check_api_key().ok


def test_this_interpreter_is_supported():
    assert check_python().ok
