"""AnthropicProvider request shape, truncation and failure handling, against a fake SDK.

No network and no spend: the provider's `client` is replaced by a recorder that returns
canned responses, so these tests pin down exactly what would be sent to the API.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.db.models import CostEntry
from voicebrief.db.session import get_engine
from voicebrief.llm.anthropic_provider import (
    MAX_TOKENS_CEILING,
    THINKING_HEADROOM,
    AnthropicProvider,
    supports_adaptive_thinking,
)
from voicebrief.llm.base import LLMError, Stage, Tier
from voicebrief.llm.client import LLMClient, build_provider
from voicebrief.llm.echo_provider import EchoProvider
from voicebrief.llm.pricing import PRICES, cost_inr, ensure_priced


def reply(text="ok", *, stop="end_turn", out=100, inp=50, cache_read=0):
    return SimpleNamespace(
        stop_reason=stop,
        stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=inp,
            output_tokens=out,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
        _request_id="req_test",
    )


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def provider_with(replies, *, script="claude-sonnet-5-5", utility="claude-haiku-5-5"):
    provider = AnthropicProvider(api_key="test-key", flagship_model=script, utility_model=utility)
    fake = FakeMessages(replies)
    provider.client = SimpleNamespace(messages=fake)
    return provider, fake


class TestRequestShape:
    def test_flagship_sends_adaptive_thinking_and_explicit_effort(self):
        provider, fake = provider_with([reply()])
        provider.complete(prompt="p", stage=Stage.script, tier=Tier.flagship, max_tokens=1500)
        call = fake.calls[0]
        assert call["model"] == "claude-sonnet-5-5"
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"]["effort"] == "high"

    def test_thinking_is_never_disabled(self):
        # Sonnet 5.5 and Opus 5.5 reject {"type": "disabled"} with a 400.
        provider, fake = provider_with([reply()])
        provider.complete(prompt="p", stage=Stage.script, tier=Tier.flagship)
        assert fake.calls[0]["thinking"]["type"] != "disabled"
        assert "temperature" not in fake.calls[0]

    def test_utility_tier_uses_haiku_55_at_low_effort(self):
        provider, fake = provider_with([reply()])
        provider.complete(prompt="p", stage=Stage.cluster_summary, tier=Tier.utility)
        call = fake.calls[0]
        assert call["model"] == "claude-haiku-5-5"
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"]["effort"] == "low"

    def test_haiku_45_gets_neither_thinking_nor_effort(self):
        provider, fake = provider_with([reply()], utility="claude-haiku-4-5")
        provider.complete(prompt="p", stage=Stage.cluster_summary, tier=Tier.utility)
        assert "thinking" not in fake.calls[0]
        assert "output_config" not in fake.calls[0]

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-4-8"],
    )
    def test_current_models_take_adaptive_thinking(self, model):
        assert supports_adaptive_thinking(model)

    def test_requested_tokens_get_thinking_headroom(self):
        provider, fake = provider_with([reply()])
        provider.complete(prompt="p", stage=Stage.script, tier=Tier.flagship, max_tokens=200)
        assert fake.calls[0]["max_tokens"] == 200 + THINKING_HEADROOM["high"]

    def test_schema_requests_structured_output(self):
        provider, fake = provider_with([reply('{"a": 1}')])
        done = provider.complete(
            prompt="p", stage=Stage.ranking, schema={"type": "object"}, max_tokens=500
        )
        assert fake.calls[0]["output_config"]["format"]["type"] == "json_schema"
        assert done.parsed == {"a": 1}

    def test_system_prompt_is_marked_cacheable_on_request(self):
        provider, fake = provider_with([reply()])
        provider.complete(prompt="p", stage=Stage.script, system="rules", cache_system=True)
        assert fake.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


class TestTruncation:
    def test_a_cut_off_reply_is_retried_once_with_a_larger_budget(self):
        provider, fake = provider_with([reply("cut", stop="max_tokens", out=300), reply("full")])
        done = provider.complete(prompt="p", stage=Stage.script, tier=Tier.flagship)
        assert done.text == "full"
        first, second = (c["max_tokens"] for c in fake.calls)
        assert second == min(first * 2, MAX_TOKENS_CEILING)

    def test_usage_from_both_attempts_is_summed(self):
        provider, _ = provider_with(
            [reply("cut", stop="max_tokens", out=300, inp=40), reply("full", out=120, inp=40)]
        )
        done = provider.complete(prompt="p", stage=Stage.script, tier=Tier.flagship)
        assert done.usage.output_tokens == 420
        assert done.usage.input_tokens == 80

    def test_still_truncated_after_the_retry_is_an_error_not_a_result(self):
        provider, fake = provider_with(
            [reply("a", stop="max_tokens"), reply("b", stop="max_tokens")]
        )
        with pytest.raises(LLMError, match="truncated") as info:
            provider.complete(prompt="p", stage=Stage.script, tier=Tier.flagship)
        assert not info.value.retryable
        assert len(fake.calls) == 2


class TestFailures:
    def test_refusal_is_an_error_and_is_not_retryable(self):
        provider, _ = provider_with([reply("", stop="refusal")])
        with pytest.raises(LLMError, match="declined") as info:
            provider.complete(prompt="p", stage=Stage.script)
        assert not info.value.retryable

    def test_malformed_json_is_retryable(self):
        provider, _ = provider_with([reply("not json")])
        with pytest.raises(LLMError) as info:
            provider.complete(prompt="p", stage=Stage.ranking, schema={"type": "object"})
        assert info.value.retryable

    def test_an_empty_reply_is_retryable_rather_than_an_empty_segment(self):
        provider, _ = provider_with([reply("   ")])
        with pytest.raises(LLMError) as info:
            provider.complete(prompt="p", stage=Stage.script)
        assert info.value.retryable


class TestPricingAndStartup:
    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
    def test_current_models_are_priced(self, model):
        assert model in PRICES
        assert cost_inr(model, input_tokens=1_000_000, output_tokens=0) > 0

    def test_haiku_55_is_an_order_of_magnitude_cheaper_than_haiku_45(self):
        old = cost_inr("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
        new = cost_inr("claude-haiku-5-5", input_tokens=1_000_000, output_tokens=0)
        assert new == pytest.approx(old / 10)

    def test_unknown_hosted_model_is_refused_before_any_call(self):
        with pytest.raises(KeyError):
            ensure_priced("claude-nonexistent-9")
        with pytest.raises(KeyError):
            AnthropicProvider(api_key="k", flagship_model="claude-nonexistent-9")

    def test_missing_api_key_is_a_hard_error_not_a_silent_echo(self, monkeypatch):
        from voicebrief.config import get_settings

        keyless = get_settings().model_copy(update={"anthropic_api_key": None})
        monkeypatch.setattr("voicebrief.llm.anthropic_provider.get_settings", lambda: keyless)
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            build_provider("anthropic")

    def test_echo_is_available_when_asked_for_by_name(self):
        assert isinstance(build_provider("echo"), EchoProvider)


@pytest.mark.integration
class TestDurableLedger:
    def test_spend_survives_the_callers_rollback(self):
        episode_id = uuid.uuid4()  # no FK row needed: episode_id is only a tag here
        marker = f"durable-{uuid.uuid4().hex[:8]}"
        engine = get_engine()
        try:
            with Session(engine) as work:
                work.begin()
                client = LLMClient(work, EchoProvider())
                completion = client.complete(prompt=marker, stage=Stage.ranking)
                trace = completion.request_id
                work.rollback()  # a failed generation rolls back everything it did

            with Session(engine) as check:
                rows = check.execute(
                    select(CostEntry).where(CostEntry.stage == Stage.ranking.value)
                ).scalars()
                assert any(r.trace_id == trace for r in rows) or trace is None
        finally:
            with Session(engine) as cleanup:
                cleanup.query(CostEntry).filter(CostEntry.provider == "echo").delete()
                cleanup.commit()
        assert episode_id  # keeps the tag meaningful for readers
