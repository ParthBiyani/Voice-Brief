from __future__ import annotations

import pytest

from voicebrief.llm.pricing import (
    CACHE_READ_MULTIPLIER,
    USD_TO_INR,
    cost_inr,
    price_for,
)


class TestPriceLookup:
    def test_known_models_are_priced(self):
        assert price_for("claude-opus-5").input_per_mtok == 5.00
        assert price_for("claude-haiku-4-5").output_per_mtok == 5.00

    @pytest.mark.parametrize(
        "model", ["qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b", "mistral-small"]
    )
    def test_local_models_are_free(self, model):
        assert price_for(model).input_per_mtok == 0.0

    def test_unknown_hosted_model_raises_rather_than_costing_zero(self):
        """Silent zero-cost would make the budget kill-switch unable to trip."""
        with pytest.raises(KeyError, match="No price recorded"):
            price_for("some-new-hosted-model")


class TestCostCalculation:
    def test_basic_cost(self):
        # 1M input + 1M output on Opus 5 = (5 + 25) USD
        cost = cost_inr("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(30.0 * USD_TO_INR)

    def test_cache_reads_are_cheaper_than_fresh_input(self):
        fresh = cost_inr("claude-opus-5", input_tokens=100_000, output_tokens=0)
        cached = cost_inr(
            "claude-opus-5", input_tokens=0, output_tokens=0, cache_read_tokens=100_000
        )
        assert cached == pytest.approx(fresh * CACHE_READ_MULTIPLIER)

    def test_cache_writes_cost_more_than_fresh_input(self):
        fresh = cost_inr("claude-opus-5", input_tokens=100_000, output_tokens=0)
        written = cost_inr(
            "claude-opus-5", input_tokens=0, output_tokens=0, cache_write_tokens=100_000
        )
        assert written > fresh

    def test_local_models_cost_nothing(self):
        assert cost_inr("qwen2.5:7b", input_tokens=999_999, output_tokens=999_999) == 0.0

    def test_zero_tokens_is_zero(self):
        assert cost_inr("claude-opus-5", input_tokens=0, output_tokens=0) == 0.0

    def test_shipped_design_stays_under_the_prd_budget(self):
        """PRD target: under INR 8 per episode.

        Token shapes are taken from real measured calls. This asserts the design the
        pipeline actually implements: summarize only the clusters that survive
        pre-ranking, cache the shared system prompts, Sonnet for the script.
        """
        summaries = 15 * cost_inr(
            "claude-haiku-4-5", input_tokens=200, output_tokens=120, cache_read_tokens=700
        )
        ranking = cost_inr(
            "claude-haiku-4-5", input_tokens=2_000, output_tokens=1_500, cache_read_tokens=6_000
        )
        script = cost_inr("claude-sonnet-5", input_tokens=6_000, output_tokens=2_400)
        total = summaries + ranking + script
        assert total < 8.0, f"projected episode cost INR {total:.2f} exceeds the budget"

    def test_summarizing_every_cluster_is_the_waste_it_looks_like(self):
        """Guards the design decision: summarizing all 60 clusters instead of the ~15
        that can reach an episode costs more than the entire optimized episode."""
        all_sixty = 60 * cost_inr("claude-haiku-4-5", input_tokens=900, output_tokens=120)
        survivors = 15 * cost_inr(
            "claude-haiku-4-5", input_tokens=200, output_tokens=120, cache_read_tokens=700
        )
        assert all_sixty > survivors * 5

    def test_opus_script_alone_would_break_the_budget(self):
        """Why VB_LLM_SCRIPT_MODEL defaults to Sonnet. Documented so that flipping it
        to Opus is a conscious choice with a known price."""
        opus = cost_inr("claude-opus-5", input_tokens=6_000, output_tokens=2_400)
        sonnet = cost_inr("claude-sonnet-5", input_tokens=6_000, output_tokens=2_400)
        assert opus > 7.0
        assert sonnet < 3.5


def test_exchange_rate_is_explicit():
    """Guards against the rate silently drifting far from reality and making every
    reported cost wrong."""
    assert 70 < USD_TO_INR < 100
