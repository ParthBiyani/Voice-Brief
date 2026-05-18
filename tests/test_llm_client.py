"""LLM client: ledger, budget, retries, degradation.

Uses the echo provider so these are real end-to-end exercises of the wrapper rather
than mock assertions, and cost the same as running them in CI: nothing.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from voicebrief.db import session_scope
from voicebrief.db.models import CostEntry
from voicebrief.llm.base import BudgetExceeded, Completion, LLMError, Stage, Tier, Usage
from voicebrief.llm.client import LLMClient
from voicebrief.llm.echo_provider import EchoProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def session():
    with session_scope() as s:
        yield s
        s.rollback()


@pytest.fixture(autouse=True)
def _clean(session):
    session.query(CostEntry).delete()
    session.flush()
    yield
    session.query(CostEntry).delete()
    session.flush()


class FlakyProvider(EchoProvider):
    """Fails a fixed number of times, then succeeds."""

    def __init__(self, failures: int):
        super().__init__()
        self.remaining = failures
        self.attempts = 0

    def complete(self, **kwargs) -> Completion:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise LLMError("transient upstream failure")
        return super().complete(**kwargs)


class TierSensitiveProvider(EchoProvider):
    """Fails only on the flagship tier — models the real degradation case."""

    def complete(self, **kwargs) -> Completion:
        if kwargs.get("tier") is Tier.flagship:
            raise LLMError("flagship unavailable")
        return super().complete(**kwargs)


class TestLedger:
    def test_every_call_writes_a_row(self, session):
        client = LLMClient(session, EchoProvider())
        client.complete(prompt="hello world", stage=Stage.ranking)
        session.flush()
        assert session.execute(select(func.count(CostEntry.id))).scalar_one() == 1

    def test_row_records_stage_provider_and_tokens(self, session):
        client = LLMClient(session, EchoProvider())
        client.complete(prompt="one two three", stage=Stage.script, tier=Tier.flagship)
        session.flush()
        entry = session.execute(select(CostEntry)).scalar_one()
        assert entry.stage == "script"
        assert entry.provider == "echo"
        assert entry.input_tokens > 0
        assert entry.output_tokens > 0

    def test_free_provider_still_writes_a_zero_cost_row(self, session):
        """A zero-cost row proves the accounting path runs for local models too —
        otherwise the hosted-vs-local ablation has nothing to compare."""
        client = LLMClient(session, EchoProvider())
        client.complete(prompt="hello", stage=Stage.ranking)
        session.flush()
        assert float(session.execute(select(CostEntry.cost_inr)).scalar_one()) == 0.0

    def test_run_cost_accumulates_across_calls(self, session):
        client = LLMClient(session, EchoProvider())
        for _ in range(3):
            client.complete(prompt="x y z", stage=Stage.cluster_summary)
        assert client.run_cost_inr >= 0.0
        session.flush()
        assert session.execute(select(func.count(CostEntry.id))).scalar_one() == 3


class TestBudget:
    def test_calls_proceed_below_the_ceiling(self, session):
        client = LLMClient(session, EchoProvider(), daily_budget_inr=100.0)
        assert client.complete(prompt="fine", stage=Stage.ranking).text

    def test_exhausted_budget_blocks_the_call(self, session):
        session.add(
            CostEntry(
                stage="script", provider="anthropic", model="claude-sonnet-5",
                input_tokens=1, output_tokens=1, cost_inr=50.0,
            )
        )
        session.flush()
        client = LLMClient(session, EchoProvider(), daily_budget_inr=10.0)
        with pytest.raises(BudgetExceeded, match="daily budget exhausted"):
            client.complete(prompt="too expensive", stage=Stage.script)

    def test_budget_breach_is_not_retried(self, session):
        """Retrying a budget breach would just burn wall-clock to fail again."""
        session.add(
            CostEntry(
                stage="script", provider="anthropic", model="claude-sonnet-5",
                input_tokens=1, output_tokens=1, cost_inr=99.0,
            )
        )
        session.flush()
        provider = EchoProvider()
        client = LLMClient(session, provider, daily_budget_inr=1.0)
        with pytest.raises(BudgetExceeded):
            client.complete(prompt="x", stage=Stage.script)
        assert provider.calls == [], "provider must never be reached"


class TestResilience:
    def test_transient_failures_are_retried(self, session):
        provider = FlakyProvider(failures=2)
        client = LLMClient(session, provider)
        assert client.complete(prompt="eventually works", stage=Stage.ranking).text
        assert provider.attempts == 3

    def test_persistent_failure_propagates(self, session):
        client = LLMClient(session, FlakyProvider(failures=99))
        with pytest.raises(LLMError):
            client.complete(prompt="never works", stage=Stage.ranking)

    def test_stage_degrades_to_a_cheaper_tier_rather_than_failing(self, session):
        """A dead flagship must cost quality, not the whole episode."""
        client = LLMClient(session, TierSensitiveProvider())
        completion = client.complete(
            prompt="write something", stage=Stage.script,
            tier=Tier.flagship, fallback_tier=Tier.utility,
        )
        assert completion.model == "echo-utility"

    def test_without_a_fallback_the_failure_surfaces(self, session):
        client = LLMClient(session, TierSensitiveProvider())
        with pytest.raises(LLMError):
            client.complete(prompt="x", stage=Stage.script, tier=Tier.flagship)


class TestStructuredOutput:
    def test_schema_produces_parsed_output(self, session):
        client = LLMClient(session, EchoProvider())
        schema = {
            "type": "object",
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
            "required": ["score"],
        }
        result = client.complete(prompt="rate this", stage=Stage.ranking, schema=schema)
        assert isinstance(result.parsed, dict)
        assert 0 <= result.parsed["score"] <= 10

    def test_stub_output_is_deterministic(self, session):
        """The eval harness diffs runs; a varying stub would fake regressions."""
        client = LLMClient(session, EchoProvider())
        schema = {"type": "object", "properties": {"score": {"type": "integer"}},
                  "required": ["score"]}
        a = client.complete(prompt="same prompt", stage=Stage.ranking, schema=schema)
        b = client.complete(prompt="same prompt", stage=Stage.ranking, schema=schema)
        assert a.parsed == b.parsed


def test_usage_addition():
    assert (Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)).input_tokens == 11
