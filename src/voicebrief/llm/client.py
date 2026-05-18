"""The one LLM entry point.

Wraps a provider with the three things every call needs and none of them should have
to remember:

* **Cost accounting.** Every completion writes a `cost_entry` row. This is what turns
  the PRD's "< ₹8 per episode" from an estimate into a measurement.
* **A budget kill-switch.** Spend is checked against a daily ceiling *before* each
  call. A runaway loop costs one call's worth of money, not a night's worth.
* **Retries with degradation.** Transient failures retry; a stage that still fails can
  fall back to a cheaper tier rather than failing the whole episode.

Nothing in the pipeline constructs a provider directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from voicebrief.config import get_settings
from voicebrief.db.models import CostEntry
from voicebrief.llm.base import (
    BudgetExceeded,
    Completion,
    LLMError,
    LLMProvider,
    Stage,
    Tier,
)
from voicebrief.llm.pricing import cost_inr
from voicebrief.logging import get_logger

log = get_logger(__name__)


def build_provider(name: str | None = None) -> LLMProvider:
    """Construct the configured provider.

    Falls back to `echo` rather than raising when a hosted provider is selected but
    unusable, so that a missing key degrades the *quality* of a run instead of
    preventing the system from starting at all.
    """
    settings = get_settings()
    name = name or settings.llm_provider

    if name == "anthropic":
        from voicebrief.llm.anthropic_provider import AnthropicProvider

        try:
            return AnthropicProvider()
        except LLMError as exc:
            log.warning("llm.anthropic_unavailable", error=str(exc), fallback="echo")
            from voicebrief.llm.echo_provider import EchoProvider

            return EchoProvider()

    if name == "ollama":
        from voicebrief.llm.ollama_provider import OllamaProvider

        return OllamaProvider()

    from voicebrief.llm.echo_provider import EchoProvider

    return EchoProvider()


class LLMClient:
    """Provider + ledger + budget."""

    def __init__(
        self,
        session: Session,
        provider: LLMProvider | None = None,
        *,
        episode_id: uuid.UUID | None = None,
        daily_budget_inr: float | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session
        self.provider = provider or build_provider()
        self.episode_id = episode_id
        self.daily_budget_inr = (
            daily_budget_inr if daily_budget_inr is not None else settings.daily_budget_inr
        )
        self._run_cost_inr = 0.0

    # ── budget ────────────────────────────────────────────────────────────────
    def spend_today(self) -> float:
        since = datetime.now(timezone.utc) - timedelta(days=1)
        total = self.session.execute(
            select(func.coalesce(func.sum(CostEntry.cost_inr), 0)).where(
                CostEntry.created_at >= since
            )
        ).scalar_one()
        return float(total or 0.0)

    def _check_budget(self) -> None:
        spent = self.spend_today()
        if spent >= self.daily_budget_inr:
            raise BudgetExceeded(
                f"daily budget exhausted: spent INR {spent:.2f} of "
                f"{self.daily_budget_inr:.2f}. Raise VB_DAILY_BUDGET_INR or wait."
            )

    @property
    def run_cost_inr(self) -> float:
        """Spend attributed to this client instance — i.e. this episode."""
        return self._run_cost_inr

    # ── calls ─────────────────────────────────────────────────────────────────
    def complete(
        self,
        *,
        prompt: str,
        stage: Stage,
        tier: Tier = Tier.utility,
        system: str | None = None,
        max_tokens: int = 4096,
        schema: dict | None = None,
        cache_system: bool = False,
        fallback_tier: Tier | None = None,
    ) -> Completion:
        self._check_budget()

        try:
            completion = self._complete_with_retry(
                prompt=prompt,
                stage=stage,
                tier=tier,
                system=system,
                max_tokens=max_tokens,
                schema=schema,
                cache_system=cache_system,
            )
        except LLMError:
            if fallback_tier is None or fallback_tier is tier:
                raise
            # A stage that can degrade should degrade rather than fail the episode.
            log.warning("llm.tier_fallback", stage=stage.value, frm=tier.value,
                        to=fallback_tier.value)
            completion = self._complete_with_retry(
                prompt=prompt,
                stage=stage,
                tier=fallback_tier,
                system=system,
                max_tokens=max_tokens,
                schema=schema,
                cache_system=cache_system,
            )

        self._record(completion)
        return completion

    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _complete_with_retry(self, **kwargs) -> Completion:
        return self.provider.complete(**kwargs)

    # ── ledger ────────────────────────────────────────────────────────────────
    def _record(self, completion: Completion) -> None:
        usage = completion.usage
        cost = cost_inr(
            completion.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        self._run_cost_inr += cost

        self.session.add(
            CostEntry(
                episode_id=self.episode_id,
                stage=completion.stage.value,
                provider=completion.provider,
                model=completion.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cache_read_tokens,
                cost_inr=cost,
                latency_ms=completion.latency_ms,
                trace_id=completion.request_id,
            )
        )
        log.info(
            "llm.call",
            stage=completion.stage.value,
            model=completion.model,
            tokens=usage.total,
            cost_inr=round(cost, 4),
            ms=completion.latency_ms,
        )


def episode_cost_inr(session: Session, episode_id: uuid.UUID) -> float:
    total = session.execute(
        select(func.coalesce(func.sum(CostEntry.cost_inr), 0)).where(
            CostEntry.episode_id == episode_id
        )
    ).scalar_one()
    return float(total or 0.0)
