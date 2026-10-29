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
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from voicebrief.config import get_settings
from voicebrief.db import session_scope
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

    A hosted provider that cannot be built is an error, not a reason to substitute the
    `echo` stub: that used to ship `[echo:script:...]` text through text-to-speech and
    mark the episode ready. `echo` is available only when asked for by name.
    """
    settings = get_settings()
    name = name or settings.llm_provider

    if name == "anthropic":
        from voicebrief.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()

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
        durable_ledger: bool = True,
        run_id: uuid.UUID | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session
        self.provider = provider or build_provider()
        self.episode_id = episode_id
        self.daily_budget_inr = (
            daily_budget_inr if daily_budget_inr is not None else settings.daily_budget_inr
        )
        self._run_cost_inr = 0.0
        # Ledger rows are committed on their own connection the moment a call
        # finishes. Written through the caller's session instead, they vanished with
        # it whenever generation failed and rolled back, so the money for a crashed
        # run was spent but never counted against the daily budget.
        self.durable_ledger = durable_ledger
        self.run_id = run_id

    # ── budget ────────────────────────────────────────────────────────────────
    def spend_today(self) -> float:
        since = datetime.now(UTC) - timedelta(days=1)
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
            log.warning(
                "llm.tier_fallback", stage=stage.value, frm=tier.value, to=fallback_tier.value
            )
            completion = self._complete_with_retry(
                prompt=prompt,
                stage=stage,
                tier=fallback_tier,
                system=system,
                max_tokens=max_tokens,
                schema=schema,
                cache_system=cache_system,
            )

        self._record(completion, prompt=prompt, system=system)
        return completion

    @retry(
        # One more attempt, and only where a repeat can plausibly succeed. The SDK has
        # already retried transport failures; refusals, truncation and 4xx responses
        # are deterministic and would just be paid for again.
        retry=retry_if_exception(lambda exc: getattr(exc, "retryable", False)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _complete_with_retry(self, **kwargs) -> Completion:
        return self.provider.complete(**kwargs)

    # ── ledger ────────────────────────────────────────────────────────────────
    def _write_ledger(self, entry: CostEntry) -> None:
        if self.durable_ledger:
            try:
                with session_scope() as ledger_session:
                    ledger_session.add(entry)
                return
            except Exception as exc:  # the paid result must survive a ledger hiccup
                log.error("llm.ledger_write_failed", error=str(exc), fallback="caller session")
                entry = CostEntry(
                    **{c.name: getattr(entry, c.name) for c in CostEntry.__table__.columns}
                )
        self.session.add(entry)

    def _record(
        self, completion: Completion, *, prompt: str | None = None, system: str | None = None
    ) -> None:
        usage = completion.usage
        cost = cost_inr(
            completion.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        self._run_cost_inr += cost

        entry = CostEntry(
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
            prompt=prompt,
            system_prompt=system,
            response=completion.text,
            stop_reason=completion.stop_reason,
            run_id=self.run_id,
        )
        self._write_ledger(entry)
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
