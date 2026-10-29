"""Anthropic provider.

Written against the current Messages API, which differs from older patterns in ways
that matter here:

* Thinking is `{"type": "adaptive"}`. The `budget_tokens` form is rejected outright
  on the 5.x generation, and `{"type": "disabled"}` is rejected on Sonnet 5.5 and
  Opus 5.5, so thinking is never switched off; depth is controlled with effort.
* Depth is controlled by `output_config.effort`, not by a token budget.
* Assistant prefill is gone, so response shape is steered with structured outputs
  (`output_config.format`) rather than by seeding the reply.
* Model ids carry no date suffix.
"""

from __future__ import annotations

import json
import time

from voicebrief.config import get_settings
from voicebrief.llm.base import (
    Completion,
    LLMError,
    LLMProvider,
    Stage,
    Tier,
    Usage,
)
from voicebrief.llm.pricing import ensure_priced
from voicebrief.logging import get_logger

log = get_logger(__name__)

# Effort per tier. Script generation is the one place quality is worth paying for;
# bulk summarization and ranking are deliberately cheap. Always sent explicitly: the
# default differs between models (`medium` on Opus 5.5 and Haiku 5.5, `high` on
# Sonnet), and a pipeline whose quality depends on which default a model ships with
# is a pipeline that changes when the config does.
EFFORT_BY_TIER = {Tier.flagship: "high", Tier.utility: "low"}

# Adaptive thinking spends output tokens *before* the answer and they count against
# `max_tokens`. A 25-word cold open given a 200-token cap used to come back empty or
# cut off because thinking ate the budget, so callers state how much *text* they want
# and the provider adds room for the reasoning on top.
THINKING_HEADROOM = {"low": 1024, "medium": 3000, "high": 6000}
# On truncation the call is repeated once with a larger budget, up to this ceiling.
MAX_TOKENS_CEILING = 32_000

# Not every Claude model takes the same request shape, and sending the wrong one is a
# hard 400 rather than a silent downgrade. Capabilities are declared per model prefix
# and the request is built to fit. `claude-haiku-4-5` rejects adaptive thinking and
# `output_config.effort`; they arrived with the 4.6+ generation.
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-5",  # also matches claude-opus-5-5
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",  # also matches claude-sonnet-5-5
    "claude-sonnet-4-6",
    "claude-haiku-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def supports_adaptive_thinking(model: str) -> bool:
    return model.startswith(_ADAPTIVE_THINKING_MODELS)


def supports_effort(model: str) -> bool:
    # Same generation boundary as adaptive thinking.
    return supports_adaptive_thinking(model)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        flagship_model: str | None = None,
        utility_model: str | None = None,
    ) -> None:
        import anthropic

        settings = get_settings()
        key = api_key or (
            settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        )
        if not key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Set it in .env, or switch "
                "VB_LLM_PROVIDER to 'ollama' for a local model."
            )

        self._anthropic = anthropic
        # The SDK already retries 429s, 5xx and connection errors with backoff; no
        # second retry layer sits on top of that for transport failures.
        self.client = anthropic.Anthropic(api_key=key, max_retries=3, timeout=120.0)
        self._models = {
            Tier.flagship: flagship_model or settings.llm_script_model,
            Tier.utility: utility_model or settings.llm_utility_model,
        }
        # Refuse to start with an unpriced model. Discovering it after the call would
        # mean the money was spent and the ledger never saw it.
        for model in self._models.values():
            ensure_priced(model)

    def model_for(self, tier: Tier) -> str:
        return self._models[tier]

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
    ) -> Completion:
        model = self.model_for(tier)
        started = time.perf_counter()

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        output_config: dict = {}
        headroom = 0

        if supports_adaptive_thinking(model):
            kwargs["thinking"] = {"type": "adaptive"}
        if supports_effort(model):
            effort = EFFORT_BY_TIER[tier]
            output_config["effort"] = effort
            headroom = THINKING_HEADROOM[effort]

        if system:
            if cache_system:
                # Marks the shared instructions cacheable. This only takes effect once
                # the prefix clears the model's minimum cacheable length; shorter
                # prompts are silently billed at the normal rate, so the ledger (which
                # records cache reads) is the source of truth for whether it worked.
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system

        if schema:
            # Prefill is unavailable on this model generation, so structured output is
            # the supported way to guarantee parseable JSON.
            output_config["format"] = {"type": "json_schema", "schema": schema}

        if output_config:
            kwargs["output_config"] = output_config

        budget = max_tokens + headroom
        usage = Usage()
        response = None
        text = ""

        # At most two attempts, and only for truncation: the first with the requested
        # budget plus thinking headroom, the second with double. Both are paid for, so
        # usage is summed across them and the ledger sees the real spend.
        for attempt in (1, 2):
            response = self._create({**kwargs, "max_tokens": budget})
            usage = usage + self._usage(response)

            # A refusal is a successful HTTP response with no usable content. Callers
            # must not treat it as an empty completion and carry on.
            if response.stop_reason == "refusal":
                category = getattr(response.stop_details, "category", None)
                raise LLMError(f"model declined the request (category={category})")

            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            if response.stop_reason != "max_tokens":
                break
            if attempt == 1:
                budget = min(budget * 2, MAX_TOKENS_CEILING)
                log.warning(
                    "llm.truncated_retry",
                    stage=stage.value,
                    model=model,
                    text_chars=len(text),
                    next_budget=budget,
                )
                continue
            raise LLMError(
                f"response truncated at {budget} tokens for stage {stage.value} "
                f"(model={model}); refusing to use a cut-off result"
            )

        parsed = None
        if schema and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                # Malformed output is the one failure worth a fresh attempt; nothing
                # was cut off, the model simply produced bad JSON.
                raise LLMError(
                    f"structured output was not valid JSON: {exc}", retryable=True
                ) from exc
        if not text.strip():
            raise LLMError(f"model returned no text for stage {stage.value}", retryable=True)

        return Completion(
            text=text,
            usage=usage,
            model=model,
            provider=self.name,
            stage=stage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            stop_reason=response.stop_reason,
            request_id=getattr(response, "_request_id", None),
            parsed=parsed,
        )

    # ── internals ─────────────────────────────────────────────────────────────
    def _create(self, kwargs: dict):
        try:
            return self.client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited by Anthropic: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach Anthropic: {exc}") from exc

    @staticmethod
    def _usage(response) -> Usage:
        return Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )
