"""Anthropic provider.

Written against the current Messages API, which differs from older patterns in ways
that matter here:

* Thinking is `{"type": "adaptive"}`. The `budget_tokens` form is rejected outright
  on the Opus 5 / Sonnet 5 generation.
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
from voicebrief.logging import get_logger

log = get_logger(__name__)

# Effort per tier. Script generation is the one place quality is worth paying for;
# bulk summarization and ranking are deliberately cheap.
EFFORT_BY_TIER = {Tier.flagship: "high", Tier.utility: "low"}

# Not every Claude model takes the same request shape, and sending the wrong one is a
# hard 400 rather than a silent downgrade. Confirmed against the live API:
# `claude-haiku-4-5` rejects both adaptive thinking and `output_config.effort` —
# those arrived with the 4.6+ generation. Rather than pin the pipeline to one model
# family, capabilities are declared per model prefix and the request is built to fit.
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
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
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        if not key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Set it in .env, or switch "
                "VB_LLM_PROVIDER to 'ollama' for a local model."
            )

        self._anthropic = anthropic
        # The SDK already retries 429s and 5xx with backoff; a second retry layer on
        # top would multiply the wait rather than help.
        self.client = anthropic.Anthropic(api_key=key, max_retries=3, timeout=120.0)
        self._models = {
            Tier.flagship: flagship_model or settings.llm_script_model,
            Tier.utility: utility_model or settings.llm_utility_model,
        }

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
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        output_config: dict = {}

        if supports_adaptive_thinking(model):
            kwargs["thinking"] = {"type": "adaptive"}
        if supports_effort(model):
            output_config["effort"] = EFFORT_BY_TIER[tier]

        if system:
            if cache_system:
                # Long shared instructions reused across ~60 calls in a run. Caching
                # them turns the dominant input cost of bulk stages into ~10%.
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

        try:
            response = self.client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited by Anthropic: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach Anthropic: {exc}") from exc

        # A refusal is a successful HTTP response with no usable content. Callers must
        # not treat it as an empty completion and carry on.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMError(f"model declined the request (category={category})")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )

        parsed = None
        if schema and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(f"structured output was not valid JSON: {exc}") from exc

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
