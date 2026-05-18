"""Ollama provider — local models, zero marginal cost.

Exists for three reasons, in order of importance:

1. **The project runs without a key.** Anyone cloning this repo can generate a real
   episode with no account and no spend.
2. **It is the cost ablation.** The PRD asks what an episode costs; "₹6.40 hosted vs
   ₹0 local, with this quality difference" is a far better answer than either number
   alone, and the cost ledger records both identically.
3. **Privacy.** Mode 2 ingests the user's own documents. Some of those should never
   leave the machine, and a local provider is the only honest answer to that.

Quality is genuinely lower than the flagship for long-form script register. That is a
documented trade-off, not something to paper over.
"""

from __future__ import annotations

import json
import time

import httpx

from voicebrief.config import get_settings
from voicebrief.llm.base import Completion, LLMError, LLMProvider, Stage, Tier, Usage
from voicebrief.logging import get_logger

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        flagship_model: str | None = None,
        utility_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        model = flagship_model or settings.ollama_script_model
        self._models = {
            Tier.flagship: model,
            # One model serves both tiers by default. Running a second smaller model
            # would mean two sets of weights resident on a 6GB card, which is worse
            # than reusing one.
            Tier.utility: utility_model or model,
        }
        self._client = httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0))

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

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                # Deterministic by default: the eval harness compares runs, and a
                # sampling temperature would make every diff look like a regression.
                "temperature": 0.3 if tier is Tier.flagship else 0.0,
            },
        }
        if system:
            payload["system"] = system
        if schema:
            # Ollama constrains generation to a JSON schema natively.
            payload["format"] = schema

        try:
            response = self._client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Ollama is not reachable at {self.base_url}. Start it with "
                f"`ollama serve` and pull the model with `ollama pull {model}`."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        body = response.json()
        text = (body.get("response") or "").strip()

        parsed = None
        if schema and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Ollama returned invalid JSON for schema: {exc}") from exc

        return Completion(
            text=text,
            usage=Usage(
                input_tokens=int(body.get("prompt_eval_count") or 0),
                output_tokens=int(body.get("eval_count") or 0),
            ),
            model=model,
            provider=self.name,
            stage=stage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            stop_reason=body.get("done_reason"),
            parsed=parsed,
        )

    def close(self) -> None:
        self._client.close()
