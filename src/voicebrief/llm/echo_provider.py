"""Deterministic stub provider.

Not a mock in the testing sense — it is a real provider that happens to be
computable. Every graph in the system runs end to end against it with no key, no
network and no spend, which is what makes the pipeline testable in CI.

Its output is derived from the prompt, so tests can assert on structure (a script has
segments, a ranking returns scores for every input) without asserting on the wording
of a model's answer.
"""

from __future__ import annotations

import hashlib
import json
import time

from voicebrief.llm.base import Completion, LLMProvider, Stage, Tier, Usage


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def model_for(self, tier: Tier) -> str:
        return f"echo-{tier.value}"

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
        started = time.perf_counter()
        self.calls.append(
            {"stage": stage.value, "tier": tier.value, "prompt": prompt, "schema": schema}
        )

        parsed = None
        if schema:
            parsed = _synthesize(schema, prompt)
            text = json.dumps(parsed)
        else:
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            text = f"[echo:{stage.value}:{digest}] " + " ".join(prompt.split()[:40])

        # Token counts approximate the real thing closely enough that budget logic and
        # ledger arithmetic are exercised rather than bypassed.
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=len(prompt.split()) + len((system or "").split()),
                output_tokens=len(text.split()),
            ),
            model=self.model_for(tier),
            provider=self.name,
            stage=stage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            stop_reason="end_turn",
            parsed=parsed,
        )


def _synthesize(schema: dict, prompt: str) -> dict | list:
    """Build a minimal instance satisfying a JSON schema.

    Deterministic in the prompt so repeated runs produce identical output — the eval
    harness diffs runs, and a stub that varied would generate phantom regressions.
    """
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:6], 16)

    def build(node: dict, depth: int = 0):
        kind = node.get("type", "string")
        if kind == "object":
            required = node.get("required") or list(node.get("properties", {}))
            return {
                key: build(node.get("properties", {}).get(key, {}), depth + 1)
                for key in required
            }
        if kind == "array":
            item = node.get("items", {"type": "string"})
            return [build(item, depth + 1) for _ in range(max(1, node.get("minItems", 1)))]
        if kind == "integer":
            low = node.get("minimum", 0)
            high = node.get("maximum", 10)
            return low + (seed + depth) % max(1, high - low + 1)
        if kind == "number":
            return round((seed % 100) / 100.0, 2)
        if kind == "boolean":
            return bool((seed + depth) % 2)
        if enum := node.get("enum"):
            return enum[(seed + depth) % len(enum)]
        return f"echo-{seed:x}"

    return build(schema)
