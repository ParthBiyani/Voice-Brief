"""Model pricing, in USD per million tokens.

The PRD's cost target is denominated in rupees, so the conversion lives here too.
Prices are a moving target — this table is the one place to update, and every figure
the system reports is derived from it rather than hardcoded anywhere else.

Cache economics matter more than they look: a cache read is ~10% of the input rate
and a cache write ~125%. Cluster summarization reuses one long system prompt across
many calls, so caching turns the dominant cost of that stage into a rounding error.

Measured cost of one 2,400-word episode, which drove three design decisions:

    naive (summarize all 60 clusters, Opus script)      INR 20.08
    + summarize only the ~15 that survive pre-ranking   INR 13.66
    + cache the shared system prompts                   INR  9.45
    + Sonnet rather than Opus for the script            INR  4.94

The first two are pure waste removal and are now how the pipeline works. The third is
a real quality trade-off, defaulted to meet the PRD's INR 8 ceiling and reversible in
one config line. Opus alone accounts for INR 7.51 of the naive figure — script output
tokens are the single largest line item in the whole system.
"""

from __future__ import annotations

from dataclasses import dataclass

# Updated 2026-06-08. A stale rate makes the cost ledger quietly wrong, so this
# constant is asserted against in tests to force a conscious update.
USD_TO_INR = 83.5

CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


# Anthropic first-party API rates.
PRICES: dict[str, Price] = {
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-fable-5-1": Price(10.00, 50.00),
}

# Local models cost nothing per token. They still get ledger rows: a stage that
# produces zero-cost entries proves the accounting path works, and it makes the
# hosted-vs-local ablation a single query rather than a guess.
LOCAL_PRICE = Price(0.0, 0.0)


# Models that genuinely cost nothing per token. Listed explicitly rather than by a
# loose fallback, so that a typo'd hosted model name still raises instead of silently
# billing zero and disarming the budget kill-switch.
_FREE_PREFIXES = ("qwen", "llama", "mistral", "gemma", "phi", "deepseek", "echo-")


def price_for(model: str) -> Price:
    if model in PRICES:
        return PRICES[model]
    # Ollama tags look like "qwen2.5:7b-instruct-q4_K_M".
    if ":" in model or model.startswith(_FREE_PREFIXES):
        return LOCAL_PRICE
    # An unknown hosted model must not silently cost zero — that would make the
    # budget kill-switch unable to trip.
    raise KeyError(
        f"No price recorded for model {model!r}. Add it to voicebrief.llm.pricing "
        f"before using it, or the cost ledger will under-report."
    )


def cost_inr(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost of one call in INR."""
    price = price_for(model)
    usd = (
        input_tokens * price.input_per_mtok
        + cache_read_tokens * price.input_per_mtok * CACHE_READ_MULTIPLIER
        + cache_write_tokens * price.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + output_tokens * price.output_per_mtok
    ) / 1_000_000
    return usd * USD_TO_INR
