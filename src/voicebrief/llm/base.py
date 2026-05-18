"""LLM provider contract.

Every model call in VoiceBrief goes through one interface. That is not architecture
astronautics — it is what makes three things possible at once:

* **A real cost number.** The PRD promises "< ₹8 per episode" as a *measured* figure.
  One wrapper means one place that records tokens, so the ledger cannot drift from
  reality by someone adding a call that forgets to account for itself.
* **BYOK without code changes.** Anthropic when a key is present, a local Ollama model
  when it isn't. Same call sites either way.
* **A testable pipeline.** The `echo` provider makes every graph runnable in CI with
  no key, no network and no spend.

Providers are responsible for transport and token accounting only. Prompt
construction, retries at the semantic level, and budget enforcement live above them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    """What a call is for.

    Carried into the cost ledger so per-episode spend can be attributed to the stage
    that caused it — which is the only way to know what to optimise.
    """

    cluster_summary = "cluster_summary"
    ranking = "ranking"
    script = "script"
    translation = "translation"
    claim_extraction = "claim_extraction"
    contrast = "contrast"
    segment_qa = "segment_qa"
    outline = "outline"


class Tier(str, Enum):
    """Which model class a stage needs.

    Two tiers, not a model name at the call site. Stages declare the *capability* they
    need; the provider maps that to a concrete model. Swapping models is then config,
    not a search-and-replace across the pipeline.
    """

    # Long-form register, careful grounding. Script generation and contrast graphs.
    flagship = "flagship"
    # Bulk work at volume: summarizing 60 clusters, scoring 60 items.
    utility = "utility"


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class Completion:
    text: str
    usage: Usage
    model: str
    provider: str
    stage: Stage
    latency_ms: int = 0
    stop_reason: str | None = None
    request_id: str | None = None
    # Populated when the caller asked for structured output.
    parsed: dict | list | None = None
    meta: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    """Provider failure that the caller may retry or degrade around."""


class BudgetExceeded(LLMError):
    """The daily spend ceiling was hit. Never retried — that is the point."""


class LLMProvider(ABC):
    """One model backend."""

    name: str

    @abstractmethod
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
        """Run one completion.

        `schema`, when given, requests structured JSON output conforming to it.
        `cache_system` marks the system prompt as cacheable — worth it whenever the
        same instructions are reused across many calls in a run, which is the case
        for cluster summarization and ranking.
        """

    @abstractmethod
    def model_for(self, tier: Tier) -> str:
        """Concrete model id backing a tier."""

    def close(self) -> None:  # pragma: no cover - most providers need nothing
        return None
