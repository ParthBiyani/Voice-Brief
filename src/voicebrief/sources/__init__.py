"""Source adapters.

Importing this package registers every adapter, which is what lets the orchestrator
resolve a `source.config.adapter` string to a class at runtime.
"""

from voicebrief.sources.base import RawItem, SourceAdapter, SourceConfig, registry

# Registration side effects — imported for `registry.build()` to resolve kinds.
from voicebrief.sources import arxiv, github, hackernews, huggingface  # noqa: E402,F401

__all__ = ["RawItem", "SourceAdapter", "SourceConfig", "registry"]
