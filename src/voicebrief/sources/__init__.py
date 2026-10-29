"""Source adapters.

Importing this package registers every adapter, which is what lets the orchestrator
resolve a `source.config.adapter` string to a class at runtime.
"""

# Registration side effects — imported for `registry.build()` to resolve kinds.
from voicebrief.sources import (  # noqa: F401
    arxiv,
    github,
    hackernews,
    huggingface,
    rss,
)
from voicebrief.sources.base import RawItem, SourceAdapter, SourceConfig, registry

__all__ = ["RawItem", "SourceAdapter", "SourceConfig", "registry"]
