"""Topics read from a story's own words.

Sources carry blanket tags: every arXiv cs.AI paper arrives as `agentic-ai,
machine-learning, research`, whether it is about GUI agents or medical imaging. Ranked on
those tags alone, a whole day of papers ties, and nothing a person declared ("I follow
computer vision") can tell them apart. So the ranker also asks what the story itself is
about, in the same vocabulary people pick their topics from and sources are tagged with.

Matching is whole-word and deliberately conservative: a topic is claimed only on a term
that is specific to it ("quantization", "retrieval-augmented"), never on words every AI
story uses ("model", "AI", "learning").
"""

from __future__ import annotations

import re
from functools import lru_cache

# Product topic → terms that, on their own, say a story is about it. Lowercase; a term
# ending in "*" matches any word starting with it ("quantiz*" → quantized, quantization).
CONTENT_TERMS: dict[str, tuple[str, ...]] = {
    "agentic-ai": (
        "agent", "agents", "agentic", "multi-agent", "tool use", "tool-use", "tool calling",
        "function calling", "mcp", "model context protocol", "orchestration", "planner",
        "computer use", "browser use", "gui agent", "autonomous", "workflow automation",
        "langgraph", "langchain", "crewai", "autogen",
    ),
    "retrieval": (
        "retrieval", "retrieval-augmented", "rag", "vector database", "vector search",
        "vector store", "embedding*", "rerank*", "semantic search", "knowledge graph",
        "knowledge base", "long-term memory", "agent memory", "memory-augmented",
        "long-context", "context window",
        "qdrant", "pinecone", "weaviate", "pgvector", "faiss",
    ),
    "computer-vision": (
        "vision", "visual", "image*", "video*", "pixel*", "segmentation", "object detection", "diffusion", "text-to-image", "vlm", "vision-language",
        "multimodal", "3d", "point cloud", "depth estimation", "camera", "ocr",
        "stable diffusion", "gaussian splatting", "nerf",
    ),
    "edge-ai": (
        "on-device", "on device", "run locally", "runs locally", "local model*", "edge ai", "edge device*", "edge inference", "mobile", "quantiz*", "gguf", "llama.cpp", "ollama", "int4",
        "int8", "4-bit", "8-bit", "distill*", "small language model", "slm", "latency",
        "efficient inference", "inference efficiency", "onnx", "tflite", "coreml",
        "microcontroller", "raspberry pi", "npu", "lora", "pruning", "sparsity",
    ),
    "nlp": (
        "language model", "llm", "llms", "nlp", "natural language", "translation",
        "multilingual", "tokeniz*", "text generation", "summarization", "question answering",
        "speech", "asr", "text-to-speech", "tts", "dialogue", "chatbot",
    ),
    "systems": (
        "distributed", "kernel", "kernels", "gpu cluster", "scheduler", "scheduling",
        "throughput", "database", "compiler", "storage", "networking", "serving",
        "inference server", "cuda", "datacenter", "data center", "fault tolerance",
    ),
    "tooling": (
        "sdk", "cli", "open-source tool", "developer tool*", "ide", "plugin", "vs code",
        "toolkit", "devtools", "command-line",
    ),
    "releases": ("release", "released", "releases", "launch*", "v[0-9]*", "changelog"),
    "research": ("paper", "arxiv", "benchmark", "we propose", "we present", "we introduce"),
    "industry": (
        "funding", "raises", "acquisition", "acquires", "ipo", "valuation", "revenue",
        "layoffs", "lawsuit", "regulation", "startup", "partnership",
    ),
    "open-source": ("open-source", "open source", "open-weight*", "apache 2.0", "mit license"),
    "cloud": ("aws", "azure", "gcp", "google cloud", "serverless", "kubernetes", "cloud"),
}  # fmt: skip


def _term_regex(term: str) -> str:
    if term.endswith("*"):
        stem = re.escape(term[:-1]).replace(r"\[0\-9\]", "[0-9]")
        return rf"(?<![a-z0-9]){stem}[a-z0-9.\-]*"
    return rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"


@lru_cache(maxsize=1)
def _patterns() -> dict[str, re.Pattern[str]]:
    return {
        topic: re.compile("|".join(_term_regex(t) for t in terms))
        for topic, terms in CONTENT_TERMS.items()
    }


def content_topics(text: str) -> set[str]:
    """The product topics a story's own words are specifically about."""
    lowered = text.lower()
    return {topic for topic, pattern in _patterns().items() if pattern.search(lowered)}
