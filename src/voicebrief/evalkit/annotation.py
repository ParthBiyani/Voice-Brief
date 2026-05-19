"""Annotation rubric for the evaluation sets.

Honesty note, because it matters for how the numbers in the README should be read:

The topic and relevance labels are **rubric-based annotations**, not model output.
The rubric below was authored by reading the actual crawled corpus and identifying
the subject areas present in it; it is then applied deterministically so that
re-running the harness reproduces the same labels exactly.

This is the standard trade-off in rubric annotation. What it buys:
  * reproducibility — the labels are a pure function of the corpus and this file
  * auditability — every decision is a rule you can read and disagree with
  * independence — the rules key off title/summary keywords and never off an
    embedding, a cosine score, or a clustering output, so the evaluation is not
    scoring the system against itself

What it costs: the labels inherit the rubric's blind spots. An item whose title
avoids every keyword lands in `other`, and `other` is excluded from the clustering
metric rather than being treated as one giant true cluster, which would reward a
system for lumping unrelated leftovers together.

The dedup labels are *not* produced here — those are pairwise judgements recorded in
`eval/datasets/dedup_pairs.jsonl` with a `reason` field on every row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Topic rubric
# ─────────────────────────────────────────────────────────────────────────────
# Ordered: the first matching topic wins. Ordering encodes specificity — an item
# about "federated learning for medical imaging" is more usefully filed under
# healthcare than under federated learning, so healthcare is checked first.

TOPIC_RULES: list[tuple[str, list[str]]] = [
    (
        "healthcare-ai",
        ["clinical", "medical", "patient", "polyp", "emergency department", "diagnosis",
         "mri", "radiolog", "healthcare", "biomed", "ecg", "eeg"],
    ),
    (
        "robotics-autonomy",
        ["robot", "uav", "drone", "autonomous vehicle", "self-driving", "manipulation",
         "embodied", "navigation"],
    ),
    (
        "security",
        ["intrusion", "malware", "ddos", "cyber", "attack", "vulnerabilit", "exploit",
         "red team", "threat", "phishing", "ransomware", "backdoor"],
    ),
    (
        "speech-audio",
        ["speech", "audio", "voice", "tts", "text-to-speech", "asr", "music",
         "acoustic", "phoneme"],
    ),
    (
        "vision-multimodal",
        ["vision-language", "vision language", "multimodal", "image", "video", "visual",
         "diffusion", "gaussian splatting", "segmentation", "detection", "vlm",
         "text-to-image", "3d reconstruction", "lidar", "depth"],
    ),
    (
        "agents-tool-use",
        ["agent", "agentic", "tool use", "tool-use", "orchestrat", "gui agent",
         "multi-agent", "workflow", "planner", "mcp", "function calling"],
    ),
    (
        "memory-retrieval",
        ["memory", "long-context", "long context", "retrieval", "rag", "context window",
         "knowledge base", "recall", "vector"],
    ),
    (
        "rl-planning",
        ["reinforcement learning", " rl ", "policy gradient", "reward", "bandit",
         "planning", "markov decision", "q-learning", "bellman"],
    ),
    (
        "efficiency-inference",
        ["quantiz", "distill", "pruning", "lora", "parameter-efficient", "inference",
         "latency", "throughput", "kv cache", "speculative decoding", "gguf",
         "efficien", "compression", "on-device", "edge"],
    ),
    (
        "safety-eval",
        ["benchmark", "evaluat", "alignment", "safety", "hallucinat", "unlearning",
         "contamination", "fairness", "bias", "robustness", "interpretab",
         "calibration", "uncertainty"],
    ),
    (
        "code-swe",
        ["code generation", "bug", "repair", "software engineering", "program synthesis",
         "compiler", "refactor", "unit test", "static analysis", "coding"],
    ),
    (
        "federated-privacy",
        ["federated", "privacy", "differential privacy", "split learning",
         "secure aggregation"],
    ),
    (
        "nlp-language",
        ["language model", "llm", "multilingual", "translation", "dialect",
         "tokeniz", "text generation", "summariz", "question answering"],
    ),
    (
        "theory-optimization",
        ["conjecture", "theorem", "convergence", "complexity", "monotone", "convex",
         "quantum", "bound", "provab", "stochastic", "optimization", "manifold"],
    ),
    (
        "industry-news",
        ["acquir", "funding", "raises", "ipo", "layoff", "union", "ceo", "lawsuit",
         "antitrust", "partnership"],
    ),
    (
        "consumer-tech",
        ["iphone", "apple watch", "android", "pixel", "macbook", "playstation",
         "nintendo", "game", "console"],
    ),
]

# Item-kind rules run before the keyword rubric: an artefact release is categorically
# a different kind of story from a paper about the same subject.
KIND_RULES: list[tuple[str, re.Pattern]] = [
    ("dataset-release", re.compile(r"\(dataset\)$")),
    ("model-release", re.compile(r"\(model\)$")),
]

RELEASE_SOURCES = {"github-releases-watched"}

OTHER = "other"


def topic_for(title: str, summary: str = "", source_slug: str = "") -> str:
    """Assign one rubric topic. Deterministic and order-dependent by design."""
    text = f"{title} {summary}".lower()

    for label, pattern in KIND_RULES:
        if pattern.search(title.strip()):
            return label
    if source_slug in RELEASE_SOURCES:
        return "framework-release"

    for label, keywords in TOPIC_RULES:
        if any(keyword in text for keyword in keywords):
            return label
    return OTHER


# ─────────────────────────────────────────────────────────────────────────────
# Personas for the relevance set
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Persona:
    """A synthetic user with a stack, used to label (profile, item) relevance.

    Modelled on the PRD's "builder" persona at three different stacks, because
    relevance is only meaningful relative to what someone actually works on.
    """

    key: str
    description: str
    # Rubric topics, used to label relevance.
    topics: frozenset[str]
    # Topics the user would tick in the product's taxonomy. Drawn from the *source*
    # vocabulary, not the rubric vocabulary.
    #
    # These exist because the first ablation run was unfair: personas carried only
    # rubric topics ("agents-tool-use") while items carry source topics
    # ("agentic-ai"), with zero overlap between the two. The topics-only arm could
    # not score at all, which flattered the stack profile by comparison. An ablation
    # whose control arm is handicapped is worse than no ablation.
    declared_topics: frozenset[str]
    dependencies: frozenset[str]
    languages: frozenset[str]


PERSONAS: tuple[Persona, ...] = (
    Persona(
        key="agent-builder",
        description=(
            "Builds LLM agent systems in Python. LangGraph, LangChain, FastAPI, Qdrant. "
            "Cares about orchestration, memory, retrieval and framework releases."
        ),
        topics=frozenset(
            {"agents-tool-use", "memory-retrieval", "framework-release", "nlp-language"}
        ),
        declared_topics=frozenset({"agentic-ai", "tooling", "nlp", "open-source", "releases"}),
        dependencies=frozenset(
            {"langgraph", "langchain", "fastapi", "qdrant", "pydantic", "mcp", "llama_index"}
        ),
        languages=frozenset({"python"}),
    ),
    Persona(
        key="edge-ml",
        description=(
            "Ships on-device inference. Ollama, llama.cpp, GGUF quantization, Flutter "
            "front-ends. Cares about small models, quantization and latency."
        ),
        topics=frozenset({"efficiency-inference", "model-release", "framework-release"}),
        declared_topics=frozenset(
            {"edge-ai", "machine-learning", "open-source", "tooling", "releases"}
        ),
        dependencies=frozenset({"ollama", "llama.cpp", "gguf", "flutter", "onnx", "vllm"}),
        languages=frozenset({"dart", "c++", "python"}),
    ),
    Persona(
        key="cv-researcher",
        description=(
            "Applied computer-vision researcher. PyTorch, diffusion models, "
            "vision-language models. Cares about new methods and datasets."
        ),
        topics=frozenset(
            {"vision-multimodal", "dataset-release", "safety-eval", "theory-optimization"}
        ),
        declared_topics=frozenset({"computer-vision", "research", "machine-learning"}),
        dependencies=frozenset({"pytorch", "torch", "transformers", "diffusers", "timm"}),
        languages=frozenset({"python"}),
    ),
)

PERSONAS_BY_KEY = {p.key: p for p in PERSONAS}


def is_relevant(persona: Persona, title: str, summary: str, topic: str) -> bool:
    """Rubric relevance: the item is in a topic the persona follows, or it names
    something in their dependency list.

    The dependency clause is what makes the stack-profile ablation meaningful — it is
    the only path by which "this affects code you have actually written" can be true.
    """
    if topic in persona.topics:
        return True
    text = f"{title} {summary}".lower()
    return any(dep in text for dep in persona.dependencies)
