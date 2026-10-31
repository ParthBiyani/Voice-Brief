"""Topics from a story's own words, and how they enter ranking."""

from __future__ import annotations

import uuid

import pytest

from voicebrief.pipeline.ranking import (
    W_SOURCE_TOPIC,
    W_TOPIC,
    RankCandidate,
    is_ambiguous_dependency,
    score_parts,
)
from voicebrief.pipeline.topics import content_topics


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        ("JarvisGUI: cross-device GUI agents with dynamic task planning", "agentic-ai"),
        ("A retrieval-augmented generation pipeline over a vector database", "retrieval"),
        ("Segmentation of satellite images with a vision-language model", "computer-vision"),
        ("Qwen 3 quantized to GGUF runs on device at 40 tokens/s", "edge-ai"),
        ("Kimi K3 runs locally on a MacBook", "edge-ai"),
        ("A multilingual tokenizer for low-resource translation", "nlp"),
        ("Startup raises $40M for AI chips", "industry"),
    ],
)
def test_specific_terms_name_the_topic(text, topic):
    assert topic in content_topics(text)


@pytest.mark.parametrize(
    ("text", "not_topic"),
    [
        ("Deep learning-based detection of electrical faults", "computer-vision"),
        ("A cutting edge result in theorem proving", "edge-ai"),
        ("Training dynamics of large neural networks", "agentic-ai"),
        ("GPU memory bandwidth limits decoding speed", "retrieval"),
        ("The new framework simplifies configuration", "tooling"),
    ],
)
def test_generic_words_claim_nothing(text, not_topic):
    assert not_topic not in content_topics(text)


def test_matching_is_whole_word():
    assert "nlp" not in content_topics("The llmish vibes of the day")  # not "llm"
    assert "edge-ai" in content_topics("4-bit quantization")


def candidate(title, summary="", topics=()):
    return RankCandidate(
        cluster_id=uuid.uuid4(),
        title=title,
        summary=summary,
        url="https://x",
        topics=list(topics),
        engagement=0.0,
        trust_weight=0.6,
        cluster_size=1,
        source_slug="arxiv-cs-ai",
    )


class TestRanking:
    def test_what_a_story_is_about_beats_its_feeds_blanket_tag(self):
        tags = ["agentic-ai", "machine-learning", "research"]
        vision = candidate("Segmentation of satellite imagery with diffusion priors", topics=tags)
        agents = candidate("GUI agents that plan across devices", topics=tags)
        declared = {"computer-vision"}
        v = score_parts(vision, profile=None, declared_topics=declared)[2]["topics"]
        a = score_parts(agents, profile=None, declared_topics=declared).__getitem__(2)
        assert v == pytest.approx(W_TOPIC) and "topics" not in a

    def test_a_tag_without_content_is_only_a_prior(self):
        paper = candidate("On the convergence of a gradient method", topics=["agentic-ai"])
        parts = score_parts(paper, profile=None, declared_topics={"agentic-ai"})[2]
        assert parts["topics"] == pytest.approx(W_SOURCE_TOPIC)

    def test_an_architecture_is_not_a_library(self):
        assert is_ambiguous_dependency("transformers")
        assert is_ambiguous_dependency("datasets")
