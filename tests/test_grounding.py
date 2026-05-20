"""Grounding verifier tests.

Nearly every case here comes from a real failure observed while generating actual
episodes. The verifier's first version scored a correct script at 0.43 because it
disagreed with the script generator about how numbers are written; these lock the
resolution in.
"""

from __future__ import annotations

import uuid

import pytest

from voicebrief.pipeline.grounding import (
    content_words,
    is_claim,
    spoken_numbers,
    split_sentences,
    verify_segment,
)
from voicebrief.pipeline.summarize import ClusterSummary, SourceRef


def summary(facts: list[str], *, headline="A release", urls=("https://example.com/a",)):
    sources = [
        SourceRef(item_id=uuid.uuid4(), title=headline, url=u, source_slug="test") for u in urls
    ]
    return ClusterSummary(
        cluster_id=uuid.uuid4(),
        headline=headline,
        what_changed=facts[0] if facts else "",
        why_it_matters="",
        key_facts=[
            {"fact": f, "url": urls[0], "title": headline, "item_id": "1"} for f in facts
        ],
        sources=sources,
    )


class TestSpokenNumbers:
    """The script generator is told to speak numerals; the verifier must read them."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("four point two", "4.2"),
            ("eighty two percent", "82"),
            ("thirty-three thousand two hundred thirty-one", "33231"),
            ("two thousand four hundred seventy six", "2476"),
            ("three billion", "3000000000"),
            ("ninety seven", "97"),
        ],
    )
    def test_parses_spoken_forms(self, text, expected):
        assert expected in spoken_numbers(text)

    def test_hyphens_are_number_internal(self):
        """'thirty-three thousand' is one figure, not 30 and 3000."""
        assert "33000" in spoken_numbers("thirty-three thousand")

    def test_commas_separate_distinct_figures(self):
        """'four point two, three billion' is two numbers. Reading it as one produced
        the phantom value 4000000000.23 on a real episode."""
        found = spoken_numbers("Nanbeige four point two, three billion parameters")
        assert "4.2" in found and "3000000000" in found
        assert not any(f.startswith("4000000000") for f in found)

    def test_billions_also_yield_the_mantissa(self):
        """'three billion' must satisfy a source that wrote '3B'."""
        assert "3" in spoken_numbers("three billion parameters")

    def test_prose_without_numbers_yields_nothing(self):
        assert spoken_numbers("a method that adapts at test time") == set()


class TestContentWords:
    def test_number_words_are_excluded(self):
        """Otherwise a spoken figure counts as eight unmatched prose words against a
        source that simply wrote 33,231."""
        words = content_words("thirty three thousand downloads")
        assert words == {"downloads"}

    def test_identifiers_also_emit_their_stem(self):
        assert "nanbeige" in content_words("Nanbeige4.2-3B")

    def test_stopwords_are_excluded(self):
        assert content_words("it is the one that we have") == set()


class TestClaimDetection:
    @pytest.mark.parametrize(
        "sentence",
        [
            "It has thirty-three thousand downloads.",
            "DIPTTA is a test-time adaptation method.",
            "MiniCPM5 was released by OpenBMB.",
            "Throughput rose 12 percent.",
        ],
    )
    def test_specific_sentences_are_claims(self, sentence):
        assert is_claim(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "Four papers this week tackle the same problem from different angles.",
            "The point is to stop catastrophic forgetting.",
            "Here is what changed today.",
        ],
    )
    def test_connective_prose_is_not_a_claim(self, sentence):
        """Scoring narration against lexical overlap measures writing style, not
        factuality — it dragged a correct script from 0.95 to 0.49."""
        assert not is_claim(sentence)


class TestVerification:
    def test_grounded_claim_is_supported(self):
        s = summary(["LangGraph 0.4 adds durable execution with checkpointing"])
        report = verify_segment("LangGraph zero point four adds durable execution.", s)
        assert report.grounded_ratio == 1.0

    def test_invented_number_is_caught(self):
        """The failure that matters most in a news brief."""
        s = summary(["Throughput improved by 12 percent"])
        report = verify_segment("Throughput improved by ninety seven percent.", s)
        assert report.supported_sentences == 0
        assert report.unsupported

    def test_url_outside_the_source_set_is_flagged(self):
        s = summary(["A release happened"], urls=("https://example.com/a",))
        report = verify_segment("See https://evil.example.org/made-up for details.", s)
        assert report.hallucinated_links == ["https://evil.example.org/made-up"]

    def test_a_cited_url_is_not_flagged(self):
        s = summary(["A release happened"], urls=("https://example.com/a",))
        report = verify_segment("Details at https://example.com/a here.", s)
        assert report.hallucinated_links == []

    def test_source_documents_count_as_sources(self):
        """The PRD's bar is attribution to a source span, so the articles are the
        comparison target — not the summarizer's lossy compression of them."""
        s = summary(["A release happened"])
        script = "The benchmark covers eighteen object types and twenty risk categories."
        without = verify_segment(script, s)
        with_source = verify_segment(
            script,
            s,
            source_text="benchmark covering 18 object types and 20 risk categories",
        )
        assert with_source.grounded_ratio > without.grounded_ratio

    def test_listener_vocabulary_is_grounded(self):
        """'you already use transformers in Voice-Brief' is true, sourced from the
        stack profile. Flagging it would penalise the product's whole point."""
        s = summary(["A transformers-compatible model was released"])
        script = "It is transformers compatible, the library you use in Voice-Brief."
        report = verify_segment(
            script, s, extra_vocabulary={"voice-brief", "transformers", "you"}
        )
        assert report.grounded_ratio == 1.0

    def test_narration_is_excluded_from_the_denominator(self):
        s = summary(["A release happened"])
        report = verify_segment("Here is what changed. The point is clear.", s)
        assert report.total_sentences == 0
        assert report.skipped_narration == 2

    def test_empty_script_is_vacuously_grounded(self):
        assert verify_segment("", summary(["x"])).grounded_ratio == 1.0


class TestSentenceSplitting:
    def test_splits_on_terminators(self):
        assert len(split_sentences("One thing. Two things! Three things?")) == 3

    def test_two_host_labels_are_stripped(self):
        parts = split_sentences("HOST A: First point. HOST B: Second point.")
        assert not any("HOST" in p for p in parts)

    def test_empty_text(self):
        assert split_sentences("   ") == []
