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
    _numbers,
    content_words,
    is_claim,
    split_sentences,
    spoken_numbers,
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
        key_facts=[{"fact": f, "url": urls[0], "title": headline, "item_id": "1"} for f in facts],
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
        report = verify_segment(script, s, extra_vocabulary={"voice-brief", "transformers", "you"})
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


class TestSpokenVersionsAndMagnitudes:
    """Regressions from the first real episodes: correct sentences scored 0.66 because
    the verifier could not read the way a script says versions and sizes aloud."""

    @pytest.mark.parametrize(
        ("spoken", "expected"),
        [
            ("version one point seven point one", "1.7.1"),
            ("eighty-three point zero point zero", "83.0.0"),
            ("five point twenty-two", "5.22"),
            ("version four point two", "4.2"),
            ("one trillion total parameters", "1000000000000"),
            ("forty-nine billion are active", "49000000000"),
            ("thirty-eight hundred GPUs", "3800"),
            ("one hundred twenty-eight routed experts", "128"),
        ],
    )
    def test_speech_is_read_back_as_the_number_a_source_would_write(self, spoken, expected):
        assert expected in spoken_numbers(spoken)

    def test_a_version_in_the_source_supports_its_shorter_spoken_form(self):
        # The source says 5.22.0; the script says "five point twenty-two".
        assert "5.22" in _numbers("huggingface/transformers v5.22.0")

    def test_trillion_suffix_in_a_source_matches_a_spoken_trillion(self):
        assert "1000000000000" in _numbers("Mistral Large 4 is a 1T parameter model")

    def test_a_fabricated_version_is_still_caught(self):
        source = _numbers("langchain-core 1.7.1 released")
        assert "1.7.9" not in source
        assert "1.7.9" in spoken_numbers("version one point seven point nine")


class TestFigureOnlyClaims:
    def test_a_claim_of_only_numbers_does_not_crash_and_is_checked_by_its_numbers(self):
        facts = summary(["Version 5.22.0 shipped"])
        report = verify_segment("5.22.0.", facts, source_text="transformers v5.22.0 shipped")
        assert report.supported_sentences == 1

    def test_a_figure_only_claim_with_an_invented_number_is_unsupported(self):
        facts = summary(["Version 5.22.0 shipped"])
        report = verify_segment("9.99.9.", facts, source_text="transformers v5.22.0 shipped")
        assert report.unsupported


class TestCountingAndShortClaims:
    def test_counting_what_is_in_front_of_you_is_not_an_invented_figure(self):
        facts = summary(["langchain-core 1.7.1 and 1.7.2 were released"])
        report = verify_segment(
            "langchain-core has two new releases.", facts, source_text="langchain-core 1.7.1 1.7.2"
        )
        assert report.unsupported == []

    def test_a_percentage_is_still_a_figure_that_must_be_sourced(self):
        facts = summary(["Throughput improved"])
        report = verify_segment(
            "Throughput improved by ten percent.", facts, source_text="Throughput improved"
        )
        assert report.unsupported

    def test_a_larger_invented_count_is_still_caught(self):
        facts = summary(["The release fixes bugs"])
        report = verify_segment(
            "The release fixes forty bugs.", facts, source_text="The release fixes bugs"
        )
        assert report.unsupported

    def test_a_short_sentence_with_a_verified_version_is_supported(self):
        facts = summary(["Version 1.7.2 shipped"])
        report = verify_segment(
            "Start with one point seven point two.", facts, source_text="Version 1.7.2 shipped"
        )
        assert report.supported_sentences == 1

    def test_written_out_magnitudes_in_a_source_match_a_spoken_figure(self):
        facts = summary(["49 billion parameters are active per token"])
        report = verify_segment(
            "Forty-nine billion parameters are active per token.",
            facts,
            source_text="49 billion parameters are active per token",
        )
        assert report.supported_sentences == 1


class TestPhrasingThatIsFaithfulButNotLiteral:
    """From a real episode that scored 0.36 though its content was right: the verifier
    penalised rewordings that a spoken script legitimately makes."""

    def test_counting_the_sources_structure_is_narration_not_a_claim(self):
        for sentence in (
            "The material gives three reasons.",
            "There is one practical detail about versions.",
            "So the case comes down to four points.",
        ):
            assert not is_claim(sentence), sentence

    def test_a_percentage_is_still_a_claim_even_when_small(self):
        assert is_claim("Throughput improved by ten percent.")

    def test_a_spoken_ordinal_percentile_reads_as_the_figure_a_source_writes(self):
        assert "95" in spoken_numbers("the ninety-fifth percentile")
        assert "90" not in spoken_numbers("the ninety-fifth percentile")

    def test_plain_ordinals_in_prose_are_not_numbers(self):
        assert spoken_numbers("first, second and third") == set()

    def test_ms_and_p95_in_a_source_support_milliseconds_and_percentile_in_a_script(self):
        facts = summary(["p95 query latency was 12 ms at 100,000 vectors"])
        report = verify_segment(
            "That means queries came back in twelve milliseconds at the ninety-fifth percentile.",
            facts,
            source_text="p95 query latency was 12 ms at 100,000 vectors",
        )
        assert report.unsupported == [] and report.supported_sentences == 1

    def test_plurals_match_their_singular(self):
        facts = summary(["The index handles each query quickly"])
        report = verify_segment(
            "The indexes handle queries quickly.",
            facts,
            source_text="The index handles each query quickly",
        )
        assert report.unsupported == []

    def test_a_fabricated_percentile_is_still_caught(self):
        facts = summary(["p95 query latency was 12 ms"])
        report = verify_segment(
            "Latency was twelve milliseconds at the ninety-ninth percentile.",
            facts,
            source_text="p95 query latency was 12 ms",
        )
        assert report.unsupported, "99 does not appear in the source"

    def test_an_invented_figure_is_still_caught_after_all_the_leniency(self):
        facts = summary(["latency was 12 ms"])
        report = verify_segment(
            "Latency was forty milliseconds.", facts, source_text="latency was 12 ms"
        )
        assert report.unsupported


class TestFlagsFromARealEpisode:
    """Both sentences were correct and were flagged: a verifier bug, not a script bug."""

    SPAWN = (
        "About 60% of our work starts from a production error, a Savi bug report, "
        "something odd in our logs, or a feature request."
    )

    def test_counts_beside_a_percentage_are_still_ordinary_counts(self):
        facts = summary([self.SPAWN])
        report = verify_segment(
            "About sixty percent of the work starts from one of four places: a production error, "
            "a Savi bug report, something odd in the logs, or a feature request.",
            facts,
            source_text=self.SPAWN,
        )
        assert report.unsupported == [] and report.supported_sentences == 1

    def test_the_percentage_itself_must_still_be_sourced(self):
        facts = summary([self.SPAWN])
        report = verify_segment(
            "About ninety percent of the work starts from a production error or a bug report.",
            facts,
            source_text=self.SPAWN,
        )
        assert report.unsupported, "90 does not appear in the source"

    def test_a_small_percentage_is_not_waved_through_as_a_count(self):
        facts = summary([self.SPAWN])
        report = verify_segment(
            "About ten percent of the work starts from a production error.",
            facts,
            source_text=self.SPAWN,
        )
        assert report.unsupported, "ten percent is a figure, not a count"

    def test_a_compound_in_the_source_supports_its_parts_in_the_script(self):
        text = "Against open-source baselines, our 1.8M-parameter model attains the highest F1."
        report = verify_segment(
            "The model has 1.8 million parameters.", summary([text]), source_text=text
        )
        assert report.unsupported == [] and report.supported_sentences == 1

    def test_a_wrong_figure_next_to_a_matching_compound_is_still_caught(self):
        text = "our 1.8M-parameter model attains the highest F1 on every test set"
        report = verify_segment(
            "The model has 3.2 million parameters.", summary([text]), source_text=text
        )
        assert report.unsupported
