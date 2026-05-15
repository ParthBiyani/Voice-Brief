"""Dedup unit tests.

Vectors here are hand-built rather than model-generated so the tests assert on the
grouping logic, not on a model's behaviour. Model-in-the-loop quality is measured by
the eval harness against labelled data, which is a different question.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from voicebrief.pipeline.dedup import (
    DedupCandidate,
    canonical_url,
    find_duplicates,
    normalized_title,
    pairs_from_groups,
    title_fingerprint,
    url_fingerprint,
)


def cand(title="A Story", url="https://example.com/a", **over) -> DedupCandidate:
    base = dict(id=uuid.uuid4(), title=title, url=url, trust_weight=0.5, engagement=0.0,
                published_ts=1000.0)
    return DedupCandidate(**{**base, **over})


def unit(*values) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestCanonicalUrl:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("https://example.com/a?utm_source=twitter", "https://example.com/a"),
            ("https://example.com/a/", "https://example.com/a"),
            ("https://WWW.Example.com/a", "https://example.com/a"),
            ("https://example.com/a#section", "https://example.com/a"),
            ("https://example.com/a?ref=hn&gclid=x", "https://example.com/a"),
        ],
    )
    def test_strips_noise(self, given, expected):
        assert canonical_url(given) == expected

    def test_meaningful_query_params_are_preserved(self):
        assert canonical_url("https://example.com/a?id=42") == "https://example.com/a?id=42"

    def test_empty_is_safe(self):
        assert canonical_url("") == ""


class TestNormalizedTitle:
    def test_punctuation_and_case_are_ignored(self):
        a = normalized_title("LangGraph 0.4: Durable Execution!")
        b = normalized_title("langgraph 0 4 - durable execution")
        assert a == b

    def test_fingerprints_match_for_equivalent_titles(self):
        assert title_fingerprint("A Post: One") == title_fingerprint("a post one")

    def test_fingerprints_differ_for_different_titles(self):
        assert title_fingerprint("Alpha release") != title_fingerprint("Beta release")


class TestExactMatching:
    def test_same_url_groups_without_vectors(self):
        a = cand(title="Story A", url="https://example.com/x?utm_source=hn")
        b = cand(title="Different Wording", url="https://example.com/x")
        groups = find_duplicates([a, b])
        assert len(groups) == 1
        assert groups[0].size == 2
        assert groups[0].reason == "url"

    def test_same_title_different_url_groups(self):
        a = cand(title="LangGraph 0.4 ships", url="https://one.com/a")
        b = cand(title="langgraph 0 4 ships", url="https://two.com/b")
        assert len(find_duplicates([a, b])) == 1

    def test_unrelated_items_do_not_group(self):
        a = cand(title="Story about cats", url="https://one.com/a")
        b = cand(title="Story about databases", url="https://two.com/b")
        assert find_duplicates([a, b]) == []


class TestSemanticMatching:
    def test_similar_vectors_group_above_threshold(self):
        a, b = cand(title="Alpha one two", url="https://a.com/1"), cand(
            title="Beta three four", url="https://b.com/2"
        )
        vectors = np.stack([unit(1, 0, 0), unit(0.99, 0.14, 0)])
        groups = find_duplicates([a, b], vectors, threshold=0.92)
        assert len(groups) == 1

    def test_dissimilar_vectors_do_not_group(self):
        a, b = cand(title="Alpha one two", url="https://a.com/1"), cand(
            title="Beta three four", url="https://b.com/2"
        )
        vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0)])
        assert find_duplicates([a, b], vectors, threshold=0.92) == []

    def test_threshold_is_respected(self):
        a, b = cand(title="Alpha one", url="https://a.com/1"), cand(
            title="Beta two", url="https://b.com/2"
        )
        vectors = np.stack([unit(1, 0, 0), unit(0.95, 0.31, 0)])
        assert find_duplicates([a, b], vectors, threshold=0.99) == []
        assert len(find_duplicates([a, b], vectors, threshold=0.90)) == 1

    def test_duplication_is_transitive(self):
        """A≈B and B≈C must yield one group of three, not two groups of two."""
        items = [
            cand(title="Alpha one", url="https://a.com/1"),
            cand(title="Beta two", url="https://b.com/2"),
            cand(title="Gamma three", url="https://c.com/3"),
        ]
        vectors = np.stack([unit(1, 0, 0), unit(0.99, 0.14, 0), unit(0.97, 0.24, 0)])
        groups = find_duplicates(items, vectors, threshold=0.92)
        assert len(groups) == 1
        assert groups[0].size == 3


class TestCanonicalElection:
    def test_highest_trust_wins(self):
        low = cand(title="Same story here", url="https://a.com/1", trust_weight=0.3)
        high = cand(title="same story here", url="https://b.com/2", trust_weight=0.9)
        groups = find_duplicates([low, high])
        assert groups[0].canonical_id == high.id

    def test_engagement_breaks_a_trust_tie(self):
        quiet = cand(title="Same story here", url="https://a.com/1", trust_weight=0.5, engagement=0.1)
        loud = cand(title="same story here", url="https://b.com/2", trust_weight=0.5, engagement=0.9)
        assert find_duplicates([quiet, loud])[0].canonical_id == loud.id

    def test_earliest_publication_breaks_remaining_ties(self):
        first = cand(title="Same story here", url="https://a.com/1", published_ts=100.0)
        later = cand(title="same story here", url="https://b.com/2", published_ts=900.0)
        assert find_duplicates([first, later])[0].canonical_id == first.id

    def test_grouping_is_order_independent(self):
        """The eval harness compares group membership across runs; instability there
        would show up as phantom metric drift."""
        a = cand(title="Same story here", url="https://a.com/1", trust_weight=0.7)
        b = cand(title="same story here", url="https://b.com/2", trust_weight=0.4)
        c = cand(title="SAME story, here!", url="https://c.com/3", trust_weight=0.2)

        forward = find_duplicates([a, b, c])
        backward = find_duplicates([c, b, a])
        assert forward[0].canonical_id == backward[0].canonical_id
        assert set(forward[0].member_ids) == set(backward[0].member_ids)


class TestEdgeCases:
    def test_empty_input(self):
        assert find_duplicates([]) == []

    def test_single_item(self):
        assert find_duplicates([cand()]) == []

    def test_mismatched_vector_count_falls_back_to_exact_only(self):
        a, b = cand(title="Alpha one", url="https://a.com/1"), cand(
            title="Beta two", url="https://b.com/2"
        )
        assert find_duplicates([a, b], np.stack([unit(1, 0, 0)])) == []

    def test_pairs_expansion(self):
        items = [
            cand(title="Same story here", url="https://a.com/1"),
            cand(title="same story here", url="https://b.com/2"),
            cand(title="Same Story Here!", url="https://c.com/3"),
        ]
        pairs = pairs_from_groups(find_duplicates(items))
        assert len(pairs) == 3, "a group of 3 expands to 3 unordered pairs"
