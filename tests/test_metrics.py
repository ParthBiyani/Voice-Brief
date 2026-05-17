"""Metric tests.

These check the metric implementations against cases with known answers, because a
subtly wrong metric produces a confident, plausible, wrong number — the worst kind
of bug in an evaluation harness.
"""

from __future__ import annotations

import pytest

from voicebrief.evalkit.metrics import (
    adjusted_rand_index,
    attribution_rate,
    clustered_fraction,
    homogeneity,
    ndcg_at_k,
    pairwise_prf,
    precision_at_k,
    recall_at_k,
)


def pair(a, b) -> frozenset:
    return frozenset((a, b))


class TestPairwisePRF:
    def test_perfect_prediction(self):
        truth = {pair(1, 2), pair(3, 4)}
        r = pairwise_prf(truth, truth)
        assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)

    def test_half_precision_half_recall(self):
        r = pairwise_prf({pair(1, 2), pair(5, 6)}, {pair(1, 2), pair(3, 4)})
        assert r.precision == 0.5
        assert r.recall == 0.5
        assert r.f1 == 0.5

    def test_no_predictions_when_truth_is_empty_is_perfect(self):
        r = pairwise_prf(set(), set())
        assert r.precision == 1.0 and r.recall == 1.0

    def test_missing_everything_scores_zero_recall(self):
        r = pairwise_prf(set(), {pair(1, 2)})
        assert r.recall == 0.0
        assert r.false_negatives == 1

    def test_pairs_are_unordered(self):
        assert pairwise_prf({pair(2, 1)}, {pair(1, 2)}).precision == 1.0


class TestAdjustedRandIndex:
    def test_identical_labelings_score_one(self):
        labels = {"a": 0, "b": 0, "c": 1, "d": 1}
        assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)

    def test_relabeled_but_equivalent_partition_scores_one(self):
        """ARI must be invariant to what the cluster ids are called."""
        a = {"w": 0, "x": 0, "y": 1, "z": 1}
        b = {"w": 7, "x": 7, "y": 3, "z": 3}
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)

    def test_completely_wrong_partition_scores_near_zero_or_below(self):
        truth = {"a": 0, "b": 0, "c": 1, "d": 1}
        crossed = {"a": 0, "b": 1, "c": 0, "d": 1}
        assert adjusted_rand_index(crossed, truth) < 0.1

    def test_all_singletons_against_real_clusters_is_not_rewarded(self):
        """The reason ARI is used instead of the plain Rand index: a degenerate
        all-singleton clustering must not look good."""
        truth = {k: i // 3 for i, k in enumerate("abcdefghi")}
        singletons = {k: i for i, k in enumerate("abcdefghi")}
        assert adjusted_rand_index(singletons, truth) < 0.1

    def test_only_shared_keys_are_compared(self):
        a = {"x": 0, "y": 0, "extra": 5}
        b = {"x": 1, "y": 1}
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)

    def test_degenerate_sizes(self):
        assert adjusted_rand_index({}, {}) == 0.0
        assert adjusted_rand_index({"a": 0}, {"a": 0}) == 1.0


class TestRankingMetrics:
    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_precision_at_k_with_short_list(self):
        assert precision_at_k(["a"], {"a"}, 10) == 1.0

    def test_recall_at_k(self):
        assert recall_at_k(["a", "b"], {"a", "c"}, 2) == 0.5

    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_ndcg_rewards_putting_relevant_items_first(self):
        good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        bad = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert good > bad

    def test_ndcg_is_bounded(self):
        assert 0.0 <= ndcg_at_k(["x", "y", "a"], {"a"}, 3) <= 1.0

    def test_ndcg_with_no_relevant_items_is_zero(self):
        assert ndcg_at_k(["a", "b"], set(), 2) == 0.0

    def test_ndcg_ideal_accounts_for_k_smaller_than_relevant_set(self):
        """With 5 relevant items but k=2, retrieving 2 relevant items is perfect."""
        assert ndcg_at_k(["a", "b"], {"a", "b", "c", "d", "e"}, 2) == pytest.approx(1.0)


def test_attribution_rate():
    assert attribution_rate(95, 100) == 0.95
    assert attribution_rate(0, 0) == 1.0


class TestHomogeneity:
    def test_pure_clusters_score_one(self):
        predicted = {"a": 0, "b": 0, "c": 1, "d": 1}
        truth = {"a": 9, "b": 9, "c": 4, "d": 4}
        assert homogeneity(predicted, truth) == pytest.approx(1.0)

    def test_mixed_cluster_is_penalised(self):
        predicted = {"a": 0, "b": 0, "c": 0, "d": 0}
        truth = {"a": 1, "b": 1, "c": 2, "d": 3}
        assert homogeneity(predicted, truth) == pytest.approx(0.5)

    def test_singletons_are_excluded_so_purity_cannot_be_gamed(self):
        """Refusing to cluster must not score as perfect purity."""
        predicted = {k: i for i, k in enumerate("abcd")}
        truth = dict.fromkeys("abcd", 1)
        assert homogeneity(predicted, truth) == 0.0

    def test_story_level_clusters_against_topic_labels_score_well(self):
        """The case ARI gets wrong: many small, correct clusters inside big topics."""
        predicted = {f"i{n}": n // 2 for n in range(20)}
        truth = {f"i{n}": n // 10 for n in range(20)}
        assert homogeneity(predicted, truth) > 0.9
        assert adjusted_rand_index(predicted, truth) < 0.3

    def test_empty_input(self):
        assert homogeneity({}, {}) == 0.0


class TestClusteredFraction:
    def test_all_singletons_is_zero(self):
        assert clustered_fraction({k: i for i, k in enumerate("abc")}) == 0.0

    def test_all_grouped_is_one(self):
        assert clustered_fraction(dict.fromkeys("abcd", 0)) == 1.0

    def test_mixed(self):
        assert clustered_fraction({"a": 0, "b": 0, "c": 1, "d": 2}) == 0.5

    def test_empty(self):
        assert clustered_fraction({}) == 0.0
