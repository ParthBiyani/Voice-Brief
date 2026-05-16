from __future__ import annotations

import uuid

import numpy as np
import pytest

from voicebrief.pipeline.clustering import (
    MIN_ITEMS_FOR_CLUSTERING,
    cluster_items,
    labels_from_clusters,
)


def blob(center: np.ndarray, n: int, spread: float, rng: np.random.Generator) -> np.ndarray:
    """n unit vectors scattered around `center`."""
    noise = rng.normal(0, spread, size=(n, center.shape[0]))
    vectors = center[None, :] + noise
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260606)


@pytest.fixture
def three_blobs(rng):
    """Three well-separated topics, 12 items each."""
    dim = 32
    centers = np.eye(dim, dtype=np.float32)[:3]
    vectors = np.vstack([blob(c, 12, 0.05, rng) for c in centers])
    ids = [uuid.uuid4() for _ in range(len(vectors))]
    return ids, vectors


class TestBasicClustering:
    def test_separated_blobs_become_separate_clusters(self, three_blobs):
        ids, vectors = three_blobs
        clusters = cluster_items(ids, vectors)
        multi = [c for c in clusters if c.size > 1]
        assert len(multi) >= 3
        assert all(c.size <= 15 for c in multi)

    def test_every_item_is_accounted_for(self, three_blobs):
        ids, vectors = three_blobs
        clusters = cluster_items(ids, vectors)
        assigned = {m for c in clusters for m in c.member_ids}
        assert assigned == set(ids), "noise items must survive as singletons"

    def test_centroid_is_a_real_member(self, three_blobs):
        ids, vectors = three_blobs
        for cluster in cluster_items(ids, vectors):
            assert cluster.centroid_id in cluster.member_ids

    def test_clusters_are_sorted_largest_first(self, three_blobs):
        ids, vectors = three_blobs
        sizes = [c.size for c in cluster_items(ids, vectors)]
        assert sizes == sorted(sizes, reverse=True)

    def test_cohesion_is_reported(self, three_blobs):
        ids, vectors = three_blobs
        multi = [c for c in cluster_items(ids, vectors) if c.size > 1]
        assert all(0.0 <= c.cohesion <= 1.0001 for c in multi)


class TestMegaClusterRegression:
    """Regression: on a real 300-item crawl, `cluster_selection_method='eom'` fused
    207 of 300 items into one cluster. An AI-news corpus has a mean pairwise cosine
    around 0.61 — one broad density basin — which is exactly where EOM fails."""

    def test_a_diffuse_corpus_does_not_collapse_into_one_cluster(self, rng):
        # Everything mildly similar to everything else, as in the real corpus.
        dim = 64
        base = np.ones(dim, dtype=np.float32)
        vectors = blob(base, 200, 0.55, rng)
        ids = [uuid.uuid4() for _ in range(200)]

        clusters = cluster_items(ids, vectors)
        largest = max(c.size for c in clusters)
        assert largest < len(ids) * 0.5, (
            f"largest cluster holds {largest}/{len(ids)} items — the corpus collapsed"
        )

    def test_input_vectors_are_not_mutated(self, three_blobs):
        """sklearn's HDBSCAN defaults to copy=False. The same array is reused for
        dedup and Qdrant payloads, so in-place mutation would corrupt both."""
        ids, vectors = three_blobs
        before = vectors.copy()
        cluster_items(ids, vectors)
        np.testing.assert_array_equal(vectors, before)


class TestEdgeCases:
    def test_empty_input(self):
        assert cluster_items([], np.zeros((0, 8), dtype=np.float32)) == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="ids but"):
            cluster_items([uuid.uuid4()], np.zeros((3, 8), dtype=np.float32))

    def test_below_threshold_falls_back_to_singletons(self, rng):
        n = MIN_ITEMS_FOR_CLUSTERING - 1
        ids = [uuid.uuid4() for _ in range(n)]
        vectors = blob(np.eye(16, dtype=np.float32)[0], n, 0.02, rng)
        clusters = cluster_items(ids, vectors)
        assert len(clusters) == n
        assert all(c.size == 1 for c in clusters)

    def test_noise_can_be_dropped_when_requested(self, three_blobs):
        ids, vectors = three_blobs
        kept = cluster_items(ids, vectors, keep_noise_as_singletons=False)
        assert all(c.size >= 1 for c in kept)
        assert sum(c.size for c in kept) <= len(ids)

    def test_identical_vectors_form_one_cluster(self, rng):
        vectors = np.tile(np.eye(16, dtype=np.float32)[0], (12, 1))
        ids = [uuid.uuid4() for _ in range(12)]
        clusters = cluster_items(ids, vectors)
        assert max(c.size for c in clusters) == 12


class TestTopics:
    def test_dominant_topics_are_surfaced(self, three_blobs):
        ids, vectors = three_blobs
        topics = {i: ["agentic-ai", "research"] for i in ids}
        clusters = cluster_items(ids, vectors, topics_by_id=topics)
        multi = next(c for c in clusters if c.size > 1)
        assert "agentic-ai" in multi.topics

    def test_missing_topics_are_tolerated(self, three_blobs):
        ids, vectors = three_blobs
        clusters = cluster_items(ids, vectors, topics_by_id={})
        assert all(c.topics == [] for c in clusters)


def test_labels_map_covers_every_member(three_blobs):
    ids, vectors = three_blobs
    clusters = cluster_items(ids, vectors)
    labels = labels_from_clusters(clusters)
    assert set(labels) == set(ids)
