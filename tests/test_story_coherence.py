"""Clusters must be single stories, not topic bags."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from voicebrief.pipeline.clustering import MIN_PAIRWISE_SIMILARITY, cluster_items
from voicebrief.pipeline.summarize import SUMMARY_SYSTEM

DIM = 48


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def at_similarity(base: np.ndarray, sim: float, rng: np.random.Generator) -> np.ndarray:
    """A unit vector with cosine similarity `sim` to `base`."""
    other = rng.normal(size=base.shape)
    other -= other @ base * base  # make it orthogonal to base
    other = unit(other)
    return unit(sim * base + np.sqrt(1 - sim**2) * other)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def noise_filler(rng, n=12):
    """Unrelated items so the corpus is big enough for HDBSCAN to run."""
    return [unit(rng.normal(size=DIM)) for _ in range(n)]


def ids(n):
    return [uuid.uuid4() for _ in range(n)]


def sizes(clusters):
    return sorted((c.size for c in clusters), reverse=True)


class TestPairwiseTightening:
    def test_a_genuine_story_stays_together(self, rng):
        base = unit(rng.normal(size=DIM))
        story = [base, at_similarity(base, 0.92, rng), at_similarity(base, 0.92, rng)]
        vectors = np.stack(story + noise_filler(rng))
        clusters = cluster_items(ids(len(vectors)), vectors)
        assert sizes(clusters)[0] == 3

    def test_a_loose_topic_group_is_broken_into_singletons(self, rng):
        # Two items that are merely "about AI agents": similar enough for density
        # clustering to join them, not similar enough to be one story.
        base = unit(rng.normal(size=DIM))
        loose = [base, at_similarity(base, 0.62, rng)]
        vectors = np.stack(loose + noise_filler(rng))
        with_tightening = cluster_items(ids(len(vectors)), vectors)
        without = cluster_items(ids(len(vectors)), vectors, min_pairwise=None)
        assert max(c.size for c in with_tightening) == 1
        # Guard against a vacuous test: without tightening they would have merged.
        assert max(c.size for c in without) >= 2

    def test_the_weakest_member_is_shed_not_the_whole_story(self, rng):
        base = unit(rng.normal(size=DIM))
        members = [
            base,
            at_similarity(base, 0.93, rng),
            at_similarity(base, 0.93, rng),
            at_similarity(base, 0.50, rng),  # the stray
        ]
        vectors = np.stack(members + noise_filler(rng))
        item_ids = ids(len(vectors))
        clusters = cluster_items(item_ids, vectors, min_pairwise=MIN_PAIRWISE_SIMILARITY)
        stray_cluster = next(c for c in clusters if item_ids[3] in c.member_ids)
        assert stray_cluster.size == 1
        core = next(c for c in clusters if item_ids[0] in c.member_ids)
        assert core.size == 3

    def test_no_item_is_lost(self, rng):
        vectors = np.stack([unit(rng.normal(size=DIM)) for _ in range(30)])
        item_ids = ids(30)
        clusters = cluster_items(item_ids, vectors)
        assert sorted(m for c in clusters for m in c.member_ids) == sorted(item_ids)


class TestSoloSources:
    def test_solo_items_never_merge_even_when_nearly_identical(self, rng):
        base = unit(rng.normal(size=DIM))
        twins = [base, at_similarity(base, 0.97, rng)]
        vectors = np.stack(twins + noise_filler(rng))
        item_ids = ids(len(vectors))
        clusters = cluster_items(item_ids, vectors, solo_ids={item_ids[0], item_ids[1]})
        for c in clusters:
            assert not ({item_ids[0], item_ids[1]} <= set(c.member_ids))

    def test_a_solo_item_does_not_stop_others_forming_a_story(self, rng):
        base = unit(rng.normal(size=DIM))
        story = [base, at_similarity(base, 0.93, rng), at_similarity(base, 0.93, rng)]
        vectors = np.stack(story + noise_filler(rng))
        item_ids = ids(len(vectors))
        clusters = cluster_items(item_ids, vectors, solo_ids={item_ids[2]})
        merged = next(c for c in clusters if item_ids[0] in c.member_ids)
        assert item_ids[2] not in merged.member_ids
        assert item_ids[1] in merged.member_ids


def test_summarizer_is_told_to_write_one_story_not_a_survey():
    assert "ONE story" in SUMMARY_SYSTEM
    assert "source 0" in SUMMARY_SYSTEM


class TestNoRepeatedEventInOneBrief:
    @staticmethod
    def story(title, score, *source_titles):
        from voicebrief.pipeline.ranking import RankCandidate, RankedStory
        from voicebrief.pipeline.summarize import SourceRef

        sources = [
            SourceRef(item_id=uuid.uuid4(), title=t, url=f"https://x/{i}", source_slug="s")
            for i, t in enumerate(source_titles)
        ]
        candidate = RankCandidate(
            cluster_id=uuid.uuid4(), title=title, summary="", url="https://x", sources=sources
        )
        return RankedStory(candidate=candidate, score=score, heuristic_score=score)

    def test_three_fragments_of_one_launch_become_one_segment(self):
        from voicebrief.pipeline.ranking import select_for_episode

        stories = [
            self.story("huggingface/transformers v5.22.0: Mistral Large 4 (Le Chonk)", 3.5),
            self.story("We're releasing the weights of Mistral Large 4", 3.4),
            self.story("mistralai/Mistral-Large-4-1T-A52B (model)", 2.8),
            self.story("Google Search is now less than half of Alphabet's revenue", 2.7),
            self.story("PyTorch 2.15 adds a faster compiler backend", 2.6),
        ]
        picked = select_for_episode(stories, max_stories=3, min_personal=0)
        titles = [s.candidate.title for s in picked]
        assert sum("Mistral" in t or "mistral" in t for t in titles) == 1
        assert len(picked) == 3

    def test_a_thin_day_still_fills_the_brief(self):
        from voicebrief.pipeline.ranking import select_for_episode

        stories = [
            self.story("Mistral Large 4 weights released", 3.0),
            self.story("Mistral Large 4 on Hugging Face", 2.0),
        ]
        assert len(select_for_episode(stories, max_stories=2, min_personal=0)) == 2

    def test_unrelated_stories_are_untouched(self):
        from voicebrief.pipeline.ranking import covers_same_ground, story_tokens

        a = story_tokens(self.story("Gemini 4 Carbon launches", 1.0))
        b = story_tokens(self.story("Google Search is now less than half of revenue", 1.0))
        assert not covers_same_ground(a, b)
