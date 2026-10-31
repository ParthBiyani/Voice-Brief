"""Thumbs shape tomorrow's ranking: a bounded, smoothed nudge that evidence can outweigh."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from voicebrief.db.models import Feedback
from voicebrief.personalization.feedback import (
    Affinity,
    affinity_from_votes,
    examples_block,
    load_affinity,
)
from voicebrief.pipeline.ranking import (
    W_DEPENDENCY,
    W_FEEDBACK,
    RankCandidate,
    prerank,
    rerank,
    score_parts,
)

NOW = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


def candidate(title="A story", source="blog-a", topics=None, **kw) -> RankCandidate:
    return RankCandidate(
        cluster_id=uuid.uuid4(),
        title=title,
        summary="Something happened.",
        url=f"https://example.com/{uuid.uuid4().hex}",
        topics=topics or ["ai"],
        source_slug=source,
        **kw,
    )


def votes(*specs):
    """(segment, vote, heading, source, topics)"""
    return [(uuid.UUID(int=seg), vote, head, src, tops) for seg, vote, head, src, tops in specs]


class TestAffinity:
    def test_one_thumb_is_a_small_nudge_not_a_verdict(self):
        a = affinity_from_votes(votes((1, 1, "Liked", "blog-a", ["ai"])))
        assert 0 < a.source["blog-a"] < 0.5

    def test_a_pattern_moves_a_source_further_than_a_single_vote(self):
        one = affinity_from_votes(votes((1, -1, "x", "blog-a", [])))
        many = affinity_from_votes(votes(*[(i, -1, f"x{i}", "blog-a", []) for i in range(1, 7)]))
        assert many.source["blog-a"] < one.source["blog-a"] < 0

    def test_mixed_votes_cancel(self):
        a = affinity_from_votes(votes((1, 1, "up", "blog-a", []), (2, -1, "down", "blog-a", [])))
        assert "blog-a" not in a.source, "an exact tie carries no information"

    def test_a_segment_built_from_several_items_counts_once_per_source(self):
        a = affinity_from_votes(
            votes(
                (1, 1, "H", "blog-a", ["ai"]),
                (1, 1, "H", "blog-a", ["ai"]),
                (1, 1, "H", "hn", ["ai"]),
            )
        )
        single = affinity_from_votes(votes((1, 1, "H", "blog-a", ["ai"])))
        assert a.source["blog-a"] == single.source["blog-a"]
        assert a.votes == 1

    def test_topics_are_case_insensitive(self):
        a = affinity_from_votes(votes((1, 1, "H", "s", ["LLM"])))
        assert "llm" in a.topic

    def test_affinity_is_always_inside_open_unit_interval(self):
        a = affinity_from_votes(votes(*[(i, 1, f"h{i}", "s", ["t"]) for i in range(1, 500)]))
        assert 0 < a.source["s"] < 1 and 0 < a.topic["t"] < 1

    def test_value_is_zero_for_the_unknown(self):
        a = affinity_from_votes(votes((1, 1, "H", "blog-a", ["ai"])))
        assert a.value("never-seen", ["never-seen-topic"]) == 0.0

    def test_source_and_topic_are_averaged(self):
        a = Affinity(source={"s": 0.6}, topic={"t": -0.2}, votes=2)
        assert abs(a.value("s", ["t"]) - 0.2) < 1e-9


class TestRankingEffect:
    def test_a_disliked_source_ranks_below_an_equal_story_from_a_liked_one(self):
        affinity = Affinity(source={"liked": 0.6, "disliked": -0.6}, votes=6)
        stories = prerank(
            [candidate("same", source="disliked"), candidate("same", source="liked")],
            affinity=affinity,
            now=NOW,
        )
        assert stories[0].candidate.source_slug == "liked"
        assert "feedback" in stories[0].signals and stories[1].signals["feedback"] < 0

    def test_no_votes_means_no_signal_at_all(self):
        _score, _m, parts = score_parts(
            candidate(), profile=None, declared_topics=set(), now=NOW, affinity=Affinity()
        )
        assert "feedback" not in parts

    def test_taste_cannot_outweigh_a_dependency_release(self):
        # Even a fully disliked source loses at most W_FEEDBACK, less than one matched dependency.
        assert W_FEEDBACK < W_DEPENDENCY

    def test_scores_without_an_affinity_are_unchanged(self):
        c = candidate()
        base = score_parts(c, profile=None, declared_topics=set(), now=NOW)
        assert base == score_parts(c, profile=None, declared_topics=set(), now=NOW, affinity=None)


class TestRerankPrompt:
    def test_recent_thumbs_are_shown_to_the_model(self):
        block = examples_block(
            Affinity(liked=["Qdrant 1.13 released"], disliked=["Funding round"], votes=2)
        )
        assert "liked: Qdrant 1.13 released" in block and "disliked: Funding round" in block

    def test_no_examples_adds_nothing(self):
        assert examples_block(None) == "" and examples_block(Affinity()) == ""

    def test_the_prompt_carries_the_examples(self):
        class Capture:
            prompt = ""

            def complete(self, **kw):
                Capture.prompt = kw["prompt"]
                raise RuntimeError("stop here")

        stories = prerank([candidate()], now=NOW)
        rerank(
            stories,
            Capture(),
            affinity=Affinity(liked=["Liked headline"], disliked=["Disliked headline"], votes=2),
        )
        assert "liked: Liked headline" in Capture.prompt


class TestLoading:
    def test_a_database_error_degrades_to_no_effect(self):
        class Broken:
            def execute(self, *_a, **_k):
                raise RuntimeError("db down")

        assert load_affinity(Broken(), uuid.uuid4()).is_empty


def test_feedback_model_is_what_the_loader_reads():
    assert {"segment_id", "vote", "user_id"} <= set(Feedback.__table__.columns.keys())
