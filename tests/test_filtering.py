from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from voicebrief.pipeline.filtering import (
    RECENCY_HALF_LIFE_HOURS,
    is_noise,
    prefilter,
    score_item,
)

NOW = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
SRC_A = uuid.uuid4()
SRC_B = uuid.uuid4()


class FakeItem:
    """Stands in for the Item row; the filter only touches these five fields."""

    def __init__(
        self,
        title="A reasonable article title",
        *,
        hours_ago=1.0,
        engagement=0.0,
        topics=None,
        source_id=SRC_A,
        url="https://example.com/a",
    ):
        self.title = title
        self.published_at = NOW - timedelta(hours=hours_ago)
        self.engagement = engagement
        self.topics = topics if topics is not None else []
        self.source_id = source_id
        self.url = url


class TestNoiseDetection:
    @pytest.mark.parametrize(
        "title",
        [
            "Ask HN: what are you working on?",
            "Who is hiring? (June 2026)",
            "Monthly thread for beginners",
            "Daily thread",
        ],
    )
    def test_recurring_community_threads_are_noise(self, title):
        assert is_noise(FakeItem(title))

    def test_very_short_titles_are_noise(self):
        assert is_noise(FakeItem("v2.1"))

    def test_missing_url_is_noise(self):
        assert is_noise(FakeItem(url=""))

    @pytest.mark.parametrize(
        "title",
        [
            "Show HN: I built a local-first vector store",
            "LangGraph 0.4 ships durable execution",
            "Ask me about distributed systems design",
        ],
    )
    def test_legitimate_titles_survive(self, title):
        """The blocklist is deliberately narrow — over-filtering destroys recall
        silently, and 'Show HN' posts are frequently the best item of the day."""
        assert not is_noise(FakeItem(title))


class TestRecency:
    def test_fresh_item_scores_near_one(self):
        s = score_item(FakeItem(hours_ago=0), trust_weight=0.5, user_topics=set(), now=NOW)
        assert s.reasons["recency"] == pytest.approx(1.0)

    def test_half_life_halves_the_score(self):
        s = score_item(
            FakeItem(hours_ago=RECENCY_HALF_LIFE_HOURS),
            trust_weight=0.5,
            user_topics=set(),
            now=NOW,
        )
        assert s.reasons["recency"] == pytest.approx(0.5)

    def test_recency_decays_monotonically(self):
        scores = [
            score_item(FakeItem(hours_ago=h), trust_weight=0.5, user_topics=set(), now=NOW)
            .reasons["recency"]
            for h in (1, 12, 48, 168)
        ]
        assert scores == sorted(scores, reverse=True)

    def test_future_dated_item_does_not_exceed_one(self):
        """Feeds occasionally publish with a clock ahead of ours."""
        s = score_item(FakeItem(hours_ago=-5), trust_weight=0.5, user_topics=set(), now=NOW)
        assert s.reasons["recency"] <= 1.0

    def test_naive_timestamp_is_treated_as_utc(self):
        item = FakeItem()
        item.published_at = item.published_at.replace(tzinfo=None)
        s = score_item(item, trust_weight=0.5, user_topics=set(), now=NOW)
        assert 0.0 < s.reasons["recency"] <= 1.0


class TestTopicMatch:
    def test_no_declared_topics_is_neutral_not_zero(self):
        s = score_item(FakeItem(topics=["x"]), trust_weight=0.5, user_topics=set(), now=NOW)
        assert s.reasons["topic"] == 0.5

    def test_overlap_scores_above_no_overlap(self):
        hit = score_item(
            FakeItem(topics=["agentic-ai"]),
            trust_weight=0.5,
            user_topics={"agentic-ai"},
            now=NOW,
        )
        miss = score_item(
            FakeItem(topics=["gardening"]),
            trust_weight=0.5,
            user_topics={"agentic-ai"},
            now=NOW,
        )
        assert hit.reasons["topic"] > miss.reasons["topic"] == 0.0

    def test_matching_is_case_insensitive(self):
        s = score_item(
            FakeItem(topics=["Agentic-AI"]),
            trust_weight=0.5,
            user_topics={"agentic-ai"},
            now=NOW,
        )
        assert s.reasons["topic"] > 0

    def test_extra_item_tags_are_not_penalised(self):
        """A Hub model carrying 12 tags shouldn't score below one carrying 1 tag
        when both match the single thing the user asked for."""
        many = score_item(
            FakeItem(topics=["agentic-ai", *[f"tag{i}" for i in range(11)]]),
            trust_weight=0.5,
            user_topics={"agentic-ai"},
            now=NOW,
        )
        few = score_item(
            FakeItem(topics=["agentic-ai"]), trust_weight=0.5, user_topics={"agentic-ai"}, now=NOW
        )
        assert many.reasons["topic"] == few.reasons["topic"]


class TestPrefilter:
    def test_truncates_to_keep(self):
        items = [FakeItem(f"Article number {i} here") for i in range(500)]
        result = prefilter(items, trust_by_source={SRC_A: 0.5}, keep=300, now=NOW)
        assert len(result) == 300

    def test_results_are_sorted_best_first(self):
        items = [FakeItem(f"Article number {i} here", hours_ago=i) for i in range(50)]
        result = prefilter(items, trust_by_source={SRC_A: 0.5}, keep=50, now=NOW)
        assert [r.score for r in result] == sorted((r.score for r in result), reverse=True)

    def test_high_trust_release_outranks_a_flood_of_low_trust_items(self):
        """The failure this guards against: a thousand preprints burying the one
        release that affects the user's lockfile."""
        flood = [
            FakeItem(f"Preprint number {i} on something", hours_ago=2, source_id=SRC_B)
            for i in range(400)
        ]
        release = FakeItem(
            "langgraph v0.4.0 ships durable execution", hours_ago=6, source_id=SRC_A, engagement=0.6
        )
        result = prefilter(
            [*flood, release],
            trust_by_source={SRC_A: 0.95, SRC_B: 0.4},
            keep=50,
            now=NOW,
        )
        assert any("langgraph" in r.item.title for r in result[:5])

    def test_noise_is_removed_before_scoring(self):
        items = [FakeItem("Ask HN: anything?"), FakeItem("A genuine article title")]
        result = prefilter(items, trust_by_source={SRC_A: 0.5}, keep=10, now=NOW)
        assert [r.item.title for r in result] == ["A genuine article title"]

    def test_unknown_source_gets_a_neutral_trust_prior(self):
        result = prefilter([FakeItem()], trust_by_source={}, keep=10, now=NOW)
        assert result[0].reasons["trust"] == 0.5

    def test_min_score_floor_is_applied(self):
        items = [FakeItem("A very old article title", hours_ago=1000)]
        assert prefilter(items, trust_by_source={SRC_A: 0.1}, keep=10, min_score=0.5, now=NOW) == []

    def test_empty_input_is_safe(self):
        assert prefilter([], trust_by_source={}, keep=10, now=NOW) == []

    def test_reasons_are_reported_for_explainability(self):
        result = prefilter([FakeItem()], trust_by_source={SRC_A: 0.8}, keep=1, now=NOW)
        assert set(result[0].reasons) == {"recency", "trust", "engagement", "topic"}


class TestIdentifierTitles:
    """Regression: the naive word-count rule deleted every GitHub and Hugging Face
    item on a real 560-item crawl — 59 items, three whole source types — because
    'langchain-ai/langgraph' is a single word."""

    @pytest.mark.parametrize(
        "title",
        [
            "langchain-ai/langgraph",
            "vllm-project/vllm v0.29.0",
            "nvidia/PhysicalAI-Autonomous-Vehicles (dataset)",
            "org/model-name (model)",
        ],
    )
    def test_repo_and_model_identifiers_are_not_noise(self, title):
        assert not is_noise(FakeItem(title))

    def test_a_stub_identifier_is_still_noise(self):
        assert is_noise(FakeItem("a/b"))

    def test_prose_titles_still_need_three_words(self):
        assert is_noise(FakeItem("Short title"))
        assert not is_noise(FakeItem("A three word title"))

    def test_github_and_hf_items_survive_a_mixed_corpus(self):
        items = [
            FakeItem("langchain-ai/langchain v1.6.2", source_id=SRC_B, engagement=0.6),
            *[FakeItem(f"Some preprint about topic {i}") for i in range(50)],
        ]
        kept = prefilter(items, trust_by_source={SRC_A: 0.8, SRC_B: 0.95}, keep=20, now=NOW)
        assert any("/" in r.item.title for r in kept)


class TestSourceDiversity:
    """Regression: on a real crawl, pure score ordering gave arXiv 235 of 300 slots
    and evicted every GitHub release despite releases carrying higher trust."""

    def test_one_high_volume_source_cannot_monopolise(self):
        flood = [FakeItem(f"Preprint number {i} here", source_id=SRC_A) for i in range(400)]
        others = [FakeItem(f"Release number {i} here", source_id=SRC_B) for i in range(20)]
        kept = prefilter(
            [*flood, *others], trust_by_source={SRC_A: 0.8, SRC_B: 0.7}, keep=100, now=NOW
        )
        from collections import Counter

        counts = Counter(r.item.source_id for r in kept)
        assert counts[SRC_A] <= 100, "quota must bound the dominant source's primary share"
        assert counts[SRC_B] > 0, "the smaller source must not be evicted entirely"

    def test_quota_backfills_rather_than_wasting_slots(self):
        """With only one source available, the cap must not return fewer than `keep`."""
        items = [FakeItem(f"Article number {i} here", source_id=SRC_A) for i in range(200)]
        kept = prefilter(items, trust_by_source={SRC_A: 0.8}, keep=100, now=NOW)
        assert len(kept) == 100

    def test_ratio_of_one_disables_the_quota(self):
        items = [FakeItem(f"Article number {i} here", source_id=SRC_A) for i in range(50)]
        kept = prefilter(
            items, trust_by_source={SRC_A: 0.8}, keep=50, max_per_source_ratio=1.0, now=NOW
        )
        assert len(kept) == 50
