"""A brief is as long as the news deserves: only impact earns a place."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from voicebrief.graphs.brief import (
    TARGET_MINUTES,
    WORDS_PER_MINUTE,
    BriefGraph,
)
from voicebrief.pipeline.ranking import (
    IMPACT_FLOOR,
    RankCandidate,
    RankedStory,
    select_by_impact,
)


def story(title, *, impact=None, buzz=None, source="rss", chars=2000, stack=False):
    candidate = RankCandidate(
        cluster_id=uuid.uuid4(),
        title=title,
        summary=title,
        url=f"https://example.com/{uuid.uuid4()}",
        source_slug=source,
        substance_chars=chars,
    )
    deps = {"langgraph": ["app"]} if stack else {}
    return RankedStory(
        candidate=candidate,
        score=(impact or 0) * 0.6,
        heuristic_score=1.0,
        llm_score=impact,
        buzz=buzz,
        matched_dependencies=deps,
    )


class TestWhoGetsIn:
    def test_only_stories_that_clear_the_bar_are_told(self):
        stories = [story(f"Alpha{i} beta{i}", impact=9 - i) for i in range(8)]
        picked = select_by_impact(stories)
        assert {s.llm_score for s in picked} == {
            s.llm_score for s in stories if s.llm_score >= IMPACT_FLOOR
        }
        assert all((s.llm_score or 0) >= IMPACT_FLOOR for s in picked)

    def test_a_quiet_day_is_short_and_a_busy_one_is_long(self):
        quiet = [story(f"Minor{i} tweak{i}", impact=3) for i in range(10)] + [
            story("Top thing here", impact=5)
        ]
        busy = [story(f"Major{i} change{i} word{i}", impact=8) for i in range(9)]
        assert len(select_by_impact(quiet)) == 1  # never empty, never padded
        assert len(select_by_impact(busy)) == 9

    def test_a_small_release_of_a_library_they_use_stays_out(self):
        minor = story("pkg 1.2.3 adds an option", impact=3, source="github-releases-x", stack=True)
        major = story("Streaming rewritten end to end", impact=9)
        picked = select_by_impact([minor, major])
        assert major in picked and minor not in picked

    def test_the_days_big_news_leads_even_when_it_is_not_about_them(self):
        mine = story("Their library ships a big change", impact=9, stack=True)
        big = story("A leading lab launches a new flagship model", impact=6, buzz=10)
        picked = select_by_impact([mine, big])
        assert picked[0] is big and picked[0].role == "headline"
        assert picked[1] is mine

    def test_a_rising_repository_is_always_featured_after_the_news(self):
        news = story("Big model news", impact=9, buzz=9)
        repo = story("someone/video-tool", impact=6, source="github-trending-open")
        picked = select_by_impact([news, repo])
        assert [s.role for s in picked] == ["headline", "spotlight"]

    def test_a_repository_with_nothing_to_say_is_not_featured(self):
        bare = story("someone/empty", impact=6, source="github-trending-open", chars=50)
        assert bare not in select_by_impact([story("Big model news", impact=9), bare])

    def test_without_a_model_score_the_best_few_are_used(self):
        stories = [story(f"Item{i} thing{i}") for i in range(6)]
        for i, s in enumerate(stories):
            s.score = 10 - i
        assert len(select_by_impact(stories)) == 3


class TestHowLongEachRuns:
    def graph(self):
        return BriefGraph(SimpleNamespace(), target_minutes=TARGET_MINUTES)

    def test_big_news_gets_more_words_than_a_spotlight(self):
        big = story("Headline thing", impact=9, buzz=9)
        big.role = "headline"
        repo = story("a/b", impact=6, source="github-trending-open")
        repo.role = "spotlight"
        budgets = self.graph()._word_budgets([big, repo])
        assert budgets[0] > budgets[1] > 0

    def test_a_full_day_never_runs_past_the_ceiling(self):
        stories = [story(f"S{i} thing{i}", impact=9) for i in range(14)]
        for s in stories:
            s.role = "relevant"
        total = sum(self.graph()._word_budgets(stories))
        assert total <= TARGET_MINUTES * WORDS_PER_MINUTE
