"""Freshness, repeat-avoidance and the ranking rules that depend on them (unit)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.pipeline.candidates import (
    DEFAULT_FRESH_HOURS,
    MAX_FRESH_HOURS,
    freshness_window_hours,
    is_aired,
    normalize_url,
)
from voicebrief.pipeline.filtering import FIRST_SEEN_GRACE_HOURS, item_freshness
from voicebrief.pipeline.ranking import (
    RankCandidate,
    RankedStory,
    heuristic_score,
    match_dependencies,
    mentions_language,
    recency_score,
    rerank,
)

NOW = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


def cand(title: str = "Some story", **over) -> RankCandidate:
    base = {
        "cluster_id": uuid.uuid4(),
        "title": title,
        "summary": "",
        "url": "https://example.com/a",
        "trust_weight": 0.7,
        "cluster_size": 1,
    }
    return RankCandidate(**{**base, **over})


class TestItemFreshness:
    def test_ordinary_news_keeps_its_publication_time(self):
        published = NOW - timedelta(hours=2)
        assert item_freshness(published, NOW - timedelta(hours=1)) == published

    def test_old_upstream_timestamp_is_capped_at_first_sight(self):
        # A trending repo created three weeks ago, first seen now, is news now.
        published = NOW - timedelta(days=21)
        fresh = item_freshness(published, NOW)
        assert fresh == NOW - timedelta(hours=FIRST_SEEN_GRACE_HOURS)

    def test_missing_first_seen_falls_back_to_publication(self):
        published = NOW - timedelta(hours=5)
        assert item_freshness(published, None) == published

    def test_naive_datetimes_are_treated_as_utc(self):
        naive = datetime(2026, 10, 30, 1, 0)
        assert item_freshness(naive, None).tzinfo is not None


class TestFreshnessWindow:
    def test_defaults_when_there_is_no_previous_episode(self):
        assert freshness_window_hours(None, now=NOW) == DEFAULT_FRESH_HOURS

    def test_recent_episode_does_not_shrink_the_window(self):
        assert freshness_window_hours(NOW - timedelta(hours=3), now=NOW) == DEFAULT_FRESH_HOURS

    def test_a_missed_weekend_widens_the_window_to_cover_the_gap(self):
        window = freshness_window_hours(NOW - timedelta(days=3), now=NOW)
        assert window > 72

    def test_window_is_capped_at_a_week(self):
        assert freshness_window_hours(NOW - timedelta(days=60), now=NOW) == MAX_FRESH_HOURS


class TestAired:
    def test_normalize_ignores_scheme_www_fragment_and_trailing_slash(self):
        assert normalize_url("HTTPS://www.Example.com/post/#top") == "example.com/post"
        assert normalize_url("http://example.com/post") == "example.com/post"

    def test_story_with_its_only_url_aired_is_aired(self):
        assert is_aired(["https://a.com/x"], {"a.com/x"})

    def test_matching_survives_formatting_differences(self):
        assert is_aired(["https://www.a.com/x/"], {"a.com/x"})

    def test_developing_story_with_one_aired_item_among_many_is_not(self):
        urls = ["https://a.com/1", "https://b.com/2", "https://c.com/3"]
        assert not is_aired(urls, {"a.com/1"})

    def test_nothing_aired_means_nothing_is_excluded(self):
        assert not is_aired(["https://a.com/x"], set())
        assert not is_aired([], {"a.com/x"})


class TestRecencyInRanking:
    def test_newer_beats_older_all_else_equal(self):
        new, _ = heuristic_score(
            cand(freshness=NOW - timedelta(hours=2)), profile=None, declared_topics=set(), now=NOW
        )
        old, _ = heuristic_score(
            cand(freshness=NOW - timedelta(days=6)), profile=None, declared_topics=set(), now=NOW
        )
        assert new > old

    def test_todays_single_item_beats_a_stale_cluster(self):
        today, _ = heuristic_score(
            cand(cluster_size=1, freshness=NOW - timedelta(hours=1)),
            profile=None,
            declared_topics=set(),
            now=NOW,
        )
        stale, _ = heuristic_score(
            cand(cluster_size=4, freshness=NOW - timedelta(days=7)),
            profile=None,
            declared_topics=set(),
            now=NOW,
        )
        assert today > stale

    def test_unknown_freshness_adds_nothing(self):
        assert recency_score(None, now=NOW) == 0.0
        with_none, _ = heuristic_score(cand(), profile=None, declared_topics=set(), now=NOW)
        without = heuristic_score(cand(), profile=None, declared_topics=set())[0]
        assert with_none == pytest.approx(without, abs=0.5)


class TestLanguageMatching:
    @pytest.mark.parametrize(
        ("text", "language", "expected"),
        [
            ("Python 3.14 ships a JIT", "python", True),
            ("Google announces a new model", "go", False),
            ("A decade ago we shipped", "go", False),
            ("Why we rewrote our service in Golang", "go", True),
            ("Release notes for version c", "c", False),
            ("TypeScript 6 is here", "typescript", True),
            ("Pythonic style guide", "python", False),
        ],
    )
    def test_whole_words_only(self, text, language, expected):
        assert mentions_language(text, language) is expected


class TestAmbiguousDependencies:
    @pytest.fixture
    def profile(self) -> StackProfileData:
        return StackProfileData(
            login="dev",
            dependencies={
                "next": {"repos": ["dev/site"], "ecosystem": "npm"},
                "rich": {"repos": ["dev/cli"], "ecosystem": "pypi"},
                "langgraph": {"repos": ["dev/agent"], "ecosystem": "pypi"},
            },
        )

    def test_plain_english_does_not_match(self):
        text = "The next step for rich media is here"
        assert match_dependencies(text, None) == {}

    def test_common_word_package_ignores_prose(self, profile):
        assert match_dependencies("What comes next for rich text editors", profile) == {}

    def test_common_word_package_matches_when_written_like_a_package(self, profile):
        assert "next" in match_dependencies("vercel/next 16 released", profile)
        assert "next" in match_dependencies("Upgrading to next.js today", profile)

    def test_distinctive_names_still_match_in_prose(self, profile):
        assert "langgraph" in match_dependencies("LangGraph adds durable execution", profile)


class _StubClient:
    """Returns fixed rerank scores for the first N stories."""

    def __init__(self, scores: dict[int, int]) -> None:
        self.scores = scores

    def complete(self, **_):
        class _Completion:
            text = ""
            parsed = None

        completion = _Completion()
        completion.parsed = {
            "rankings": [{"index": i, "score": s, "reason": "r"} for i, s in self.scores.items()]
        }
        return completion


def story(title: str, heuristic: float) -> RankedStory:
    return RankedStory(candidate=cand(title), score=heuristic, heuristic_score=heuristic)


class TestRerankScale:
    def test_unreviewed_tail_never_beats_a_story_the_model_judged(self):
        # The model scores the head low; a tail story with a big raw heuristic used to
        # leapfrog it because the two scores lived on different scales.
        stories = [story("head-a", 6.0), story("head-b", 5.0), story("tail", 8.0)]
        ranked = rerank(stories, _StubClient({0: 3, 1: 9}), limit=2)
        order = [s.candidate.title for s in ranked]
        assert order.index("head-b") < order.index("tail")
        # And it still sits where its own blended score puts it relative to head-a.
        tail = next(s for s in ranked if s.candidate.title == "tail")
        assert tail.score == pytest.approx(0.4 * 8.0 + 0.6 * 3)


class TestThinStories:
    def test_a_story_with_no_source_text_ranks_below_a_substantive_one(self):
        thin, _ = heuristic_score(
            cand(substance_chars=60), profile=None, declared_topics=set(), now=NOW
        )
        rich, _ = heuristic_score(
            cand(substance_chars=1500), profile=None, declared_topics=set(), now=NOW
        )
        assert rich > thin

    def test_unknown_substance_is_not_penalised(self):
        unknown, _ = heuristic_score(cand(), profile=None, declared_topics=set(), now=NOW)
        rich, _ = heuristic_score(
            cand(substance_chars=1500), profile=None, declared_topics=set(), now=NOW
        )
        assert unknown == pytest.approx(rich)


class TestCommonWordDependencies:
    """A frozen requirements file drags in packages named `path`, `http`, `catalogue`.
    Matching those in prose made an unrelated repo look like it touched the user's stack
    and added three points to its score."""

    @pytest.fixture
    def profile(self) -> StackProfileData:
        deps = ["path", "http", "catalogue", "build", "kokoro", "langgraph", "react", "pandas"]
        return StackProfileData(
            login="dev", dependencies={d: {"repos": ["dev/app"], "ecosystem": "x"} for d in deps}
        )

    @pytest.mark.parametrize("word", ["path", "http", "catalogue", "build"])
    def test_ordinary_words_do_not_match_in_prose(self, profile, word):
        text = f"A new open source project proposes a different {word} for agents"
        assert word not in match_dependencies(text, profile)

    @pytest.mark.parametrize("name", ["kokoro", "langgraph", "pandas"])
    def test_distinctive_names_still_match_in_prose(self, profile, name):
        assert name in match_dependencies(f"{name.title()} 2.0 ships a faster core", profile)

    def test_react_is_treated_as_the_library_it_usually_is(self, profile):
        assert "react" in match_dependencies("React 19 changes how effects run", profile)

    def test_a_common_word_still_matches_when_written_like_a_package(self, profile):
        assert "path" in match_dependencies("see the path.py module in the repo", profile)

    def test_the_generated_word_list_is_present_and_sensible(self):
        from voicebrief.pipeline.ranking import COMMON_WORDS

        assert len(COMMON_WORDS) > 5000
        assert {"path", "http", "next"} <= COMMON_WORDS
        assert not ({"kokoro", "langgraph", "fastapi", "pydantic"} & COMMON_WORDS)
