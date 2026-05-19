from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.pipeline.ranking import (
    RankCandidate,
    heuristic_score,
    match_dependencies,
    prerank,
    select_for_episode,
)


@pytest.fixture
def profile() -> StackProfileData:
    return StackProfileData(
        login="dev",
        languages={"python": 6, "dart": 3},
        dependencies={
            "langgraph": {"repos": ["dev/ContextPilot", "dev/PlacementPilot"], "ecosystem": "pypi"},
            "qdrant-client": {"repos": ["dev/ContextPilot"], "ecosystem": "pypi"},
            "fastapi": {"repos": ["dev/ContextPilot"], "ecosystem": "pypi"},
            "dio": {"repos": ["dev/TripMate"], "ecosystem": "pub"},
        },
        repos=["dev/ContextPilot", "dev/PlacementPilot", "dev/TripMate"],
        starred_topics=["agents", "rag"],
        built_at=datetime.now(timezone.utc),
    )


def cand(title: str, summary: str = "", **over) -> RankCandidate:
    base = dict(
        cluster_id=uuid.uuid4(),
        title=title,
        summary=summary,
        url="https://example.com/a",
        topics=["agentic-ai"],
        engagement=0.3,
        trust_weight=0.7,
        cluster_size=1,
        source_slug="test",
    )
    return RankCandidate(**{**base, **over})


class TestDependencyMatching:
    def test_finds_a_dependency_by_name(self, profile):
        matched = match_dependencies("LangGraph 0.4 ships durable execution", profile)
        assert "langgraph" in matched

    def test_reports_which_repos_use_it(self, profile):
        matched = match_dependencies("langgraph released", profile)
        assert matched["langgraph"] == ["dev/ContextPilot", "dev/PlacementPilot"]

    def test_matching_is_case_insensitive(self, profile):
        assert "fastapi" in match_dependencies("FastAPI 0.116 is out", profile)

    def test_matches_inside_a_repo_path(self, profile):
        assert "langgraph" in match_dependencies("langchain-ai/langgraph v0.4", profile)

    def test_does_not_match_a_substring_of_another_word(self, profile):
        """A naive `in` check would match 'dio' inside 'audio', 'radio', 'studio' and
        make the strongest signal in the system unusable."""
        assert match_dependencies("Improved audio studio for radio", profile) == {}

    def test_very_short_names_are_skipped(self):
        """Documented limitation, not an oversight.

        Names under four characters are ignored because they collide with ordinary
        English constantly — `go` inside "let go", `dio` inside "audio"/"studio",
        `six` inside "sixty". Matching them would flood the highest-weighted signal
        in the ranker with false positives.

        The cost is real: genuine short packages (`dio`, `six`, `ply`) will never
        raise a story. Precision is the right side to err on here, because a false
        dependency match produces a confidently wrong claim in the audio — "you use
        this in ContextPilot" when they do not — which is far worse than a miss.
        """
        short = StackProfileData(
            login="d",
            dependencies={"go": {"repos": ["d/x"]}, "dio": {"repos": ["d/y"]}},
        )
        assert match_dependencies("Let go of the past", short) == {}
        assert match_dependencies("New audio studio for radio", short) == {}

    def test_no_profile_matches_nothing(self):
        assert match_dependencies("langgraph", None) == {}

    def test_empty_profile_matches_nothing(self):
        assert match_dependencies("langgraph", StackProfileData(login="d")) == {}


class TestHeuristicScore:
    def test_dependency_hit_outranks_a_popular_unrelated_story(self, profile):
        personal, _ = heuristic_score(
            cand("langgraph 0.4 ships durable execution", engagement=0.0, trust_weight=0.5),
            profile=profile,
            declared_topics=set(),
        )
        popular, _ = heuristic_score(
            cand("A viral story about nothing you use", engagement=1.0, trust_weight=1.0),
            profile=profile,
            declared_topics=set(),
        )
        assert personal > popular, "the whole product is that this ordering holds"

    def test_more_dependency_hits_score_higher(self, profile):
        one, _ = heuristic_score(cand("langgraph update"), profile=profile, declared_topics=set())
        two, _ = heuristic_score(
            cand("langgraph and qdrant-client integration"),
            profile=profile,
            declared_topics=set(),
        )
        assert two > one

    def test_declared_topics_contribute(self, profile):
        with_topic, _ = heuristic_score(
            cand("x", topics=["agentic-ai"]), profile=profile, declared_topics={"agentic-ai"}
        )
        without, _ = heuristic_score(
            cand("x", topics=["gardening"]), profile=profile, declared_topics={"agentic-ai"}
        )
        assert with_topic > without

    def test_cluster_size_contributes(self, profile):
        big, _ = heuristic_score(cand("x", cluster_size=4), profile=profile, declared_topics=set())
        small, _ = heuristic_score(cand("x", cluster_size=1), profile=profile, declared_topics=set())
        assert big > small

    def test_works_without_a_profile(self):
        score, matched = heuristic_score(cand("anything"), profile=None, declared_topics=set())
        assert score >= 0 and matched == {}


class TestPrerank:
    def test_orders_best_first(self, profile):
        stories = prerank(
            [cand("unrelated news"), cand("langgraph 0.4 released"), cand("more unrelated")],
            profile=profile,
        )
        assert "langgraph" in stories[0].candidate.title

    def test_flags_personal_stories(self, profile):
        stories = prerank([cand("langgraph released"), cand("unrelated")], profile=profile)
        assert stories[0].is_personal
        assert not stories[-1].is_personal

    def test_empty_input(self, profile):
        assert prerank([], profile=profile) == []


class TestSelection:
    def test_respects_the_episode_cap(self, profile):
        stories = prerank([cand(f"story number {i}") for i in range(30)], profile=profile)
        assert len(select_for_episode(stories, max_stories=10)) == 10

    def test_guarantees_personal_stories_make_the_cut(self, profile):
        """Without the floor, the ranker drifts to whatever is objectively biggest and
        the brief stops being personal — which is the entire product."""
        loud = [
            cand(f"huge industry story {i}", engagement=1.0, trust_weight=1.0, cluster_size=5)
            for i in range(12)
        ]
        quiet = [
            cand("a small fastapi patch lands", engagement=0.0, trust_weight=0.3),
            cand("a small qdrant-client fix", engagement=0.0, trust_weight=0.3),
        ]
        selected = select_for_episode(
            prerank([*loud, *quiet], profile=profile), max_stories=10, min_personal=2
        )
        assert sum(1 for s in selected if s.is_personal) >= 2

    def test_does_not_grow_the_episode_to_fit_personal_stories(self, profile):
        loud = [cand(f"big story {i}", engagement=1.0, cluster_size=5) for i in range(12)]
        quiet = [cand("tiny langgraph patch"), cand("tiny fastapi patch")]
        selected = select_for_episode(
            prerank([*loud, *quiet], profile=profile), max_stories=8, min_personal=2
        )
        assert len(selected) == 8

    def test_no_personal_stories_available_is_not_an_error(self):
        stories = prerank([cand(f"story {i}") for i in range(5)], profile=None)
        assert len(select_for_episode(stories, max_stories=3, min_personal=2)) == 3

    def test_output_is_ordered_by_score(self, profile):
        stories = prerank([cand(f"story number {i}") for i in range(20)], profile=profile)
        selected = select_for_episode(stories, max_stories=10)
        assert [s.score for s in selected] == sorted(
            (s.score for s in selected), reverse=True
        )
