from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from voicebrief.sources.base import SourceConfig
from voicebrief.sources.github import GitHubReleasesAdapter, GitHubSearchAdapter

SEARCH_URL = "https://api.github.com/search/repositories"
REPOS_URL = "https://api.github.com/repos"


def repo(repo_id: int, name: str, stars: int = 200, topics: list[str] | None = None) -> dict:
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    return {
        "id": repo_id,
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "description": f"{name} description",
        "owner": {"login": name.split("/")[0]},
        "created_at": recent,
        "pushed_at": recent,
        "stargazers_count": stars,
        "forks_count": 10,
        "language": "Python",
        "topics": topics or ["llm"],
    }


def release(tag: str, *, prerelease: bool = False, draft: bool = False, days_ago: int = 1) -> dict:
    published = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "tag_name": tag,
        "name": f"Release {tag}",
        "html_url": f"https://github.com/o/r/releases/tag/{tag}",
        "body": "### Highlights\nDurable execution is now the default.",
        "published_at": published.replace("+00:00", "Z"),
        "prerelease": prerelease,
        "draft": draft,
    }


@pytest.fixture
def since() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=30)


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────
def search_adapter(topics: list[str]) -> GitHubSearchAdapter:
    return GitHubSearchAdapter(
        SourceConfig(
            slug="gh-search",
            name="GitHub Search",
            endpoint=SEARCH_URL,
            default_topics=["agentic-ai"],
            config={"topics": topics, "min_stars": 50, "created_within_days": 30},
        )
    )


class TestGitHubSearch:
    @respx.mock
    async def test_issues_one_request_per_topic(self, since):
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"items": [repo(1, "a/b")]})
        )
        async with httpx.AsyncClient() as client:
            await search_adapter(["llm", "agents", "rag"]).fetch(client, since)
        assert route.call_count == 3

        queries = [c.request.url.params["q"] for c in route.calls]
        assert {"topic:llm", "topic:agents", "topic:rag"} == {q.split(" ")[0] for q in queries}
        assert all("OR" not in q for q in queries), "OR between qualifiers is a 422"

    @respx.mock
    async def test_repo_matching_two_topics_yields_one_item(self, since):
        """The same repo returned for `llm` and `agents` must merge, not duplicate."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"items": [repo(1, "a/b")]})
        )
        async with httpx.AsyncClient() as client:
            items = await search_adapter(["llm", "agents"]).fetch(client, since)
        assert len(items) == 1

    @respx.mock
    async def test_partial_topic_failure_still_returns_results(self, since):
        """One bad topic query must not cost the other topics' results."""
        responses = [
            httpx.Response(422, json={"message": "Validation Failed"}),
            httpx.Response(200, json={"items": [repo(2, "c/d")]}),
        ]
        respx.get(SEARCH_URL).mock(side_effect=responses)
        async with httpx.AsyncClient() as client:
            items = await search_adapter(["bad", "good"]).fetch(client, since)
        assert [i.title for i in items] == ["c/d"]

    @respx.mock
    async def test_total_failure_raises_so_the_run_is_recorded_as_failed(self, since):
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(422, json={}))
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPError, match="all 2 GitHub search queries failed"):
                await search_adapter(["a", "b"]).fetch(client, since)

    @respx.mock
    async def test_rate_limit_is_reported_with_an_actionable_message(self, since):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={})
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPError):
                await search_adapter(["llm"]).fetch(client, since)

    @respx.mock
    async def test_stars_normalize_into_engagement(self, since):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json={"items": [repo(1, "a/b", stars=2500), repo(2, "c/d", stars=99_000)]}
            )
        )
        async with httpx.AsyncClient() as client:
            items = {i.title: i.engagement for i in await search_adapter(["llm"]).fetch(client, since)}
        assert items["a/b"] == pytest.approx(0.5)
        assert items["c/d"] == 1.0

    @respx.mock
    async def test_language_becomes_a_topic(self, since):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"items": [repo(1, "a/b")]})
        )
        async with httpx.AsyncClient() as client:
            items = await search_adapter(["llm"]).fetch(client, since)
        assert items[0].topics == ["agentic-ai", "python"]

    @respx.mock
    async def test_stale_repo_outside_window_is_dropped(self, since):
        stale = repo(1, "a/b")
        stale["pushed_at"] = "2020-01-01T00:00:00Z"
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json={"items": [stale]}))
        async with httpx.AsyncClient() as client:
            assert await search_adapter(["llm"]).fetch(client, since) == []


# ─────────────────────────────────────────────────────────────────────────────
# Releases
# ─────────────────────────────────────────────────────────────────────────────
def releases_adapter(repos: list[str]) -> GitHubReleasesAdapter:
    return GitHubReleasesAdapter(
        SourceConfig(
            slug="gh-releases",
            name="GitHub Releases",
            endpoint=REPOS_URL,
            default_topics=["tooling"],
            config={"repos": repos},
        )
    )


class TestGitHubReleases:
    @respx.mock
    async def test_maps_release_fields(self, since):
        respx.get(f"{REPOS_URL}/o/r/releases").mock(
            return_value=httpx.Response(200, json=[release("v1.2.0")])
        )
        async with httpx.AsyncClient() as client:
            items = await releases_adapter(["o/r"]).fetch(client, since)

        assert len(items) == 1
        item = items[0]
        assert item.external_id == "o/r@v1.2.0"
        assert item.title == "o/r Release v1.2.0"
        assert "releases" in item.topics
        assert item.raw["repo"] == "o/r"

    @respx.mock
    async def test_drafts_are_excluded(self, since):
        respx.get(f"{REPOS_URL}/o/r/releases").mock(
            return_value=httpx.Response(200, json=[release("v1", draft=True), release("v2")])
        )
        async with httpx.AsyncClient() as client:
            items = await releases_adapter(["o/r"]).fetch(client, since)
        assert [i.raw["tag"] for i in items] == ["v2"]

    @respx.mock
    async def test_prereleases_score_lower_than_stable(self, since):
        respx.get(f"{REPOS_URL}/o/r/releases").mock(
            return_value=httpx.Response(
                200, json=[release("v1"), release("v2-rc1", prerelease=True)]
            )
        )
        async with httpx.AsyncClient() as client:
            items = {i.raw["tag"]: i.engagement for i in await releases_adapter(["o/r"]).fetch(client, since)}
        assert items["v1"] > items["v2-rc1"]

    @respx.mock
    async def test_one_dead_repo_does_not_lose_the_others(self, since):
        """Observed live: unauthenticated rate limits kill some repos mid-run. The
        remaining repos must still produce items."""
        respx.get(f"{REPOS_URL}/dead/repo/releases").mock(return_value=httpx.Response(404))
        respx.get(f"{REPOS_URL}/live/repo/releases").mock(
            return_value=httpx.Response(200, json=[release("v9")])
        )
        async with httpx.AsyncClient() as client:
            items = await releases_adapter(["dead/repo", "live/repo"]).fetch(client, since)
        assert [i.raw["tag"] for i in items] == ["v9"]

    @respx.mock
    async def test_release_older_than_window_is_dropped(self, since):
        respx.get(f"{REPOS_URL}/o/r/releases").mock(
            return_value=httpx.Response(
                200, json=[release("old", days_ago=400), release("new", days_ago=2)]
            )
        )
        async with httpx.AsyncClient() as client:
            items = await releases_adapter(["o/r"]).fetch(client, since)
        assert [i.raw["tag"] for i in items] == ["new"]
