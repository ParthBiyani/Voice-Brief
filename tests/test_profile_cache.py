"""Rate-limited profile builds and the persistent profile cache."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from voicebrief.db.models import User
from voicebrief.db.session import get_engine
from voicebrief.personalization.github_profile import (
    GitHubProfileBuilder,
    RateLimited,
    StackProfileData,
)
from voicebrief.personalization.profile_cache import (
    PROFILE_TTL,
    is_fresh,
    load_profile,
    save_profile,
)
from voicebrief.pipeline.daily import resolve_profile

from .test_github_profile import API, b64, repo

LIMIT_HEADERS = {"x-ratelimit-remaining": "0"}


def good_profile(login="user") -> StackProfileData:
    return StackProfileData(
        login=login,
        languages={"python": 3},
        dependencies={"langgraph": {"repos": ["user/agent"], "ecosystem": "pypi"}},
        repos=["user/agent"],
        starred_topics=["agents"],
        recent_commit_terms=["graph"],
        built_at=datetime.now(UTC),
    )


class TestRateLimitedBuilds:
    @respx.mock
    async def test_a_quota_that_runs_dry_marks_the_profile_incomplete_not_empty_and_fine(self):
        repos = [repo("a"), repo("b")]
        respx.get(f"{API}/users/user/repos").mock(return_value=httpx.Response(200, json=repos))
        respx.get(f"{API}/users/user/starred").mock(return_value=httpx.Response(200, json=[]))
        respx.get(url__regex=rf"{API}/repos/.*").mock(
            return_value=httpx.Response(403, headers=LIMIT_HEADERS, text="API rate limit exceeded")
        )
        profile = await GitHubProfileBuilder().build("user")
        assert profile.complete is False
        assert "rate limit" in profile.incomplete_reason.lower()

    @respx.mock
    async def test_the_hint_to_set_a_token_is_given_when_unauthenticated(self):
        respx.get(f"{API}/users/user/repos").mock(
            return_value=httpx.Response(200, json=[repo("a")])
        )
        respx.get(f"{API}/users/user/starred").mock(return_value=httpx.Response(200, json=[]))
        respx.get(url__regex=rf"{API}/repos/.*").mock(
            return_value=httpx.Response(429, text="slow down")
        )
        builder = GitHubProfileBuilder()
        builder.settings = builder.settings.model_copy(update={"github_token": None})
        profile = await builder.build("user")
        assert "GITHUB_TOKEN" in profile.incomplete_reason

    @respx.mock
    async def test_requests_stop_once_the_quota_is_gone(self):
        repos = [repo(f"r{i}") for i in range(8)]
        respx.get(f"{API}/users/user/repos").mock(return_value=httpx.Response(200, json=repos))
        respx.get(f"{API}/users/user/starred").mock(return_value=httpx.Response(200, json=[]))
        tree = respx.get(url__regex=rf"{API}/repos/.*/git/trees/HEAD").mock(
            return_value=httpx.Response(403, headers=LIMIT_HEADERS, text="rate limit")
        )
        await GitHubProfileBuilder().build("user")
        assert tree.call_count < len(repos), "kept hammering a quota that was already spent"

    @respx.mock
    async def test_a_listing_that_hits_the_limit_raises_a_clear_error(self):
        respx.get(f"{API}/users/user/repos").mock(
            return_value=httpx.Response(403, headers=LIMIT_HEADERS, text="rate limit exceeded")
        )
        with pytest.raises(RateLimited):
            await GitHubProfileBuilder().build("user")

    @respx.mock
    async def test_a_healthy_build_is_complete(self):
        respx.get(f"{API}/users/user/repos").mock(
            return_value=httpx.Response(200, json=[repo("a")])
        )
        respx.get(f"{API}/users/user/starred").mock(return_value=httpx.Response(200, json=[]))
        respx.get(f"{API}/repos/user/a/git/trees/HEAD").mock(
            return_value=httpx.Response(
                200, json={"tree": [{"path": "requirements.txt", "type": "blob"}]}
            )
        )
        respx.get(f"{API}/repos/user/a/contents/requirements.txt").mock(
            return_value=httpx.Response(200, json=b64("langgraph==0.4\n"))
        )
        profile = await GitHubProfileBuilder().build("user")
        assert profile.complete
        assert "langgraph" in profile.dependencies


class TestFreshness:
    def test_a_recent_profile_is_fresh(self):
        assert is_fresh(datetime.now(UTC) - timedelta(days=2))

    def test_a_profile_past_the_ttl_is_stale(self):
        assert not is_fresh(datetime.now(UTC) - PROFILE_TTL - timedelta(hours=1))


@pytest.mark.integration
class TestProfileCache:
    @pytest.fixture
    def session(self):
        with Session(get_engine()) as s:
            s.begin()
            try:
                yield s
            finally:
                s.rollback()

    @pytest.fixture
    def user(self, session):
        u = User(email=f"{uuid.uuid4().hex[:8]}@example.com", github_login="user")
        session.add(u)
        session.flush()
        return u

    def test_a_saved_profile_round_trips(self, session, user):
        save_profile(session, user.id, good_profile())
        loaded = load_profile(session, user.id, login="user")
        assert loaded is not None
        assert loaded[0].dependencies["langgraph"]["repos"] == ["user/agent"]
        assert loaded[0].starred_topics == ["agents"]

    def test_saving_again_replaces_rather_than_duplicates(self, session, user):
        save_profile(session, user.id, good_profile())
        newer = good_profile()
        newer.dependencies = {"fastapi": {"repos": ["user/api"], "ecosystem": "pypi"}}
        save_profile(session, user.id, newer)
        loaded = load_profile(session, user.id, login="user")
        assert set(loaded[0].dependencies) == {"fastapi"}

    def test_no_profile_yet(self, session, user):
        assert load_profile(session, user.id, login="user") is None

    def test_a_fresh_cached_profile_is_used_without_calling_github(
        self, session, user, monkeypatch
    ):
        save_profile(session, user.id, good_profile())

        def must_not_build(*a, **k):
            raise AssertionError("GitHub was called despite a fresh cached profile")

        monkeypatch.setattr(GitHubProfileBuilder, "build", must_not_build)
        profile = resolve_profile(session, user)
        assert "langgraph" in profile.dependencies

    def test_an_incomplete_rebuild_never_replaces_a_good_stale_profile(
        self, session, user, monkeypatch
    ):
        old = good_profile()
        old.built_at = datetime.now(UTC) - PROFILE_TTL - timedelta(days=1)
        save_profile(session, user.id, old)

        async def rate_limited(self, login):
            return StackProfileData(
                login=login, complete=False, incomplete_reason="GitHub rate limit reached"
            )

        monkeypatch.setattr(GitHubProfileBuilder, "build", rate_limited)
        profile = resolve_profile(session, user)
        assert "langgraph" in profile.dependencies, "a rate-limited build wiped the profile"
        assert "langgraph" in load_profile(session, user.id, login="user")[0].dependencies

    def test_a_complete_rebuild_is_saved(self, session, user, monkeypatch):
        async def healthy(self, login):
            return good_profile(login)

        monkeypatch.setattr(GitHubProfileBuilder, "build", healthy)
        resolve_profile(session, user)
        assert load_profile(session, user.id, login="user") is not None

    def test_a_failed_build_falls_back_to_the_cache(self, session, user, monkeypatch):
        old = good_profile()
        old.built_at = datetime.now(UTC) - PROFILE_TTL - timedelta(days=1)
        save_profile(session, user.id, old)

        async def broken(self, login):
            raise httpx.ConnectError("no network")

        monkeypatch.setattr(GitHubProfileBuilder, "build", broken)
        assert "langgraph" in resolve_profile(session, user).dependencies

    def test_no_login_means_no_profile(self, session, user):
        user.github_login = None
        assert resolve_profile(session, user) is None
