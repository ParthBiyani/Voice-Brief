from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from voicebrief.personalization.github_profile import (
    MANIFESTS,
    GitHubProfileBuilder,
    parse_manifest,
)

API = "https://api.github.com"


def b64(text: str) -> dict:
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


def repo(name: str, language: str = "Python", fork: bool = False) -> dict:
    return {
        "full_name": f"user/{name}",
        "name": name,
        "language": language,
        "fork": fork,
        "description": f"{name} description",
        "topics": ["ai"],
    }


def mock_account(repos: list[dict], files: dict[str, dict[str, str]]) -> None:
    """files: {repo_full_name: {filename: content}}"""
    respx.get(f"{API}/users/user/repos").mock(return_value=httpx.Response(200, json=repos))
    respx.get(f"{API}/users/user/starred").mock(return_value=httpx.Response(200, json=[]))

    for r in repos:
        full = r["full_name"]
        present = files.get(full, {})
        respx.get(f"{API}/repos/{full}/git/trees/HEAD").mock(
            return_value=httpx.Response(
                200,
                json={"tree": [{"path": p, "type": "blob"} for p in present]},
            )
        )
        for filename, content in present.items():
            respx.get(f"{API}/repos/{full}/contents/{filename}").mock(
                return_value=httpx.Response(200, json=b64(content))
            )


# ─────────────────────────────────────────────────────────────────────────────
# Manifest parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestManifestParsing:
    def test_requirements_txt(self):
        content = "fastapi>=0.115\nlanggraph==0.4.0\n# a comment\n\n-e .\nqdrant-client"
        names = parse_manifest("requirements.txt", content)
        assert set(names) == {"fastapi", "langgraph", "qdrant-client"}

    def test_requirements_skips_vcs_and_urls(self):
        content = "git+https://github.com/x/y.git\nhttps://example.com/pkg.whl\nnumpy"
        assert parse_manifest("requirements.txt", content) == ["numpy"]

    def test_pyproject_pep621(self):
        content = """
[project]
name = "app"
dependencies = ["langgraph>=0.4", "httpx"]
[project.optional-dependencies]
dev = ["pytest"]
"""
        names = parse_manifest("pyproject.toml", content)
        assert "langgraph" in names and "httpx" in names and "pytest" in names

    def test_pyproject_poetry_layout(self):
        content = """
[tool.poetry.dependencies]
python = "^3.11"
langchain = "^0.3"
"""
        names = parse_manifest("pyproject.toml", content)
        assert "langchain" in names
        assert "python" not in names, "the interpreter is not a dependency"

    def test_package_json(self):
        content = json.dumps(
            {"dependencies": {"react": "^18"}, "devDependencies": {"vitest": "^2"}}
        )
        assert set(parse_manifest("package.json", content)) == {"react", "vitest"}

    def test_pubspec_yaml(self):
        content = "dependencies:\n  flutter:\n    sdk: flutter\n  dio: ^5.0\n  provider: ^6.0\n"
        names = parse_manifest("pubspec.yaml", content)
        assert "dio" in names and "provider" in names
        assert "flutter" not in names, "the SDK itself carries no signal"

    def test_go_mod_uses_the_last_path_segment(self):
        content = "module x\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n"
        assert "gin" in parse_manifest("go.mod", content)

    def test_cargo_toml(self):
        content = '[dependencies]\nserde = "1.0"\ntokio = { version = "1" }\n'
        assert set(parse_manifest("Cargo.toml", content)) == {"serde", "tokio"}

    @pytest.mark.parametrize("filename", sorted(MANIFESTS))
    def test_malformed_content_never_raises(self, filename):
        assert parse_manifest(filename, "{{{ not valid anything ]]]") == []

    def test_empty_content_is_safe(self):
        assert parse_manifest("requirements.txt", "") == []


# ─────────────────────────────────────────────────────────────────────────────
# Profile building
# ─────────────────────────────────────────────────────────────────────────────
class TestProfileBuilding:
    @respx.mock
    async def test_builds_dependency_to_repo_index(self):
        """The index behind the pitch: which of *my* repos use this dependency."""
        mock_account(
            [repo("ContextPilot"), repo("PlacementPilot")],
            {
                "user/ContextPilot": {"requirements.txt": "langgraph>=0.4\nqdrant-client"},
                "user/PlacementPilot": {"requirements.txt": "langgraph==0.4.1\nfastapi"},
            },
        )
        profile = await GitHubProfileBuilder().build("user")

        assert set(profile.repos_using("langgraph")) == {
            "user/ContextPilot",
            "user/PlacementPilot",
        }
        assert profile.repos_using("fastapi") == ["user/PlacementPilot"]
        assert profile.repos_using("nonexistent") == []

    @respx.mock
    async def test_languages_are_counted(self):
        mock_account([repo("a", "Python"), repo("b", "Python"), repo("c", "Dart")], {})
        profile = await GitHubProfileBuilder().build("user")
        assert profile.languages["python"] == 2
        assert profile.languages["dart"] == 1

    @respx.mock
    async def test_forks_are_excluded(self):
        """A fork says what someone glanced at, not what they build."""
        mock_account([repo("mine"), repo("theirs", fork=True)], {})
        profile = await GitHubProfileBuilder().build("user")
        assert profile.repos == ["user/mine"]

    @respx.mock
    async def test_ubiquitous_dependencies_are_filtered(self):
        """Keeping pytest and eslint would make every profile look identical."""
        mock_account(
            [repo("a")],
            {"user/a": {"requirements.txt": "pytest\nblack\nruff\nlanggraph"}},
        )
        profile = await GitHubProfileBuilder().build("user")
        assert set(profile.dependency_names()) == {"langgraph"}

    @respx.mock
    async def test_one_unreachable_repo_does_not_fail_the_profile(self):
        mock_account([repo("good"), repo("broken")], {"user/good": {"requirements.txt": "dio"}})
        respx.get(f"{API}/repos/user/broken/git/trees/HEAD").mock(
            return_value=httpx.Response(500)
        )
        respx.get(f"{API}/repos/user/broken/contents/requirements.txt").mock(
            return_value=httpx.Response(500)
        )
        respx.get(f"{API}/repos/user/broken/contents/package.json").mock(
            return_value=httpx.Response(500)
        )
        profile = await GitHubProfileBuilder().build("user")
        assert "dio" in profile.dependency_names()

    @respx.mock
    async def test_unknown_user_raises(self):
        respx.get(f"{API}/users/ghost/repos").mock(return_value=httpx.Response(404))
        with pytest.raises(LookupError, match="not found"):
            await GitHubProfileBuilder().build("ghost")

    @respx.mock
    async def test_account_with_no_repos_returns_an_empty_profile(self):
        respx.get(f"{API}/users/user/repos").mock(return_value=httpx.Response(200, json=[]))
        profile = await GitHubProfileBuilder().build("user")
        assert profile.is_empty


class TestRequestBudget:
    """Regression: probing every manifest name per repo exhausted the unauthenticated
    rate limit on a real 31-repo account and silently truncated the profile to one
    repo — the 403s were indistinguishable from 'no manifest here'."""

    @respx.mock
    async def test_lists_the_tree_once_instead_of_probing_every_manifest(self):
        repos = [repo(f"r{i}") for i in range(10)]
        mock_account(repos, {f"user/r{i}": {"requirements.txt": "dio"} for i in range(10)})

        await GitHubProfileBuilder().build("user")

        content_calls = [
            c for c in respx.calls if "/contents/" in str(c.request.url)
        ]
        tree_calls = [c for c in respx.calls if "/git/trees/" in str(c.request.url)]

        assert len(tree_calls) == 10, "one tree listing per repo"
        assert len(content_calls) == 10, (
            f"expected one fetch per existing manifest, got {len(content_calls)} — "
            f"probing all {len(MANIFESTS)} names would be {10 * len(MANIFESTS)}"
        )

    @respx.mock
    async def test_repo_without_manifests_costs_only_the_tree_listing(self):
        mock_account([repo("empty")], {"user/empty": {}})
        await GitHubProfileBuilder().build("user")
        assert not [c for c in respx.calls if "/contents/" in str(c.request.url)]
