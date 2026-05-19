"""Build a stack profile from a user's GitHub.

This is the feature the whole pitch rests on:

    "LangGraph 0.4 shipped durable execution. You use LangGraph in ContextPilot and
     PlacementPilot — this replaces the checkpoint workaround in both."

That sentence is only possible if the system knows, concretely, which dependencies
appear in which of the user's repos. So the profile is not a vague interest vector —
it is a dependency-to-repos index built from actual manifest files.

Works unauthenticated (60 req/hr, public repos). A token raises the ceiling to 5000
and adds private repos. Every stage degrades independently: a repo whose manifest
can't be fetched is skipped, not fatal.
"""

from __future__ import annotations

import asyncio
import base64
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)

API = "https://api.github.com"
_MAX_REPOS = 40
_MAX_CONCURRENT = 5

# Manifest files worth parsing, mapped to their ecosystem.
MANIFESTS = {
    "requirements.txt": "pypi",
    "pyproject.toml": "pypi",
    "package.json": "npm",
    "pubspec.yaml": "pub",
    "go.mod": "go",
    "Cargo.toml": "crates",
}

# Dependencies too ubiquitous to say anything about a person. Keeping them would make
# every user's profile look identical and drown the signal that matters.
STOPWORDS = {
    "pytest", "black", "ruff", "mypy", "flake8", "isort", "setuptools", "wheel", "pip",
    "typing-extensions", "six", "certifi", "urllib3", "charset-normalizer", "idna",
    "packaging", "attrs", "click", "colorama", "python-dateutil", "pyyaml",
    "eslint", "prettier", "typescript", "jest", "webpack", "babel", "@types/node",
    "vite", "nodemon", "ts-node", "rimraf", "cross-env", "flutter_lints", "lints",
}

_PY_REQ = re.compile(r"^\s*([A-Za-z0-9._-]+)")
_GO_REQ = re.compile(r"^\s*([\w./-]+)\s+v[\d.]")


@dataclass(slots=True)
class StackProfileData:
    """What the user actually builds with."""

    login: str
    languages: dict[str, int] = field(default_factory=dict)
    # dependency -> {"repos": [...], "ecosystem": "pypi"}
    dependencies: dict[str, dict] = field(default_factory=dict)
    repos: list[str] = field(default_factory=list)
    starred_topics: list[str] = field(default_factory=list)
    recent_commit_terms: list[str] = field(default_factory=list)
    built_at: datetime | None = None

    def dependency_names(self) -> set[str]:
        return set(self.dependencies)

    def repos_using(self, dependency: str) -> list[str]:
        return self.dependencies.get(dependency.lower(), {}).get("repos", [])

    @property
    def is_empty(self) -> bool:
        return not self.dependencies and not self.languages


class GitHubProfileBuilder:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.settings = get_settings()
        self._client = client
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token:
            token = self.settings.github_token.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def build(self, login: str) -> StackProfileData:
        client = self._client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        try:
            return await self._build(client, login)
        finally:
            if self._owns_client:
                await client.aclose()

    async def _build(self, client: httpx.AsyncClient, login: str) -> StackProfileData:
        profile = StackProfileData(login=login, built_at=datetime.now(timezone.utc))

        repos = await self._list_repos(client, login)
        if not repos:
            log.warning("profile.no_repos", login=login)
            return profile

        profile.repos = [r["full_name"] for r in repos]
        profile.languages = dict(
            Counter(r["language"].lower() for r in repos if r.get("language")).most_common()
        )

        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
        manifest_results = await asyncio.gather(
            *(self._repo_dependencies(client, r["full_name"], semaphore) for r in repos),
            return_exceptions=True,
        )

        for repo, result in zip(repos, manifest_results, strict=True):
            if isinstance(result, BaseException):
                log.debug("profile.repo_skipped", repo=repo["full_name"], error=str(result))
                continue
            for name, ecosystem in result:
                entry = profile.dependencies.setdefault(
                    name, {"repos": [], "ecosystem": ecosystem}
                )
                if repo["full_name"] not in entry["repos"]:
                    entry["repos"].append(repo["full_name"])

        # Starred repo topics are a softer signal than dependencies — what someone is
        # curious about rather than what they maintain — so they are kept separate and
        # weighted lower at ranking time.
        profile.starred_topics = await self._starred_topics(client, login)
        profile.recent_commit_terms = self._commit_terms(repos)

        log.info(
            "profile.built",
            login=login,
            repos=len(profile.repos),
            dependencies=len(profile.dependencies),
            languages=len(profile.languages),
        )
        return profile

    async def _list_repos(self, client: httpx.AsyncClient, login: str) -> list[dict]:
        response = await client.get(
            f"{API}/users/{login}/repos",
            params={"sort": "pushed", "per_page": str(_MAX_REPOS), "type": "owner"},
            headers=self._headers(),
        )
        if response.status_code == 404:
            raise LookupError(f"GitHub user {login!r} not found")
        response.raise_for_status()
        # Forks tell you what someone glanced at, not what they build.
        return [r for r in response.json() if not r.get("fork")]

    async def _repo_dependencies(
        self, client: httpx.AsyncClient, full_name: str, semaphore: asyncio.Semaphore
    ) -> list[tuple[str, str]]:
        """List the repo root once, then fetch only the manifests that exist.

        Probing all six manifest names per repo costs 6 requests each — 186 across a
        40-repo account, against an unauthenticated ceiling of 60 per hour. Measured
        on a real account, that silently truncated the profile to a single repo: the
        failures look identical to "this repo has no manifest".

        One tree listing plus one fetch per manifest that actually exists brings a
        typical account to roughly 40-60 requests total.
        """
        async with semaphore:
            present = await self._root_files(client, full_name)
            if present is None:
                # Tree unavailable (empty repo, or rate limited). Fall back to probing
                # the two most common manifests rather than giving up on the repo.
                present = {"requirements.txt", "package.json"}

            found: list[tuple[str, str]] = []
            for filename, ecosystem in MANIFESTS.items():
                if filename not in present:
                    continue
                content = await self._get_file(client, full_name, filename)
                if content is None:
                    continue
                for name in parse_manifest(filename, content):
                    if name and name.lower() not in STOPWORDS:
                        found.append((name.lower(), ecosystem))
            return found

    async def _root_files(self, client: httpx.AsyncClient, full_name: str) -> set[str] | None:
        """Filenames at the repo root, in one request. None if unavailable."""
        try:
            response = await client.get(
                f"{API}/repos/{full_name}/git/trees/HEAD",
                headers=self._headers(),
            )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            tree = response.json().get("tree", [])
        except ValueError:
            return None
        return {entry["path"] for entry in tree if entry.get("type") == "blob"}

    async def _get_file(
        self, client: httpx.AsyncClient, full_name: str, path: str
    ) -> str | None:
        try:
            response = await client.get(
                f"{API}/repos/{full_name}/contents/{path}", headers=self._headers()
            )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None

        payload = response.json()
        if payload.get("encoding") != "base64" or not payload.get("content"):
            return None
        try:
            return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return None

    async def _starred_topics(self, client: httpx.AsyncClient, login: str) -> list[str]:
        try:
            response = await client.get(
                f"{API}/users/{login}/starred",
                params={"per_page": "50"},
                headers=self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []

        counter: Counter[str] = Counter()
        for repo in response.json():
            counter.update(t.lower() for t in repo.get("topics", []))
            if language := repo.get("language"):
                counter[language.lower()] += 1
        return [topic for topic, _ in counter.most_common(20)]

    @staticmethod
    def _commit_terms(repos: list[dict]) -> list[str]:
        """Terms from repo descriptions.

        Reading commit messages would need one API call per repo and, at 60 requests
        an hour unauthenticated, would exhaust the budget before the manifests are
        fetched — and manifests carry far more signal.
        """
        counter: Counter[str] = Counter()
        for repo in repos:
            text = f"{repo.get('description') or ''} {' '.join(repo.get('topics', []))}"
            counter.update(
                word.lower()
                for word in re.findall(r"[A-Za-z][A-Za-z0-9+#-]{2,}", text)
                if word.lower() not in STOPWORDS
            )
        return [term for term, count in counter.most_common(30) if count >= 1]


# ─────────────────────────────────────────────────────────────────────────────
# Manifest parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_manifest(filename: str, content: str) -> list[str]:
    """Extract dependency names. Never raises — a malformed manifest yields nothing."""
    try:
        if filename == "requirements.txt":
            return _parse_requirements(content)
        if filename == "pyproject.toml":
            return _parse_pyproject(content)
        if filename == "package.json":
            return _parse_package_json(content)
        if filename == "pubspec.yaml":
            return _parse_pubspec(content)
        if filename == "go.mod":
            return _parse_go_mod(content)
        if filename == "Cargo.toml":
            return _parse_cargo(content)
    except Exception as exc:  # noqa: BLE001 — one bad manifest, not one bad profile
        log.debug("profile.manifest_unparseable", filename=filename, error=str(exc))
    return []


def _parse_requirements(content: str) -> list[str]:
    names = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-", "git+", "http")):
            continue
        if match := _PY_REQ.match(line):
            names.append(match.group(1))
    return names


def _parse_pyproject(content: str) -> list[str]:
    data = tomllib.loads(content)
    names: list[str] = []

    project = data.get("project", {})
    for spec in project.get("dependencies", []) or []:
        if match := _PY_REQ.match(str(spec)):
            names.append(match.group(1))
    for group in (project.get("optional-dependencies") or {}).values():
        for spec in group:
            if match := _PY_REQ.match(str(spec)):
                names.append(match.group(1))

    # Poetry keeps dependencies somewhere else entirely.
    poetry = data.get("tool", {}).get("poetry", {})
    names.extend(k for k in (poetry.get("dependencies") or {}) if k != "python")
    return names


def _parse_package_json(content: str) -> list[str]:
    import json

    data = json.loads(content)
    names: list[str] = []
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        names.extend((data.get(key) or {}).keys())
    return names


def _parse_pubspec(content: str) -> list[str]:
    import yaml

    data = yaml.safe_load(content) or {}
    names: list[str] = []
    for key in ("dependencies", "dev_dependencies"):
        section = data.get(key) or {}
        if isinstance(section, dict):
            names.extend(k for k in section if k != "flutter")
    return names


def _parse_go_mod(content: str) -> list[str]:
    names = []
    for line in content.splitlines():
        if match := _GO_REQ.match(line):
            module = match.group(1)
            # github.com/org/pkg -> pkg is what a human would call it
            names.append(module.rsplit("/", 1)[-1])
    return names


def _parse_cargo(content: str) -> list[str]:
    data = tomllib.loads(content)
    names: list[str] = []
    for key in ("dependencies", "dev-dependencies"):
        section = data.get(key) or {}
        if isinstance(section, dict):
            names.extend(section.keys())
    return names
