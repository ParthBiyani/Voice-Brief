"""GitHub adapters — REST API.

Two adapters share one auth story:

- `github_search` finds repos that are new or moving fast.
- `github_releases` watches a fixed list of repos. This one carries the highest trust
  weight in the registry, because a release for a dependency the user actually has in
  a lockfile is the single most actionable item the system can produce.

Both work unauthenticated (60 req/hr). A token raises that to 5000 and is read from
the env var named by the source's `auth_ref`, never stored in the row.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import httpx

from voicebrief.sources.base import RawItem, SourceAdapter, registry

_API_VERSION = "2022-11-28"
_STAR_SATURATION = 5000.0
_MAX_CONCURRENT = 5


class _GitHubBase(SourceAdapter):
    def headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _API_VERSION,
        }
        if self.settings.github_token:
            token = self.settings.github_token.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _rate_limited(response: httpx.Response) -> bool:
        return response.status_code in (403, 429) and response.headers.get(
            "x-ratelimit-remaining"
        ) in ("0", None)

    @staticmethod
    def _parse_ts(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


@registry.register
class GitHubSearchAdapter(_GitHubBase):
    """Trending-by-proxy: recently created repos sorted by stars.

    GitHub has no official trending API, so this reconstructs it from Search with a
    creation-date window — a documented, rate-limit-respecting endpoint rather than
    scraping the trending page.

    One request per topic, merged. The Search API rejects `OR` between qualifiers
    (422) and treats repeated `topic:` as AND, which collapses the result set to the
    handful of repos carrying every tag at once. Fanning out and merging is the only
    way to express "any of these topics".
    """

    kind = "github_search"

    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        cfg = self.config.config
        window_days = int(cfg.get("created_within_days", 30))
        min_stars = int(cfg.get("min_stars", 50))
        created_after = (datetime.now(timezone.utc) - timedelta(days=window_days)).date()

        topics: list[str] = cfg.get("topics") or ["llm"]
        qualifier = f"created:>{created_after} stars:>={min_stars}"
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

        results = await asyncio.gather(
            *(
                self._search(client, f"topic:{topic} {qualifier}", since, semaphore)
                for topic in topics
            ),
            return_exceptions=True,
        )

        # Merge on repo id — a repo tagged both `llm` and `agents` appears in two
        # result sets and must not become two items.
        merged: dict[str, RawItem] = {}
        failures = 0
        for topic, result in zip(topics, results, strict=True):
            if isinstance(result, BaseException):
                failures += 1
                self.log.warning("search.topic_failed", topic=topic, error=str(result))
                continue
            for item in result:
                merged.setdefault(item.external_id, item)

        if failures == len(topics):
            raise httpx.HTTPError(f"all {failures} GitHub search queries failed")
        return list(merged.values())

    async def _search(
        self,
        client: httpx.AsyncClient,
        query: str,
        since: datetime,
        semaphore: asyncio.Semaphore,
    ) -> list[RawItem]:
        async with semaphore:
            response = await client.get(
                self.config.endpoint,
                params={"q": query, "sort": "stars", "order": "desc", "per_page": "50"},
                headers=self.headers(),
            )
            if self._rate_limited(response):
                raise httpx.HTTPError(
                    "GitHub rate limit exhausted; set GITHUB_TOKEN to raise it"
                )
            response.raise_for_status()

            items: list[RawItem] = []
            for repo in response.json().get("items", []):
                pushed = self._parse_ts(repo.get("pushed_at"))
                created = self._parse_ts(repo.get("created_at"))
                if pushed is None or pushed < since:
                    continue

                # Creation date is the publication instant so the recency filter
                # treats this as a new repo; pushed_at decides whether it is alive.
                items.append(
                    RawItem(
                        external_id=str(repo["id"]),
                        url=repo["html_url"],
                        title=repo["full_name"],
                        summary=repo.get("description"),
                        author=repo.get("owner", {}).get("login"),
                        published_at=created or pushed,
                        topics=self._topics(repo),
                        engagement=min(
                            repo.get("stargazers_count", 0) / _STAR_SATURATION, 1.0
                        ),
                        raw={
                            "stars": repo.get("stargazers_count", 0),
                            "forks": repo.get("forks_count", 0),
                            "language": repo.get("language"),
                            "github_topics": repo.get("topics", []),
                            "pushed_at": repo.get("pushed_at"),
                        },
                    )
                )
            return items

    def _topics(self, repo: dict) -> list[str]:
        topics = set(self.config.default_topics)
        language = repo.get("language")
        if language:
            topics.add(language.lower())
        return sorted(topics)


@registry.register
class GitHubReleasesAdapter(_GitHubBase):
    """Releases for an explicit repo list.

    The repo list in `config.repos` is the static floor. At ranking time the user's
    stack profile supplies the repos that actually matter to them; this adapter just
    guarantees the common ones are always in the pool.
    """

    kind = "github_releases"

    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        repos: list[str] = self.config.config.get("repos", [])
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

        results = await asyncio.gather(
            *(self._fetch_repo(client, repo, since, semaphore) for repo in repos),
            return_exceptions=True,
        )

        items: list[RawItem] = []
        for repo, result in zip(repos, results, strict=True):
            if isinstance(result, BaseException):
                # A renamed or archived repo is expected drift, not a run failure.
                self.log.warning("releases.repo_failed", repo=repo, error=str(result))
                continue
            items.extend(result)
        return items

    async def _fetch_repo(
        self,
        client: httpx.AsyncClient,
        repo: str,
        since: datetime,
        semaphore: asyncio.Semaphore,
    ) -> list[RawItem]:
        async with semaphore:
            response = await client.get(
                f"{self.config.endpoint}/{repo}/releases",
                params={"per_page": "5"},
                headers=self.headers(),
            )
            if self._rate_limited(response):
                raise httpx.HTTPError("GitHub rate limit exhausted")
            response.raise_for_status()

            out: list[RawItem] = []
            for release in response.json():
                if release.get("draft"):
                    continue
                published = self._parse_ts(release.get("published_at"))
                if published is None or published < since:
                    continue

                body = (release.get("body") or "").strip()
                tag = release["tag_name"]
                name = release.get("name") or tag

                out.append(
                    RawItem(
                        external_id=f"{repo}@{tag}",
                        url=release["html_url"],
                        title=f"{repo} {name}",
                        # Release notes run long; the head carries the headline change.
                        summary=body[:2000] or None,
                        body=body or None,
                        author=repo.split("/")[0],
                        published_at=published,
                        topics=sorted({*self.config.default_topics, "releases"}),
                        # Releases are actionable regardless of popularity — the value
                        # comes from whether the user depends on it, decided at ranking.
                        engagement=0.3 if release.get("prerelease") else 0.6,
                        raw={
                            "repo": repo,
                            "tag": tag,
                            "prerelease": bool(release.get("prerelease")),
                        },
                    )
                )
            return out
