"""Full article text for items whose feed gives only a line.

News feeds (TechCrunch, The Verge, Techmeme …) and link aggregators (Hacker News) carry a
headline and one sentence. A story built from that is thin: the ranker demotes it, and a
writer given one sentence can only repeat it. So, for sources that opt in with
`fetch_article: true`, the linked page is fetched once and its article text kept as the
item's body.

Rules, in order of importance:

* **Polite.** robots.txt is honoured for our user agent, every request identifies
  itself, and each item is tried once (the outcome is recorded in `item.raw`), so a run
  never re-fetches what an earlier one already tried.
* **Bounded.** At most `MAX_FETCHES_PER_RUN` pages per run, a few at a time, each with a
  timeout and a size cap.
* **Never fatal.** A paywall, a 403 or a page with no recognisable article leaves the
  item exactly as it was.

Extraction is deliberately simple (lxml, no new dependency): drop the page furniture,
then take the paragraphs of whichever container holds the most paragraph text.
"""

from __future__ import annotations

import re
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
from lxml import html as lxml_html
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db.models import Item, Source
from voicebrief.logging import get_logger

log = get_logger(__name__)

MAX_FETCHES_PER_RUN = 60
CONCURRENCY = 6
TIMEOUT = httpx.Timeout(10.0, connect=5.0)
MAX_PAGE_BYTES = 2_000_000
# A summary at least this long already gives the writer something to work with. Feed
# summaries of 400-700 characters (The Verge, The Decoder) proved to be the first
# paragraph of articles three to six times as long, hence the generous cut-off.
SHORT_SUMMARY_CHARS = 900
MAX_ARTICLE_CHARS = 6000
# A navigation trail some sites put in a paragraph ("InfoQ Homepage News Title …").
_BREADCRUMB = re.compile(r"^\w+ Homepage \w+ ")
# Below this, what was extracted is a teaser or a paywall notice, not an article.
MIN_ARTICLE_CHARS = 300
MIN_PARAGRAPH_CHARS = 40

_FURNITURE = (
    "script", "style", "noscript", "nav", "header", "footer", "aside", "form",
    "iframe", "svg", "button", "figure",
)  # fmt: skip
_BOILERPLATE = re.compile(
    r"^(sign up|subscribe|advertisement|read more|related:|share this|follow us|"
    r"all rights reserved|cookie|we use cookies|newsletter)",
    re.IGNORECASE,
)


def extract_text(page: str | bytes) -> str:
    """The article's paragraphs as plain text, or "" when none are recognisable."""
    try:
        doc = lxml_html.fromstring(page)
    except ValueError:  # a str that still carries an XML encoding declaration
        doc = lxml_html.fromstring(page.encode() if isinstance(page, str) else page)
    except Exception:
        return ""
    for tag in _FURNITURE:
        for node in doc.iter(tag):
            node.drop_tree()

    containers = doc.xpath("//article") or doc.xpath("//main") or doc.xpath("//body") or [doc]
    best: list[str] = []
    for container in containers:
        paragraphs = []
        for p in container.iter("p"):
            text = " ".join(p.text_content().replace("﻿", "").split())
            if len(text) >= MIN_PARAGRAPH_CHARS and not (
                _BOILERPLATE.match(text) or _BREADCRUMB.match(text)
            ):
                paragraphs.append(text)
        if sum(map(len, paragraphs)) > sum(map(len, best)):
            best = paragraphs

    text = "\n\n".join(dict.fromkeys(best))  # a repeated pull quote counts once
    return text[:MAX_ARTICLE_CHARS] if len(text) >= MIN_ARTICLE_CHARS else ""


@dataclass
class _Robots:
    """robots.txt per host, fetched at most once per run."""

    client: httpx.Client
    agent: str
    parsers: dict[str, urllib.robotparser.RobotFileParser | None] = field(default_factory=dict)

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self.parsers:
            parser: urllib.robotparser.RobotFileParser | None = None
            try:
                response = self.client.get(f"{host}/robots.txt")
                if response.status_code == 200:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(response.text.splitlines())
                elif response.status_code in (401, 403):
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(["User-agent: *", "Disallow: /"])
            except Exception:
                parser = None  # unreachable robots.txt: the convention is "allowed"
            self.parsers[host] = parser
        parser = self.parsers[host]
        return True if parser is None else parser.can_fetch(self.agent, url)


def fetch_article(url: str, client: httpx.Client, robots: _Robots) -> tuple[str, str]:
    """(status, text). Status: ok | robots | http_<code> | not_html | too_large | empty | error."""
    try:
        if not robots.allowed(url):
            return "robots", ""
        response = client.get(url)
        if response.status_code != 200:
            return f"http_{response.status_code}", ""
        if "html" not in response.headers.get("content-type", "html"):
            return "not_html", ""
        if len(response.content) > MAX_PAGE_BYTES:
            return "too_large", ""
        # Decoded text, not bytes: httpx honours the declared charset and detects a
        # missing one, where lxml given bytes guessed wrong ("ZoÃ«" for "Zoë").
        text = extract_text(response.text)
        return ("ok", text) if text else ("empty", "")
    except Exception as exc:
        return f"error:{type(exc).__name__}", ""


def needs_article(item: Item) -> bool:
    return (
        not (item.body or "").strip()
        and len(item.summary or "") < SHORT_SUMMARY_CHARS
        and "article" not in (item.raw or {})
        and item.url.startswith(("http://", "https://"))
    )


def hydrate_articles(
    session: Session, items: list[Item], *, limit: int = MAX_FETCHES_PER_RUN
) -> dict[str, int]:
    """Fetch article text for the opted-in items that need it. Returns status counts."""
    opted_in = {
        s.id
        for s in session.query(Source).filter(Source.id.in_({i.source_id for i in items}))
        if (s.config or {}).get("fetch_article")
    }
    # Most engaged first, so the cap spends its fetches on the likeliest stories.
    todo = sorted(
        (i for i in items if i.source_id in opted_in and needs_article(i)),
        key=lambda i: -(i.engagement or 0.0),
    )[:limit]
    if not todo:
        return {}

    agent = get_settings().user_agent
    counts: dict[str, int] = {}
    with httpx.Client(
        timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": agent}
    ) as client:
        robots = _Robots(client, agent)
        # Resolve robots.txt for each host up front, serially, so parallel fetches never
        # race to fill the cache.
        for item in todo:
            robots.allowed(item.url)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            results = list(pool.map(lambda i: fetch_article(i.url, client, robots), todo))

    stamp = datetime.now(UTC).isoformat()
    for item, (status, text) in zip(todo, results, strict=True):
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        if text:
            item.body = text
        item.raw = {**(item.raw or {}), "article": {"status": status, "at": stamp}}
    session.flush()
    log.info("articles.fetched", tried=len(todo), **counts)
    return counts
