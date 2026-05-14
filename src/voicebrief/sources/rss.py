"""Generic RSS/Atom adapter.

This is the highest-leverage source type in the system: one implementation turns
every company blog, release-notes feed and newsletter in Tier 2 into a database row.
Roughly 40 sources ship behind this single class.

Feeds in the wild are not well-behaved, so most of the work here is defence:

- Dates arrive in RFC-822, ISO-8601, and several things that are neither.
- Some feeds omit dates entirely on some entries.
- Content is HTML, sometimes with the full article, sometimes a teaser.
- IDs may be a GUID, a permalink, or absent.

Anything unparseable is dropped with a warning rather than raising, because a single
malformed entry must not cost the other 49 in the same feed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime, timezone

import feedparser
import httpx
from dateutil import parser as date_parser

from voicebrief.sources.base import RawItem, SourceAdapter, registry

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MAX_SUMMARY_CHARS = 1200
_MAX_ENTRIES = 60


def strip_html(raw: str | None) -> str | None:
    """Flatten feed HTML to plain text.

    Deliberately not a real HTML parser: feed summaries are shallow markup, and the
    text is destined for an LLM prompt and a TTS engine, neither of which benefits
    from structure. Entities that matter for reading aloud are unescaped.
    """
    if not raw:
        return None
    text = _TAG_RE.sub(" ", raw)
    for entity, char in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&rsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&mdash;", "—"),
    ):
        text = text.replace(entity, char)
    return _WS_RE.sub(" ", text).strip() or None


@registry.register
class RSSAdapter(SourceAdapter):
    kind = "rss"

    async def fetch(self, client: httpx.AsyncClient, since: datetime) -> Sequence[RawItem]:
        response = await client.get(
            self.config.endpoint,
            headers={
                "User-Agent": self.settings.user_agent,
                # Some CDNs serve XML only when the client asks for it explicitly.
                "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
            },
        )
        response.raise_for_status()

        feed = feedparser.parse(response.content)
        if not feed.entries:
            # bozo alone is not fatal — plenty of real feeds trip it and still parse.
            raise ValueError(
                f"no entries parsed from {self.config.endpoint}"
                + (f" ({feed.bozo_exception})" if feed.bozo else "")
            )

        feed_title = (feed.feed.get("title") or self.config.name).strip()
        items: list[RawItem] = []
        skipped = 0

        for entry in feed.entries[:_MAX_ENTRIES]:
            try:
                item = self._to_item(entry, feed_title, since)
            except Exception as exc:  # noqa: BLE001 — one bad entry, not one bad feed
                skipped += 1
                self.log.debug("rss.entry_skipped", error=str(exc))
                continue
            if item:
                items.append(item)

        if skipped:
            self.log.warning("rss.entries_skipped", count=skipped, feed=self.config.slug)
        return items

    def _to_item(self, entry, feed_title: str, since: datetime) -> RawItem | None:
        link = entry.get("link") or entry.get("id")
        if not link:
            return None

        published = self._published_at(entry)
        if published is None or published < since:
            return None

        title = strip_html(entry.get("title")) or "(untitled)"
        summary = self._summary(entry)

        return RawItem(
            # Prefer the feed's own GUID; fall back to a hash of the link so the
            # (source, external_id) uniqueness constraint still holds.
            external_id=entry.get("id") or hashlib.sha256(link.encode()).hexdigest()[:40],
            url=link,
            title=title,
            summary=summary,
            author=strip_html(entry.get("author")) or feed_title,
            published_at=published,
            topics=list(self.config.default_topics),
            # RSS carries no popularity signal. The source's trust_weight is the only
            # prior available, and it is applied later by the filter, not baked in here.
            engagement=0.0,
            raw={
                "feed": feed_title,
                "feed_slug": self.config.slug,
                "tags": [t.get("term") for t in entry.get("tags", []) if t.get("term")][:8],
            },
        )

    @staticmethod
    def _published_at(entry) -> datetime | None:
        """Feeds disagree about dates. Try the structured form, then parse the string."""
        for key in ("published_parsed", "updated_parsed", "created_parsed"):
            parsed = entry.get(key)
            if parsed:
                return datetime(*parsed[:6], tzinfo=timezone.utc)

        for key in ("published", "updated", "created", "date"):
            value = entry.get(key)
            if not value:
                continue
            try:
                dt = date_parser.parse(value)
            except (ValueError, OverflowError, TypeError):
                continue
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # A dateless entry cannot be placed in the recency window, and guessing "now"
        # would let stale posts masquerade as breaking news every single run.
        return None

    @staticmethod
    def _summary(entry) -> str | None:
        # `content` holds the full post when a feed publishes it; `summary` is the
        # teaser. Prefer whichever is longer — more grounding for the script stage.
        candidates: list[str] = []
        for block in entry.get("content", []) or []:
            if value := block.get("value"):
                candidates.append(value)
        if value := entry.get("summary"):
            candidates.append(value)

        best = max((strip_html(c) or "" for c in candidates), key=len, default="")
        if not best:
            return None
        if len(best) <= _MAX_SUMMARY_CHARS:
            return best
        # Cut on a sentence boundary so the LLM never sees a severed clause.
        clipped = best[:_MAX_SUMMARY_CHARS]
        cut = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
        return (clipped[: cut + 1] if cut > _MAX_SUMMARY_CHARS // 2 else clipped).strip() + " …"
