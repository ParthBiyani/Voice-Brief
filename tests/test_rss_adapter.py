"""RSS adapter tests.

The cases here are drawn from feeds that actually misbehaved during catalogue
validation: title-only entries (Hugging Face), missing dates, HTML-laden summaries,
and feeds that 403 non-browser clients (PyTorch).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from voicebrief.sources.base import SourceConfig
from voicebrief.sources.rss import RSSAdapter, strip_html

FEED_URL = "https://example.com/feed.xml"


def feed(entries: str, title: str = "Example Blog") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>{title}</title>{entries}</channel></rss>"""


def entry(
    title: str = "A Post",
    *,
    link: str = "https://example.com/post-1",
    guid: str | None = "guid-1",
    date: str | None = None,
    description: str | None = "A plain summary.",
) -> str:
    date = date if date is not None else datetime.now(timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )
    parts = [f"<title>{title}</title>", f"<link>{link}</link>"]
    if guid:
        parts.append(f"<guid>{guid}</guid>")
    if date:
        parts.append(f"<pubDate>{date}</pubDate>")
    if description:
        parts.append(f"<description><![CDATA[{description}]]></description>")
    return "<item>" + "".join(parts) + "</item>"


def adapter(**over) -> RSSAdapter:
    cfg = {
        "slug": "example",
        "name": "Example",
        "endpoint": FEED_URL,
        "default_topics": ["engineering"],
        "trust_weight": 0.7,
    }
    return RSSAdapter(SourceConfig(**{**cfg, **over}))


async def run(body: str, since: datetime | None = None, status: int = 200):
    since = since or datetime.now(timezone.utc) - timedelta(days=7)
    respx.get(FEED_URL).mock(return_value=httpx.Response(status, text=body))
    async with httpx.AsyncClient() as client:
        return await adapter().fetch(client, since)


class TestStripHtml:
    def test_removes_tags_and_collapses_whitespace(self):
        assert strip_html("<p>Hello   <b>world</b></p>\n\n<p>Again</p>") == "Hello world Again"

    def test_unescapes_entities_that_matter_when_read_aloud(self):
        assert strip_html("R&amp;D &mdash; 5 &lt; 10") == "R&D — 5 < 10"

    def test_none_and_empty_pass_through_as_none(self):
        assert strip_html(None) is None
        assert strip_html("   ") is None


class TestParsing:
    @respx.mock
    async def test_maps_basic_entry(self):
        items = await run(feed(entry()))
        assert len(items) == 1
        item = items[0]
        assert item.title == "A Post"
        assert item.url == "https://example.com/post-1"
        assert item.external_id == "guid-1"
        assert item.summary == "A plain summary."
        assert item.topics == ["engineering"]

    @respx.mock
    async def test_html_in_summary_is_flattened(self):
        items = await run(feed(entry(description="<p>Ships <strong>durable</strong> execution.</p>")))
        assert items[0].summary == "Ships durable execution."

    @respx.mock
    async def test_missing_guid_falls_back_to_a_hash_of_the_link(self):
        """The (source, external_id) uniqueness constraint must still hold."""
        items = await run(feed(entry(guid=None)))
        assert items[0].external_id
        assert len(items[0].external_id) == 40

    @respx.mock
    async def test_title_only_entry_is_kept_with_no_summary(self):
        """Hugging Face's blog feed publishes exactly this shape."""
        items = await run(feed(entry(description=None)))
        assert len(items) == 1
        assert items[0].summary is None
        assert items[0].title == "A Post"

    @respx.mock
    async def test_entry_without_a_date_is_dropped(self):
        """Guessing 'now' would let stale posts masquerade as news on every run."""
        items = await run(feed(entry(title="Undated", date="") + entry(title="Dated")))
        assert [i.title for i in items] == ["Dated"]

    @respx.mock
    async def test_entry_without_a_link_is_dropped(self):
        items = await run(feed("<item><title>No link</title></item>" + entry()))
        assert [i.title for i in items] == ["A Post"]

    @respx.mock
    async def test_old_entries_are_filtered_by_the_window(self):
        old = "Mon, 01 Jan 2020 00:00:00 +0000"
        items = await run(feed(entry(title="Old", date=old) + entry(title="New")))
        assert [i.title for i in items] == ["New"]

    @respx.mock
    async def test_iso_dates_are_accepted(self):
        iso = datetime.now(timezone.utc).isoformat()
        items = await run(feed(entry(date=iso)))
        assert len(items) == 1

    @respx.mock
    async def test_one_malformed_entry_does_not_lose_the_feed(self):
        items = await run(feed("<item><title>Broken</title></item>" + entry(title="Good")))
        assert [i.title for i in items] == ["Good"]

    @respx.mock
    async def test_author_defaults_to_the_feed_title(self):
        items = await run(feed(entry(), title="Cloudflare Blog"))
        assert items[0].author == "Cloudflare Blog"

    @respx.mock
    async def test_rss_carries_no_engagement_signal(self):
        items = await run(feed(entry()))
        assert items[0].engagement == 0.0


class TestSummaryTruncation:
    @respx.mock
    async def test_long_summary_is_cut_on_a_sentence_boundary(self):
        body = ("This is a full sentence. " * 120).strip()
        items = await run(feed(entry(description=body)))
        summary = items[0].summary
        assert len(summary) < 1400
        assert summary.endswith(". …"), "must not sever a clause mid-way"

    @respx.mock
    async def test_short_summary_is_untouched(self):
        items = await run(feed(entry(description="Short.")))
        assert items[0].summary == "Short."


class TestFailureModes:
    @respx.mock
    async def test_empty_feed_raises_so_the_run_is_recorded(self):
        async with httpx.AsyncClient() as client:
            respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed("")))
            with pytest.raises(ValueError, match="no entries parsed"):
                await adapter().fetch(client, datetime.now(timezone.utc) - timedelta(days=7))

    @respx.mock
    async def test_forbidden_propagates_to_the_orchestrator(self):
        """PyTorch's feed 403s every non-browser client; the orchestrator records it
        and backs the source off rather than the adapter pretending it succeeded."""
        respx.get(FEED_URL).mock(return_value=httpx.Response(403))
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await adapter().fetch(client, datetime.now(timezone.utc) - timedelta(days=7))

    @respx.mock
    async def test_entry_cap_is_enforced(self):
        items = await run(feed("".join(entry(title=f"P{i}", guid=f"g{i}") for i in range(200))))
        assert len(items) == 60
