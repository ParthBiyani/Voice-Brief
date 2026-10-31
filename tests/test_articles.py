"""Full article text: extraction, politeness, and that it reaches the story."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import httpx
import pytest
import respx

from voicebrief.pipeline import articles
from voicebrief.pipeline.articles import extract_text, fetch_article, needs_article
from voicebrief.pipeline.candidates import best_text

PARA = "The company said the new chip ships next quarter with twice the memory bandwidth."


def page(body: str, head: str = "") -> str:
    return f"<html><head>{head}<title>t</title></head><body>{body}</body></html>"


ARTICLE = page(
    "<nav><p>Home · News · Sections · Newsletters · Podcasts · Events · Account</p></nav>"
    "<header><p>Sign up for our newsletter and get the day's top stories every morning</p></header>"
    f"<article><h1>Chip</h1><p>{PARA}</p><p>{PARA} Analysts expect strong demand.</p>"
    f"<p>{PARA} Pricing was not disclosed.</p><p>{PARA} Shipping starts in March.</p>"
    "<p>Subscribe to keep reading our coverage of the semiconductor industry today</p>"
    "</article><footer><p>All rights reserved. Copyright notice and legal terms here.</p></footer>"
    "<script>var x = 'Paragraph text inside a script must never be extracted';</script>"
)


class TestExtraction:
    def test_keeps_the_article_and_drops_the_furniture(self):
        text = extract_text(ARTICLE)
        assert text.startswith(PARA)
        assert "Pricing was not disclosed" in text
        for noise in ("Newsletters", "Sign up", "Subscribe", "All rights", "script"):
            assert noise not in text

    def test_paragraphs_stay_separate(self):
        assert extract_text(ARTICLE).count("\n\n") == 3

    def test_a_teaser_or_paywall_is_not_an_article(self):
        assert (
            extract_text(page("<article><p>Subscribe to read the full story here.</p></article>"))
            == ""
        )

    def test_without_an_article_tag_the_body_is_used(self):
        text = extract_text(page("".join(f"<div><p>{PARA} Part {i}.</p></div>" for i in range(5))))
        assert "Part 4" in text

    def test_a_breadcrumb_paragraph_is_dropped(self):
        body = f"<article><p>InfoQ Homepage News Netflix Treats GPUs As Systems</p>{'<p>' + PARA + '</p>' * 1}"
        body += "".join(f"<p>{PARA} {i}</p>" for i in range(4)) + "</article>"
        assert "Homepage" not in extract_text(page(body))

    def test_garbage_does_not_raise(self):
        assert extract_text(b"\x00\x01 not html at all") == ""


def item(**over):
    base = {
        "body": None,
        "summary": "One line.",
        "raw": {},
        "url": "https://news.example.com/a",
        "engagement": 0.5,
        "source_id": uuid.uuid4(),
    }
    return SimpleNamespace(**(base | over))


class TestWhichItems:
    def test_a_one_line_item_needs_its_article(self):
        assert needs_article(item())

    @pytest.mark.parametrize(
        "over",
        [
            {"body": "already have the text"},
            {"summary": "x" * 1000},
            {"raw": {"article": {"status": "http_403"}}},  # tried once already
            {"url": "item?id=123"},
        ],
    )
    def test_not_when_it_has_text_was_tried_or_has_no_web_link(self, over):
        assert not needs_article(item(**over))


class TestPoliteness:
    @respx.mock
    def test_robots_txt_is_honoured(self):
        respx.get("https://news.example.com/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /")
        )
        article = respx.get("https://news.example.com/a")
        with httpx.Client() as client:
            robots = articles._Robots(client, "VoiceBrief/0.1")
            assert fetch_article("https://news.example.com/a", client, robots) == ("robots", "")
        assert not article.called

    @respx.mock
    def test_a_missing_robots_txt_allows_and_the_article_is_read(self):
        respx.get("https://news.example.com/robots.txt").mock(return_value=httpx.Response(404))
        respx.get("https://news.example.com/a").mock(
            return_value=httpx.Response(
                200, text=ARTICLE, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        with httpx.Client() as client:
            status, text = fetch_article(
                "https://news.example.com/a", client, articles._Robots(client, "VB")
            )
        assert status == "ok" and text.startswith(PARA)

    @respx.mock
    def test_a_refusal_is_a_status_not_an_exception(self):
        respx.get("https://news.example.com/robots.txt").mock(return_value=httpx.Response(404))
        respx.get("https://news.example.com/a").mock(return_value=httpx.Response(403))
        with httpx.Client() as client:
            result = fetch_article(
                "https://news.example.com/a", client, articles._Robots(client, "VB")
            )
        assert result == ("http_403", "")

    @respx.mock
    def test_the_declared_charset_is_respected(self):
        respx.get("https://news.example.com/robots.txt").mock(return_value=httpx.Response(404))
        accented = ARTICLE.replace("The company", "Zoë's company")
        respx.get("https://news.example.com/a").mock(
            return_value=httpx.Response(
                200,
                content=accented.encode("utf-8"),
                headers={"content-type": "text/html; charset=utf-8"},
            )
        )
        with httpx.Client() as client:
            _, text = fetch_article(
                "https://news.example.com/a", client, articles._Robots(client, "VB")
            )
        assert "Zoë" in text and "Ã" not in text


class TestReachesTheStory:
    def test_the_fuller_text_wins(self):
        assert best_text(item(body="A long article. " * 50)).startswith("A long article")

    def test_a_real_summary_beats_a_scrap_of_body(self):
        assert best_text(item(summary="A proper abstract. " * 10, body="short")).startswith(
            "A proper abstract"
        )
