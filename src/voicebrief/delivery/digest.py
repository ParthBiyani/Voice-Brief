"""The morning message: what is in today's brief, in a form a phone can skim.

A digest is plain data built once from an episode and rendered per channel (Slack
blocks, plain text, HTML), so the channels never disagree about what the brief said.
"""

from __future__ import annotations

import html
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.config import local_timezone
from voicebrief.db.models import Episode, RankingSnapshot, Segment

MAX_SUMMARY_CHARS = 280
MAX_STORIES = 10
MAX_LINKS = 2

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class DigestStory:
    heading: str
    summary: str
    links: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    matched: list[str] = field(default_factory=list)
    personal: bool = False
    start_seconds: float = 0.0


@dataclass(slots=True)
class Digest:
    title: str
    date_label: str
    status: str
    duration_minutes: float
    cost_inr: float | None
    sourced: float | None
    stories: list[DigestStory]
    issues: list[str] = field(default_factory=list)
    web_url: str | None = None
    episode_id: uuid.UUID | None = None

    @property
    def headline(self) -> str:
        return f"VoiceBrief, {self.date_label}"

    def stats_line(self) -> str:
        parts = [f"{self.duration_minutes:.0f} min", f"{len(self.stories)} stories"]
        if self.sourced is not None:
            parts.append(f"{round(self.sourced * 100)}% of claims traced to a source")
        if self.cost_inr is not None:
            parts.append(f"₹{self.cost_inr:.2f}")
        return " · ".join(parts)


def first_sentences(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """The opening of a script, cut on a sentence boundary, for a one-glance summary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    summary = ""
    for sentence in _SENTENCE_END.split(text):
        if summary and len(summary) + len(sentence) + 1 > limit:
            break
        summary = f"{summary} {sentence}".strip()
        if len(summary) >= limit:
            break
    return summary if len(summary) <= limit + 40 else summary[:limit].rstrip() + "…"


def build_digest(session: Session, episode_id: uuid.UUID, *, base_url: str | None = None) -> Digest:
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise LookupError(f"no such episode: {episode_id}")

    picked = {
        row.cluster_id: row
        for row in session.execute(
            select(RankingSnapshot).where(
                RankingSnapshot.episode_id == episode_id, RankingSnapshot.outcome == "picked"
            )
        ).scalars()
        if row.cluster_id
    }

    stories: list[DigestStory] = []
    for segment in session.execute(
        select(Segment)
        .where(Segment.episode_id == episode_id, Segment.kind == "story")
        .order_by(Segment.position)
    ).scalars():
        row = picked.get(segment.cluster_id) if segment.cluster_id else None
        matched = list(row.matched_dependencies or []) if row else []
        personal = bool(matched or (row and (row.signals or {}).get("interests")))
        stories.append(
            DigestStory(
                heading=segment.heading or "Story",
                summary=first_sentences(segment.script),
                links=[
                    (c.get("title") or c.get("url", ""), c["url"])
                    for c in (segment.citations or [])[:MAX_LINKS]
                    if c.get("url")
                ],
                matched=matched,
                personal=personal,
                start_seconds=segment.start_seconds or 0.0,
            )
        )

    grounding = episode.grounding or {}
    created: datetime = episode.created_at
    link = f"{base_url.rstrip('/')}/" if base_url else None
    return Digest(
        title=episode.title or "Today's brief",
        date_label=created.astimezone(local_timezone()).strftime("%a %d %b"),
        status=episode.status.value,
        duration_minutes=(episode.duration_seconds or 0) / 60,
        cost_inr=float(episode.cost_inr) if episode.cost_inr is not None else None,
        sourced=grounding.get("attribution_rate"),
        stories=stories[:MAX_STORIES],
        issues=[episode.error] if episode.error else [],
        web_url=link,
        episode_id=episode.id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────
def _slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_slack_blocks(digest: Digest) -> list[dict]:
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": digest.headline[:150]}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": digest.stats_line()}]},
    ]
    if digest.status != "ready" and digest.issues:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":warning: *This brief is {digest.status}.* "
                    f"{_slack_escape(digest.issues[0])}"[:2900],
                },
            }
        )
    for story in digest.stories:
        lines = [f"*{_slack_escape(story.heading)}*", _slack_escape(story.summary)]
        if story.personal:
            reason = f"your stack: {', '.join(story.matched)}" if story.matched else "for you"
            lines.append(f":dart: _{_slack_escape(reason)}_")
        if story.links:
            lines.append(
                "  ".join(
                    f"<{url}|{_slack_escape(title[:60])}>" for title, url in story.links if url
                )
            )
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)[:2900]}}
        )
    if digest.web_url:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"<{digest.web_url}|Open the transcript and ask questions>",
                    }
                ],
            }
        )
    return blocks[:50]


def to_text(digest: Digest) -> str:
    lines = [digest.headline, digest.stats_line(), ""]
    if digest.status != "ready" and digest.issues:
        lines += [f"NOTE: this brief is {digest.status}: {digest.issues[0]}", ""]
    for story in digest.stories:
        lines.append(story.heading)
        lines.append(story.summary)
        if story.personal:
            lines.append(
                "  -> "
                + (f"your stack: {', '.join(story.matched)}" if story.matched else "for you")
            )
        lines += [f"  {title}: {url}" for title, url in story.links]
        lines.append("")
    if digest.web_url:
        lines.append(f"Transcript and questions: {digest.web_url}")
    return "\n".join(lines).strip() + "\n"


def to_html(digest: Digest) -> str:
    esc = html.escape
    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;line-height:1.5">',
        f'<h2 style="margin:0">{esc(digest.headline)}</h2>',
        f'<p style="color:#6b6862;margin:4px 0 16px">{esc(digest.stats_line())}</p>',
    ]
    if digest.status != "ready" and digest.issues:
        parts.append(
            f'<p style="color:#8a4b20"><b>This brief is {esc(digest.status)}.</b> {esc(digest.issues[0])}</p>'
        )
    for story in digest.stories:
        parts.append(f'<h3 style="margin:18px 0 4px">{esc(story.heading)}</h3>')
        parts.append(f'<p style="margin:0 0 4px">{esc(story.summary)}</p>')
        if story.personal:
            reason = f"your stack: {', '.join(story.matched)}" if story.matched else "for you"
            parts.append(f'<p style="margin:0;color:#2f5d50"><i>{esc(reason)}</i></p>')
        if story.links:
            links = " · ".join(
                f'<a href="{esc(url)}">{esc(title[:60])}</a>' for title, url in story.links
            )
            parts.append(f'<p style="margin:2px 0;font-size:13px">{links}</p>')
    if digest.web_url:
        parts.append(
            f'<p style="margin-top:20px"><a href="{esc(digest.web_url)}">Open the transcript and ask questions</a></p>'
        )
    parts.append("</div>")
    return "".join(parts)
