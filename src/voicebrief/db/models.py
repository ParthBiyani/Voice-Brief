"""Persistent schema.

Design note (PRD §3): sources are rows, never code. Adding source #47 is an INSERT,
not a deploy. That is what makes "40+ sources" a property of the system rather than
40 hand-written scripts.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from voicebrief.db.base import Base, TimestampMixin


class SourceKind(str, enum.Enum):
    api = "api"
    rss = "rss"
    github = "github"
    youtube = "youtube"
    reddit = "reddit"


class Language(str, enum.Enum):
    en = "en"
    hi = "hi"


class EpisodeStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    ready = "ready"
    failed = "failed"


class EpisodeMode(str, enum.Enum):
    brief = "brief"          # Mode 1 — daily personalized brief
    knowledge = "knowledge"  # Mode 2 — knowledge-to-podcast


class EpisodeStyle(str, enum.Enum):
    solo_anchor = "solo_anchor"
    two_host = "two_host"


# ─────────────────────────────────────────────────────────────────────────────
# Source registry
# ─────────────────────────────────────────────────────────────────────────────
class Source(Base, TimestampMixin):
    __tablename__ = "source"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(96), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[SourceKind] = mapped_column(Enum(SourceKind, name="source_kind"), nullable=False)
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)

    # Name of the env var holding credentials, never the credential itself.
    auth_ref: Mapped[str | None] = mapped_column(String(64))

    poll_interval_minutes: Mapped[int] = mapped_column(Integer, default=720, nullable=False)
    default_topics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)

    # Prior on source quality, folded into the cheap pre-LLM filter.
    trust_weight: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Adapter-specific knobs (arXiv categories, GitHub query, subreddit, ...).
    config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(32))
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    items: Mapped[list[Item]] = relationship(back_populates="source")

    __table_args__ = (
        CheckConstraint("trust_weight >= 0 AND trust_weight <= 1", name="trust_weight_range"),
        Index("ix_source_enabled_kind", "enabled", "kind"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalized items
# ─────────────────────────────────────────────────────────────────────────────
class Item(Base, TimestampMixin):
    """One normalized unit of content, whatever the upstream shape."""

    __tablename__ = "item"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source.id", ondelete="CASCADE"), nullable=False
    )

    # Stable per-source identity; the (source, external_id) pair is the dedup key
    # at ingest time. Semantic dedup happens later and is a different concern.
    external_id: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(300))

    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    topics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)

    # Upstream popularity (HN points, GitHub stars, HF downloads) — normalized 0..1
    # by the adapter so ranking never needs to know which source it came from.
    engagement: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    raw: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Pipeline state
    embedded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    duplicate_of_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("item.id", ondelete="SET NULL")
    )
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("cluster.id", ondelete="SET NULL")
    )

    source: Mapped[Source] = relationship(back_populates="items")
    cluster: Mapped[Cluster | None] = relationship(back_populates="items")

    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_item_source_external"),
        Index("ix_item_published_at", "published_at"),
        Index("ix_item_embedded", "embedded"),
        Index("ix_item_cluster_id", "cluster_id"),
    )


class Cluster(Base, TimestampMixin):
    """A story: several items about the same thing, summarized once."""

    __tablename__ = "cluster"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    topics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    centroid_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    items: Mapped[list[Item]] = relationship(back_populates="cluster")

    __table_args__ = (Index("ix_cluster_run_date", "run_date"),)


# ─────────────────────────────────────────────────────────────────────────────
# Users, personalization, feedback
# ─────────────────────────────────────────────────────────────────────────────
class User(Base, TimestampMixin):
    __tablename__ = "app_user"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(200))

    topics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    language: Mapped[Language] = mapped_column(
        Enum(Language, name="language"), default=Language.en, nullable=False
    )
    style: Mapped[EpisodeStyle] = mapped_column(
        Enum(EpisodeStyle, name="episode_style"), default=EpisodeStyle.solo_anchor, nullable=False
    )
    github_login: Mapped[str | None] = mapped_column(String(120))

    profile: Mapped[StackProfile | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class StackProfile(Base, TimestampMixin):
    """What the user actually builds with, derived from their GitHub.

    This is the table behind the money line: "you use LangGraph in ContextPilot —
    this replaces the checkpoint workaround." Ranking reads it; nothing else should.
    """

    __tablename__ = "stack_profile"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    languages: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    # dependency -> {repos: [...], ecosystem: "pypi"|"npm"|"pub"}
    dependencies: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    repos: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    starred_topics: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    recent_commit_terms: Mapped[list[str]] = mapped_column(
        ARRAY(String), default=list, nullable=False
    )

    built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user: Mapped[User] = relationship(back_populates="profile")


class Feedback(Base, TimestampMixin):
    """Thumbs per segment. Replayed into the ranking prompt as few-shot examples."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("segment.id", ondelete="CASCADE"), nullable=False
    )
    vote: Mapped[int] = mapped_column(Integer, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("vote IN (-1, 1)", name="vote_is_thumb"),
        UniqueConstraint("user_id", "segment_id", name="uq_feedback_user_segment"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Episodes
# ─────────────────────────────────────────────────────────────────────────────
class Episode(Base, TimestampMixin):
    __tablename__ = "episode"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False
    )
    mode: Mapped[EpisodeMode] = mapped_column(Enum(EpisodeMode, name="episode_mode"), nullable=False)
    status: Mapped[EpisodeStatus] = mapped_column(
        Enum(EpisodeStatus, name="episode_status"), default=EpisodeStatus.pending, nullable=False
    )
    style: Mapped[EpisodeStyle] = mapped_column(
        Enum(EpisodeStyle, name="episode_style"), default=EpisodeStyle.solo_anchor, nullable=False
    )
    language: Mapped[Language] = mapped_column(
        Enum(Language, name="language"), default=Language.en, nullable=False
    )

    title: Mapped[str | None] = mapped_column(Text)
    audio_key: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    generation_seconds: Mapped[float | None] = mapped_column(Float)
    cost_inr: Mapped[float | None] = mapped_column(Numeric(10, 4))
    error: Mapped[str | None] = mapped_column(Text)

    segments: Mapped[list[Segment]] = relationship(
        back_populates="episode", cascade="all, delete-orphan", order_by="Segment.position"
    )

    __table_args__ = (Index("ix_episode_user_created", "user_id", "created_at"),)


class Segment(Base, TimestampMixin):
    """One story within an episode: the unit of playback, citation and Q&A."""

    __tablename__ = "segment"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("episode.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="story", nullable=False)

    heading: Mapped[str | None] = mapped_column(Text)
    script: Mapped[str] = mapped_column(Text, nullable=False)

    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)
    audio_key: Mapped[str | None] = mapped_column(Text)

    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("cluster.id", ondelete="SET NULL")
    )
    # [{item_id, url, title, span}] — every claim traces to one of these.
    citations: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    episode: Mapped[Episode] = relationship(back_populates="segments")

    __table_args__ = (
        UniqueConstraint("episode_id", "position", name="uq_segment_episode_position"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Documents (Mode 2)
# ─────────────────────────────────────────────────────────────────────────────
class Document(Base, TimestampMixin):
    __tablename__ = "document"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.id", ondelete="CASCADE"), nullable=False
    )
    # pdf | markdown | docx | text | repo
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str | None] = mapped_column(Text)
    storage_key: Mapped[str | None] = mapped_column(Text)

    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    claims: Mapped[list[Claim]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Claim(Base, TimestampMixin):
    """An assertion extracted from a document.

    Cross-document contrast (PRD §5) is an edge between two of these, which is why
    claims are extracted before any summarization happens.
    """

    __tablename__ = "claim"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    document: Mapped[Document] = relationship(back_populates="claims")


class ClaimRelation(Base, TimestampMixin):
    __tablename__ = "claim_relation"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="CASCADE"), nullable=False
    )
    target_claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="CASCADE"), nullable=False
    )
    # agrees | contradicts | prerequisite | extends
    relation: Mapped[str] = mapped_column(String(24), nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "source_claim_id", "target_claim_id", "relation", name="uq_claim_relation_edge"
        ),
        CheckConstraint("source_claim_id <> target_claim_id", name="no_self_relation"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cost ledger
# ─────────────────────────────────────────────────────────────────────────────
class CostEntry(Base, TimestampMixin):
    """One row per model call.

    The PRD's "< ₹8 per episode" target is only meaningful if it is measured rather
    than estimated, so every call through the LLM wrapper writes here — including
    the free local ones, which land at zero and still prove the accounting works.
    """

    __tablename__ = "cost_entry"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("episode.id", ondelete="SET NULL")
    )
    stage: Mapped[str] = mapped_column(String(48), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(96), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_inr: Mapped[float] = mapped_column(Numeric(10, 6), default=0, nullable=False)

    latency_ms: Mapped[int | None] = mapped_column(Integer)
    trace_id: Mapped[str | None] = mapped_column(String(96))

    __table_args__ = (
        Index("ix_cost_entry_created_at", "created_at"),
        Index("ix_cost_entry_episode", "episode_id"),
    )


class IngestRun(Base, TimestampMixin):
    """Per-source outcome of one ingestion pass. Failures are recorded, not raised —
    a dead source must never block the episode (PRD §10)."""

    __tablename__ = "ingest_run"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source.id", ondelete="CASCADE"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_ingest_run_source_started", "source_id", "started_at"),)
