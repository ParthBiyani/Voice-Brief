"""API response models.

Separate from the ORM on purpose. Serialising database rows directly couples the wire
format to the schema, which means a migration silently changes the API — and here it
would also leak internals like `duplicate_of_id` and raw source payloads.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Citation(BaseModel):
    url: str
    title: str = ""
    item_id: str | None = None


class SegmentOut(BaseModel):
    id: uuid.UUID
    position: int
    kind: str
    heading: str | None = None
    script: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    citations: list[Citation] = Field(default_factory=list)

    @property
    def timestamp(self) -> str:
        seconds = int(self.start_seconds or 0)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


class EpisodeSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    status: str
    mode: str
    language: str
    style: str
    duration_seconds: float | None
    cost_inr: float | None
    created_at: datetime


class EpisodeDetail(EpisodeSummary):
    segments: list[SegmentOut] = Field(default_factory=list)
    audio_url: str | None = None
    generation_seconds: float | None = None
    error: str | None = None


class TranscriptLine(BaseModel):
    position: int
    timestamp: str
    heading: str | None
    text: str
    start_seconds: float
    citations: list[Citation] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    email: str = "me@example.com"
    github_login: str | None = None
    topics: list[str] = Field(default_factory=list)
    language: str = "en"
    style: str = "solo_anchor"
    max_stories: int = Field(default=8, ge=1, le=15)
    target_minutes: int = Field(default=12, ge=3, le=30)
    render_audio: bool = True


class GenerateAccepted(BaseModel):
    episode_id: uuid.UUID
    status: str
    stream_url: str


class FeedbackIn(BaseModel):
    segment_id: uuid.UUID
    vote: int = Field(description="1 for up, -1 for down")
    note: str | None = None


class SearchHit(BaseModel):
    segment_id: uuid.UUID
    episode_id: uuid.UUID
    episode_title: str | None
    heading: str | None
    excerpt: str
    timestamp: str
    start_seconds: float
    score: float
    citations: list[Citation] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str
    # One of the canned prompts, or free text.
    preset: str | None = Field(
        default=None, description="explain_simply | show_code | compare_with"
    )


class ChatResponse(BaseModel):
    answer: str
    segment_id: uuid.UUID
    citations: list[Citation] = Field(default_factory=list)
    cost_inr: float = 0.0


class IngestStats(BaseModel):
    slug: str
    fetched: int
    inserted: int
    skipped: int
    ok: bool
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    postgres: bool
    qdrant: bool
    llm_provider: str
    tts_engine: str
