"""Mode 2 entry point: documents in, episode out.

The counterpart of `daily.generate_brief` for material the listener supplies. The CLI and
the upload API both call this, so there is one definition of how a documents episode is
made.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from voicebrief.db.models import EpisodeMode
from voicebrief.documents.parse import ParsedDocument
from voicebrief.graphs.knowledge import persist_documents, run_knowledge
from voicebrief.pipeline.daily import (
    DEFAULT_MINUTES,
    DEFAULT_STORIES,
    BriefRequest,
    resolve_user,
)
from voicebrief.pipeline.episodes import GenerationResult, generate_episode


@dataclass(slots=True)
class DocumentsRequest:
    email: str = "me@example.com"
    language: str | None = None
    style: str | None = None
    stories: int | None = None
    minutes: int | None = None
    render_audio: bool = True


def generate_documents_episode(
    session: Session,
    request: DocumentsRequest,
    documents: list[ParsedDocument],
    origins: list[str],
    *,
    run_id: uuid.UUID | None = None,
    episode_id: uuid.UUID | None = None,
) -> GenerationResult:
    """Persist the documents, then write, voice and store an episode explaining them."""
    user = resolve_user(
        session,
        BriefRequest(email=request.email, language=request.language, style=request.style),
    )
    stored = persist_documents(session, user.id, documents, origins)

    language = user.language.value
    style = user.style.value
    stories = request.stories or DEFAULT_STORIES
    minutes = request.minutes or DEFAULT_MINUTES

    return generate_episode(
        session,
        user_id=user.id,
        candidates=[],
        language=language,
        style=style,
        max_stories=stories,
        target_minutes=minutes,
        render_audio=request.render_audio,
        run_id=run_id,
        episode_id=episode_id,
        mode=EpisodeMode.knowledge,
        produce=lambda client: run_knowledge(
            client,
            session,
            stored,
            language=language,
            style=style,
            max_stories=stories,
            target_minutes=minutes,
        ),
    )
