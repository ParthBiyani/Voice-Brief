"""Episode orchestration and persistence.

Ties the pieces together: candidates -> brief graph -> TTS -> object store -> Postgres.
This is what the CLI, the cron endpoint and the API all call, so there is exactly one
definition of "generate an episode".

Ordering matters here. The episode row is written as `running` *before* generation
starts, so a crash leaves a visible failed episode rather than silence. Audio goes to
the object store before the row is marked `ready`, so a `ready` episode always has
audio behind it.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.orm import Session

from voicebrief.config import get_settings
from voicebrief.db.models import Episode, EpisodeMode, EpisodeStatus, Segment
from voicebrief.graphs.brief import BriefGraph
from voicebrief.llm.client import LLMClient, episode_cost_inr
from voicebrief.logging import get_logger
from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.tts.assemble import assemble
from voicebrief.tts.engines import build_engine

log = get_logger(__name__)

AUDIO_PREFIX = "episodes"


@dataclass(slots=True)
class GenerationResult:
    episode_id: uuid.UUID
    title: str
    duration_seconds: float
    word_count: int
    cost_inr: float
    attribution_rate: float
    hallucinated_links: int
    generation_seconds: float
    audio_key: str | None
    engine: str
    errors: list[str]


def _s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        region_name="us-east-1",
    )


def ensure_bucket() -> None:
    settings = get_settings()
    client = _s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)
        log.info("storage.bucket_created", bucket=settings.s3_bucket)


def upload_audio(path: Path, key: str) -> str | None:
    """Store the rendered episode. Returns the key, or None if storage is down.

    A storage failure must not discard a generated episode: the transcript and
    segments are still worth keeping, and the audio can be re-rendered from cache.
    """
    settings = get_settings()
    try:
        ensure_bucket()
        _s3_client().upload_file(
            str(path), settings.s3_bucket, key, ExtraArgs={"ContentType": "audio/wav"}
        )
        return key
    except (BotoCoreError, ClientError, OSError) as exc:
        log.warning("storage.upload_failed", key=key, error=str(exc))
        return None


def presigned_url(key: str, *, expires: int = 3600) -> str | None:
    settings = get_settings()
    try:
        return _s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expires,
        )
    except (BotoCoreError, ClientError) as exc:
        log.warning("storage.presign_failed", key=key, error=str(exc))
        return None


def generate_episode(
    session: Session,
    *,
    user_id: uuid.UUID,
    candidates: list,
    profile: StackProfileData | None = None,
    declared_topics: set[str] | None = None,
    language: str = "en",
    style: str = "solo_anchor",
    max_stories: int = 8,
    target_minutes: int = 12,
    render_audio: bool = True,
) -> GenerationResult:
    settings = get_settings()
    started = time.perf_counter()

    # Written before generation so a crash leaves a visible failure, not silence.
    episode = Episode(
        user_id=user_id,
        mode=EpisodeMode.brief,
        status=EpisodeStatus.running,
        language=language,
        style=style,
    )
    session.add(episode)
    session.flush()

    client = LLMClient(session, episode_id=episode.id)

    try:
        graph = BriefGraph(client, max_stories=max_stories, target_minutes=target_minutes)
        written = graph.run(
            candidates,
            profile=profile,
            declared_topics=declared_topics,
            language=language,
            style=style,
        )
    except Exception as exc:  # noqa: BLE001 — record the failure, do not lose it
        episode.status = EpisodeStatus.failed
        episode.error = f"{type(exc).__name__}: {exc}"[:2000]
        session.flush()
        log.error("episode.failed", episode=str(episode.id), error=episode.error)
        raise

    segments = written.get("segments", [])
    grounding = written.get("grounding", {})
    errors = list(written.get("errors", []))

    audio_key: str | None = None
    duration = 0.0
    engine_name = "none"
    timed = []

    if render_audio and segments:
        engine = build_engine(settings.tts_engine, language=language)
        engine_name = engine.name
        voice = settings.tts_voice_hi if language == "hi" else settings.tts_voice_en
        cache_dir = Path(tempfile.gettempdir()) / "voicebrief" / "tts-cache"

        with tempfile.TemporaryDirectory() as workdir:
            out_path = Path(workdir) / f"{episode.id}.wav"
            assembled = assemble(
                segments, engine, voice=voice, out_path=out_path, cache_dir=cache_dir
            )
            duration = assembled.duration_seconds
            timed = assembled.segments
            errors.extend(assembled.failures)
            # Upload before marking ready, so a `ready` episode always has audio.
            audio_key = upload_audio(out_path, f"{AUDIO_PREFIX}/{episode.id}.wav")

    for index, segment in enumerate(segments):
        timing = timed[index] if index < len(timed) else None
        session.add(
            Segment(
                episode_id=episode.id,
                position=segment.position,
                kind=segment.kind,
                heading=segment.heading,
                script=segment.script,
                start_seconds=timing.start_seconds if timing else None,
                end_seconds=timing.end_seconds if timing else None,
                cluster_id=segment.cluster_id,
                citations=segment.citations,
            )
        )

    episode.title = written.get("title")
    episode.audio_key = audio_key
    episode.duration_seconds = duration
    episode.generation_seconds = round(time.perf_counter() - started, 2)
    episode.status = EpisodeStatus.ready if segments else EpisodeStatus.failed
    if not segments:
        episode.error = "no segments were produced"
    session.flush()

    episode.cost_inr = episode_cost_inr(session, episode.id)
    session.flush()

    result = GenerationResult(
        episode_id=episode.id,
        title=episode.title or "",
        duration_seconds=duration,
        word_count=written.get("word_count", 0),
        cost_inr=float(episode.cost_inr or 0.0),
        attribution_rate=float(grounding.get("attribution_rate", 1.0)),
        hallucinated_links=len(grounding.get("hallucinated_links", [])),
        generation_seconds=episode.generation_seconds or 0.0,
        audio_key=audio_key,
        engine=engine_name,
        errors=errors,
    )
    log.info(
        "episode.ready",
        episode=str(episode.id),
        seconds=result.generation_seconds,
        cost_inr=result.cost_inr,
        attribution=result.attribution_rate,
    )
    return result
