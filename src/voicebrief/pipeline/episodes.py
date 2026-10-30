"""Episode orchestration and persistence.

Ties the pieces together: candidates -> brief graph -> TTS -> object store -> Postgres.
This is what the CLI, the cron endpoint and the API all call, so there is exactly one
definition of "generate an episode".

Ordering matters here. The episode row is written as `running` and *committed* before
generation starts, so a crash leaves a visible failed episode rather than silence, and
so the cost ledger (which writes on its own connection) has a row to point at. Audio
goes to the object store before the row is marked `ready`, so a `ready` episode always
has audio behind it.

What "ready" means is deliberately strict. A transcript with no story, a silent track
from a missing speech engine, or an upload that never happened used to be marked ready
and surface as a normal episode. They are now `degraded` (a usable transcript exists,
something expected is missing) or `failed` (nothing usable), with the reason stored.
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
from voicebrief.history import record_ranking
from voicebrief.llm.client import LLMClient, episode_cost_inr
from voicebrief.logging import get_logger
from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.tts.assemble import assemble
from voicebrief.tts.engines import build_engine

log = get_logger(__name__)

AUDIO_PREFIX = "episodes"


# An episode needs at least this share of the requested stories to count as complete.
MIN_STORY_SHARE = 0.5
# Speech engines that produce silence rather than speech.
SILENT_ENGINES = {"null"}


def judge_episode(
    *,
    story_count: int,
    requested_stories: int,
    render_audio: bool,
    audio_key: str | None,
    duration_seconds: float,
    engine_name: str,
    configured_engine: str,
) -> tuple[EpisodeStatus, list[str]]:
    """Decide how complete an episode really is, and say why when it is not."""
    if story_count == 0:
        return EpisodeStatus.failed, ["no stories were produced"]

    reasons: list[str] = []
    if story_count < max(1, round(requested_stories * MIN_STORY_SHARE)):
        reasons.append(f"only {story_count} of {requested_stories} requested stories were written")

    if render_audio:
        if audio_key is None:
            reasons.append("audio was not stored")
        elif duration_seconds <= 0:
            reasons.append("audio rendering produced no sound")
        elif engine_name in SILENT_ENGINES and configured_engine not in SILENT_ENGINES:
            reasons.append(
                f"speech engine {configured_engine!r} was unavailable; the track is silent"
            )

    return (EpisodeStatus.degraded if reasons else EpisodeStatus.ready), reasons


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
    status: str = "ready"


def s3_client():
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
    client = s3_client()
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
        content_type = "audio/mpeg" if key.endswith(".mp3") else "audio/wav"
        s3_client().upload_file(
            str(path), settings.s3_bucket, key, ExtraArgs={"ContentType": content_type}
        )
        return key
    except (BotoCoreError, ClientError, OSError) as exc:
        log.warning("storage.upload_failed", key=key, error=str(exc))
        return None


def encode_mp3(wav_path: Path, mp3_path: Path) -> bool:
    """Compress the finished episode. A ten-minute WAV is ~29 MB, an MP3 about a fifth.

    Done with libsndfile (bundled with `soundfile`), so no ffmpeg is needed. Returns
    False and leaves the WAV to be uploaded instead if encoding is unavailable.
    """
    try:
        import soundfile as sf

        data, rate = sf.read(str(wav_path), dtype="float32")
        sf.write(str(mp3_path), data, rate, format="MP3")
        return mp3_path.is_file() and mp3_path.stat().st_size > 0
    except Exception as exc:
        log.warning("storage.mp3_failed", error=f"{type(exc).__name__}: {exc}")
        return False


def presigned_url(key: str, *, expires: int = 3600) -> str | None:
    settings = get_settings()
    try:
        return s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expires,
        )
    except (BotoCoreError, ClientError) as exc:
        log.warning("storage.presign_failed", key=key, error=str(exc))
        return None


def _index_for_search(session: Session, episode_id: uuid.UUID) -> None:
    """Make the episode searchable. Never a reason to lose an episode.

    Indexing used to happen only on the API path, so everything the CLI or a scheduled
    run produced was invisible to search.
    """
    try:
        from voicebrief.api.memory import index_episode_segments

        index_episode_segments(session, episode_id)
    except Exception as exc:
        log.warning("episode.index_failed", episode=str(episode_id), error=str(exc))


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
    excluded: list[dict] | None = None,
    run_id: uuid.UUID | None = None,
    episode_id: uuid.UUID | None = None,
) -> GenerationResult:
    settings = get_settings()
    started = time.perf_counter()

    # Written before generation so a crash leaves a visible failure, not silence.
    episode = Episode(
        **({"id": episode_id} if episode_id else {}),
        user_id=user_id,
        mode=EpisodeMode.brief,
        status=EpisodeStatus.running,
        language=language,
        style=style,
    )
    session.add(episode)
    # Committed, not just flushed: the ledger writes cost rows on its own connection and
    # they reference this row, and a crash must leave this row behind as evidence.
    session.commit()

    client = LLMClient(session, episode_id=episode.id, run_id=run_id)

    try:
        graph = BriefGraph(client, max_stories=max_stories, target_minutes=target_minutes)
        written = graph.run(
            candidates,
            profile=profile,
            declared_topics=declared_topics,
            language=language,
            style=style,
        )
    except Exception as exc:
        # Rolled back first so the failed row is written from a clean transaction, then
        # committed: the caller's scope rolls back on the re-raise, and without this the
        # failed episode and its error vanished with it.
        session.rollback()
        episode = session.get(Episode, episode.id) or episode
        episode.status = EpisodeStatus.failed
        episode.error = f"{type(exc).__name__}: {exc}"[:2000]
        episode.generation_seconds = round(time.perf_counter() - started, 2)
        episode.cost_inr = episode_cost_inr(session, episode.id)
        session.commit()
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
            mp3_path = out_path.with_suffix(".mp3")
            if duration > 0 and encode_mp3(out_path, mp3_path):
                audio_key = upload_audio(mp3_path, f"{AUDIO_PREFIX}/{episode.id}.mp3")
            else:
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
    story_count = sum(1 for s in segments if s.kind == "story")
    status, reasons = judge_episode(
        story_count=story_count,
        requested_stories=max_stories,
        render_audio=render_audio,
        audio_key=audio_key,
        duration_seconds=duration,
        engine_name=engine_name,
        configured_engine=settings.tts_engine,
    )
    episode.status = status
    episode.error = "; ".join(reasons) or None
    episode.grounding = {
        **grounding,
        "warnings": errors,
        "tts_engine": engine_name,
    }
    session.flush()

    episode.cost_inr = episode_cost_inr(session, episode.id)
    session.flush()

    record_ranking(session, episode.id, written.get("ranking", []), excluded)
    _index_for_search(session, episode.id)

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
        errors=errors + reasons,
        status=status.value,
    )
    log.info(
        "episode.finished",
        status=status.value,
        episode=str(episode.id),
        seconds=result.generation_seconds,
        cost_inr=result.cost_inr,
        attribution=result.attribution_rate,
    )
    return result
