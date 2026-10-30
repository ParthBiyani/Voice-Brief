"""Backups of the one thing that cannot be re-crawled: the database and the audio.

Items, clusters and embeddings are rebuilt by the next ingest. Episodes, history, the
cost ledger and feedback are not, and they live in Postgres and MinIO inside Docker's
disk image, which a hard shutdown can corrupt beyond repair. So every generated brief is
followed by a dump written outside Docker, and the last few are kept.

Postgres is dumped with the `pg_dump` inside its own container, so the host needs no
Postgres tools. Audio is copied incrementally: an episode's MP3 never changes once made.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)

COMPOSE_PROJECT = "voicebrief"
DUMP_PATTERN = re.compile(r"^voicebrief-(\d{8}-\d{6})\.dump$")
DUMP_TIMEOUT_SECONDS = 300
REPO_ROOT = Path(__file__).resolve().parents[2]


class BackupError(RuntimeError):
    pass


@dataclass(slots=True)
class BackupResult:
    path: Path
    size_bytes: int
    media_copied: int
    pruned: int


def backup_dir() -> Path:
    configured = get_settings().backup_dir
    if configured:
        return Path(configured)
    base = (
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("XDG_DATA_HOME")
        or str(Path.home() / ".local" / "share")
    )
    return Path(base) / "VoiceBrief" / "backups"


def _database() -> tuple[str, str]:
    """User and database name, read from the same DSN the application uses."""
    parsed = urlparse(get_settings().postgres_dsn)
    return parsed.username or "voicebrief", (parsed.path or "/voicebrief").lstrip("/")


def _docker(*args: str) -> list[str]:
    compose = REPO_ROOT / "docker-compose.yml"
    return ["docker", "compose", "-f", str(compose), "-p", COMPOSE_PROJECT, *args]


def list_backups(folder: Path | None = None) -> list[Path]:
    """Dumps, newest first."""
    folder = folder or backup_dir()
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir() if DUMP_PATTERN.match(p.name)]
    return sorted(found, key=lambda p: p.name, reverse=True)


def prune(folder: Path, keep: int) -> int:
    removed = 0
    for old in list_backups(folder)[max(keep, 1) :]:
        old.unlink(missing_ok=True)
        removed += 1
    return removed


def dump_database(target: Path) -> None:
    user, database = _database()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    try:
        with partial.open("wb") as handle:
            done = subprocess.run(
                _docker(
                    "exec",
                    "-T",
                    "postgres",
                    "pg_dump",
                    "-U",
                    user,
                    "-d",
                    database,
                    "--format=custom",
                ),
                stdout=handle,
                stderr=subprocess.PIPE,
                timeout=DUMP_TIMEOUT_SECONDS,
                check=False,
            )
        if done.returncode != 0:
            raise BackupError(f"pg_dump exited {done.returncode}: {done.stderr.decode()[:300]}")
        verify_dump(partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def verify_dump(path: Path) -> int:
    """A dump that cannot be listed cannot be restored. Returns the number of entries."""
    with path.open("rb") as handle:
        done = subprocess.run(
            _docker("exec", "-T", "postgres", "pg_restore", "--list"),
            stdin=handle,
            capture_output=True,
            timeout=DUMP_TIMEOUT_SECONDS,
            check=False,
        )
    if done.returncode != 0:
        raise BackupError(f"the dump does not verify: {done.stderr.decode()[:300]}")
    entries = [
        line
        for line in done.stdout.decode(errors="replace").splitlines()
        if not line.startswith(";")
    ]
    if not entries:
        raise BackupError("the dump is empty")
    return len(entries)


def _media_folder(folder: Path) -> Path:
    return folder / "media"


def copy_media(folder: Path) -> int:
    """Download any audio not yet backed up. Returns how many files were new."""
    from voicebrief.pipeline.episodes import s3_client

    client = s3_client()
    bucket = get_settings().s3_bucket
    target = _media_folder(folder)
    copied = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            destination = target / obj["Key"]
            if destination.exists() and destination.stat().st_size == obj["Size"]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(destination))
            copied += 1
    return copied


def create_backup(*, now: datetime | None = None, with_media: bool = True) -> BackupResult:
    settings = get_settings()
    folder = backup_dir()
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    target = folder / f"voicebrief-{stamp}.dump"

    dump_database(target)
    copied = 0
    if with_media:
        try:
            copied = copy_media(folder)
        except Exception as exc:  # the database is the part that matters
            log.warning("backup.media_failed", error=f"{type(exc).__name__}: {exc}"[:200])
    pruned = prune(folder, settings.backup_keep)
    result = BackupResult(target, target.stat().st_size, copied, pruned)
    log.info(
        "backup.done",
        path=str(target),
        mb=round(result.size_bytes / 1_048_576, 2),
        media_copied=copied,
        pruned=pruned,
    )
    return result


def restore_backup(path: Path, *, with_media: bool = True) -> int:
    """Replace the database with the dump. Stop the app and any `daily` run first.

    Returns the number of audio files put back.
    """
    if not path.is_file():
        raise BackupError(f"no such backup: {path}")
    verify_dump(path)
    user, database = _database()
    with path.open("rb") as handle:
        done = subprocess.run(
            _docker(
                "exec",
                "-T",
                "postgres",
                "pg_restore",
                "-U",
                user,
                "-d",
                database,
                "--clean",
                "--if-exists",
                "--no-owner",
            ),  # fmt: skip
            stdin=handle,
            capture_output=True,
            timeout=DUMP_TIMEOUT_SECONDS,
            check=False,
        )
    # pg_restore exits 1 for warnings (e.g. dropping objects that did not exist yet).
    if done.returncode not in (0, 1):
        raise BackupError(f"pg_restore exited {done.returncode}: {done.stderr.decode()[:300]}")

    restored = 0
    media = _media_folder(path.parent)
    if with_media and media.is_dir():
        from voicebrief.pipeline.episodes import ensure_bucket, s3_client

        ensure_bucket()
        client, bucket = s3_client(), get_settings().s3_bucket
        for file in media.rglob("*"):
            if file.is_file():
                client.upload_file(str(file), bucket, file.relative_to(media).as_posix())
                restored += 1
    return restored


def backup_after_generation() -> dict:
    """The hook `daily` calls. Never raises: a failed backup must not cost the brief."""
    try:
        result = create_backup()
        return {"ok": True, "path": str(result.path), "mb": round(result.size_bytes / 1_048_576, 2)}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("backup.failed", error=error)
        return {"ok": False, "error": error}
