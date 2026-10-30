"""Backups: retention, a failed backup never costing a brief, and a real round trip."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime

import pytest

from voicebrief import backup
from voicebrief.config import get_settings


@pytest.fixture
def folder(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"backup_dir": str(tmp_path), "backup_keep": 3})
    monkeypatch.setattr(backup, "get_settings", lambda: settings)
    return tmp_path


def touch_dump(folder, stamp):
    path = folder / f"voicebrief-{stamp}.dump"
    path.write_bytes(b"x")
    return path


class TestRetention:
    def test_only_the_newest_dumps_are_kept(self, folder):
        for day in range(1, 7):
            touch_dump(folder, f"2026102{day}-060000")
        assert backup.prune(folder, keep=3) == 3
        assert [p.name for p in backup.list_backups(folder)] == [
            "voicebrief-20261026-060000.dump",
            "voicebrief-20261025-060000.dump",
            "voicebrief-20261024-060000.dump",
        ]

    def test_other_files_in_the_folder_are_never_touched(self, folder):
        (folder / "notes.txt").write_text("mine")
        (folder / "media").mkdir()
        touch_dump(folder, "20261021-060000")
        backup.prune(folder, keep=1)
        assert (folder / "notes.txt").exists() and (folder / "media").is_dir()

    def test_listing_a_missing_folder_is_empty_not_an_error(self, tmp_path):
        assert backup.list_backups(tmp_path / "nowhere") == []

    def test_at_least_one_dump_is_always_kept(self, folder):
        touch_dump(folder, "20261021-060000")
        assert backup.prune(folder, keep=0) == 0


class TestFailuresNeverCostABrief:
    def test_a_broken_backup_is_reported_not_raised(self, folder, monkeypatch):
        def boom(**_):
            raise backup.BackupError("docker is not running")

        monkeypatch.setattr(backup, "create_backup", boom)
        outcome = backup.backup_after_generation()
        assert outcome["ok"] is False and "docker is not running" in outcome["error"]

    def test_a_failed_dump_leaves_no_partial_file(self, folder, monkeypatch):
        monkeypatch.setattr(
            backup.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=b"", stderr=b"no container"),
        )
        with pytest.raises(backup.BackupError, match="no container"):
            backup.dump_database(folder / "voicebrief-20261030-060000.dump")
        assert list(folder.iterdir()) == []

    def test_a_dump_that_cannot_be_listed_is_rejected(self, folder, monkeypatch):
        path = touch_dump(folder, "20261030-060000")
        monkeypatch.setattr(
            backup.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=b"", stderr=b"corrupt"),
        )
        with pytest.raises(backup.BackupError, match="does not verify"):
            backup.verify_dump(path)

    def test_restoring_a_missing_file_says_so(self, folder):
        with pytest.raises(backup.BackupError, match="no such backup"):
            backup.restore_backup(folder / "absent.dump")


def _compose_postgres_running() -> bool:
    """The dump runs inside VoiceBrief's own compose Postgres. A machine with Docker but
    another Postgres (CI uses a service container) cannot take one."""
    if shutil.which("docker") is None:
        return False
    done = subprocess.run(
        backup._docker("ps", "-q", "postgres"), capture_output=True, text=True, check=False
    )
    return done.returncode == 0 and bool(done.stdout.strip())


@pytest.mark.integration
@pytest.mark.skipif(not _compose_postgres_running(), reason="needs the voicebrief compose stack")
class TestRealDump:
    def test_a_dump_of_the_running_database_verifies(self, folder):
        result = backup.create_backup(
            now=datetime(2026, 10, 30, 6, 0, tzinfo=UTC), with_media=False
        )
        assert result.path.name == "voicebrief-20261030-060000.dump"
        assert result.size_bytes > 0
        assert backup.verify_dump(result.path) > 10, "a real schema has many entries"
        assert not list(folder.glob("*.partial"))
