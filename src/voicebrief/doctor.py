"""`voicebrief doctor`: is this machine set up to make a brief?

A first run fails for ordinary reasons (Docker not started, no API key, migrations not
applied, no voice model), and each one used to surface as a stack trace somewhere deep in
the pipeline. The doctor checks them up front, one line each, and says what to type next.
Required checks decide the exit code; optional ones are advice.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from voicebrief.config import get_settings


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = True


def check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    return Check(
        "Python 3.11+", sys.version_info >= (3, 11), version, "install Python 3.11 or newer"
    )


def check_api_key() -> Check:
    settings = get_settings()
    if settings.llm_provider != "anthropic":
        return Check("Model access", True, f"provider is {settings.llm_provider}, no key needed")
    present = settings.anthropic_api_key is not None
    return Check(
        "Anthropic API key",
        present,
        "set" if present else "missing",
        "put ANTHROPIC_API_KEY=... in .env (https://console.anthropic.com)",
    )


def check_postgres() -> Check:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(get_settings().postgres_dsn, connect_args={"connect_timeout": 4})
        with engine.connect() as connection:
            version = connection.execute(text("select version_num from alembic_version")).scalar()
        return Check("Database", True, f"migrated to {version}")
    except Exception as exc:
        message = str(exc).lower()
        if "alembic_version" in message or "does not exist" in message:
            return Check("Database", False, "reachable, not migrated", "alembic upgrade head")
        return Check("Database", False, "not reachable", "docker compose up -d --wait")


def _http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=4.0).status_code < 500
    except httpx.HTTPError:
        return False


def check_qdrant() -> Check:
    ok = _http_ok(get_settings().qdrant_url + "/collections")
    return Check(
        "Vector store (Qdrant)",
        ok,
        "reachable" if ok else "not reachable",
        "docker compose up -d --wait",
    )


def check_storage() -> Check:
    ok = _http_ok(get_settings().s3_endpoint + "/minio/health/live")
    return Check(
        "Audio storage (MinIO)",
        ok,
        "reachable" if ok else "not reachable",
        "docker compose up -d --wait",
    )


def check_sources() -> Check:
    try:
        from sqlalchemy import func, select

        from voicebrief.db import session_scope
        from voicebrief.db.models import Source

        with session_scope() as session:
            count = session.execute(select(func.count()).select_from(Source)).scalar_one()
        return Check("Sources", count > 0, f"{count} registered", "voicebrief sources sync")
    except Exception:
        return Check("Sources", False, "could not be read", "voicebrief sources sync")


def check_speech() -> Check:
    from voicebrief.tts.engines import build_engine

    wanted = get_settings().tts_engine
    got = build_engine(wanted).name
    if got == wanted:
        return Check("Speech", True, f"{got}", required=False)
    hint = {
        "edge": "needs `pip install edge-tts` and an internet connection",
        "openai": "set OPENAI_API_KEY",
        "elevenlabs": "set ELEVENLABS_API_KEY",
        "kokoro": "voicebrief tts setup",
    }.get(wanted, "")
    silent = got == "null"
    return Check(
        "Speech",
        not silent,
        f"{wanted} unavailable, using {got}" + (" (briefs will be silent)" if silent else ""),
        hint or "voicebrief tts setup",
    )


def check_web() -> Check:
    dist = Path(get_settings().web_dist) if get_settings().web_dist else Path("web/dist")
    ok = (dist / "index.html").is_file()
    return Check(
        "Web app build",
        ok,
        "built" if ok else "not built",
        "cd web && npm ci && npm run build",
        required=False,
    )


def check_delivery() -> Check:
    settings = get_settings()
    channels = [
        name
        for name, on in (
            ("Slack", settings.slack_bot_token and settings.slack_channel_id),
            ("email", settings.smtp_host and settings.email_to),
        )
        if on
    ]
    return Check(
        "Delivery",
        True,
        ", ".join(channels) if channels else "none: briefs appear in the web app only",
        "see 'Slack' in the README",
        required=False,
    )


def check_github() -> Check:
    token = get_settings().github_token is not None
    return Check(
        "GitHub token",
        True,
        "set" if token else "not set (60 requests an hour; releases and trending may be cut short)",
        "GITHUB_TOKEN=... in .env",
        required=False,
    )


CHECKS: list[Callable[[], Check]] = [
    check_python,
    check_api_key,
    check_postgres,
    check_qdrant,
    check_storage,
    check_sources,
    check_speech,
    check_web,
    check_delivery,
    check_github,
]


def run_checks(checks: list[Callable[[], Check]] | None = None) -> list[Check]:
    results = []
    for check in checks or CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # a broken check is a finding, not a crash
            results.append(Check(check.__name__, False, f"check failed: {exc}"[:120]))
    return results


def exit_code(results: list[Check]) -> int:
    return 1 if any(not r.ok and r.required for r in results) else 0
