"""Operator CLI.

Every long-running action in VoiceBrief is reachable from here as well as from the
API, so a cron entry and a developer at a terminal drive the exact same code path.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from voicebrief.config import get_settings
from voicebrief.logging import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="VoiceBrief operator CLI")
sources_app = typer.Typer(no_args_is_help=True, help="Manage the source registry")
ingest_app = typer.Typer(no_args_is_help=True, help="Run ingestion")
app.add_typer(sources_app, name="sources")
app.add_typer(ingest_app, name="ingest")

console = Console()


@app.callback()
def _init() -> None:
    # A Windows console is often cp1252, and story titles are arbitrary Unicode. An
    # unencodable character would otherwise crash a report after the work was done.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    configure_logging(get_settings().log_level)


@sources_app.command("sync")
def sources_sync(
    path: Path = typer.Option(Path("config/sources.yaml"), "--file", "-f", exists=True),
) -> None:
    """Upsert the source registry from YAML. Idempotent; safe to re-run."""
    from voicebrief.db import session_scope
    from voicebrief.db.repository import disable_sources_not_in, upsert_source

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = spec.get("sources", [])

    with session_scope() as session:
        for entry in entries:
            upsert_source(session, entry)
        retired = disable_sources_not_in(session, (e["slug"] for e in entries))

    console.print(f"[green]Synced {len(entries)} sources from {path}[/green]")
    if retired:
        console.print(f"[yellow]Disabled {len(retired)} retired: {', '.join(retired)}[/yellow]")


@sources_app.command("list")
def sources_list(enabled_only: bool = typer.Option(False, "--enabled")) -> None:
    """Show the registry as the pipeline sees it."""
    from sqlalchemy import select

    from voicebrief.db import session_scope
    from voicebrief.db.models import Source

    with session_scope() as session:
        stmt = select(Source).order_by(Source.trust_weight.desc())
        if enabled_only:
            stmt = stmt.where(Source.enabled.is_(True))
        rows = session.execute(stmt).scalars().all()

        table = Table(title=f"Sources ({len(rows)})")
        for col in ("slug", "kind", "trust", "every", "topics", "last run"):
            table.add_column(col)
        for s in rows:
            table.add_row(
                s.slug if s.enabled else f"[dim]{s.slug} (off)[/dim]",
                s.kind.value,
                f"{s.trust_weight:.2f}",
                f"{s.poll_interval_minutes}m",
                ", ".join(s.default_topics[:3]),
                s.last_status or "—",
            )
        console.print(table)


@ingest_app.command("run")
def ingest_run(
    slug: str | None = typer.Option(None, "--source", "-s", help="Limit to one source"),
    force: bool = typer.Option(False, "--force", help="Ignore poll intervals"),
) -> None:
    """Run one ingestion pass."""
    from voicebrief.pipeline.ingest import run_ingest

    stats = asyncio.run(run_ingest(only_slug=slug, force=force))

    table = Table(title="Ingest run")
    for col in ("source", "fetched", "new", "dup", "status"):
        table.add_column(col)
    for row in stats:
        table.add_row(
            row.slug,
            str(row.fetched),
            str(row.inserted),
            str(row.skipped),
            "[green]ok[/green]" if row.ok else f"[red]{(row.error or '')[:40]}[/red]",
        )
    console.print(table)
    console.print(
        f"[bold]{sum(r.inserted for r in stats)} new items "
        f"from {sum(1 for r in stats if r.ok)}/{len(stats)} sources[/bold]"
    )


@app.command("enrich")
def enrich_command(
    keep: int = typer.Option(300, "--keep", help="Items surviving the cheap filter"),
    threshold: float = typer.Option(0.92, "--threshold", help="Cosine dedup threshold"),
    topics: str = typer.Option("", "--topics", help="Comma-separated user topics"),
) -> None:
    """Embed, deduplicate and cluster the recent crawl."""
    from voicebrief.db import session_scope
    from voicebrief.pipeline.enrich import enrich

    user_topics = {t.strip() for t in topics.split(",") if t.strip()}
    with session_scope() as session:
        result = enrich(
            session, user_topics=user_topics or None, keep=keep, dedup_threshold=threshold
        )

    table = Table(title="Enrichment")
    table.add_column("stage")
    table.add_column("count", justify="right")
    for label, value in (
        ("items considered", result.considered),
        ("survived filter", result.filtered),
        ("embedded", result.embedded),
        ("duplicate groups", result.duplicate_groups),
        ("items collapsed", result.collapsed),
        ("clusters", result.clusters),
        ("multi-item clusters", result.multi_item_clusters),
    ):
        table.add_row(label, str(value))
    console.print(table)


brief_app = typer.Typer(no_args_is_help=True, help="Generate episodes")
app.add_typer(brief_app, name="brief")


def _csv(value: str | None) -> list[str] | None:
    items = [t.strip() for t in (value or "").split(",") if t.strip()]
    return items or None


def _request(
    email, github, topics, stories, minutes, language, style, no_audio
):  # pragma: no cover - thin option plumbing
    from voicebrief.pipeline.daily import BriefRequest

    return BriefRequest(
        email=email,
        github_login=github or None,
        topics=_csv(topics),
        language=language,
        style=style,
        stories=stories,
        minutes=minutes,
        render_audio=not no_audio,
    )


def _print_episode(result) -> None:
    table = Table(title=result.title or "Episode")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, value in (
        ("episode id", str(result.episode_id)),
        ("status", result.status),
        ("words", str(result.word_count)),
        ("duration", f"{result.duration_seconds / 60:.1f} min"),
        ("generation time", f"{result.generation_seconds:.1f}s"),
        ("cost", f"INR {result.cost_inr:.2f}"),
        ("attribution rate", f"{result.attribution_rate:.3f}"),
        ("hallucinated links", str(result.hallucinated_links)),
        ("tts engine", result.engine),
    ):
        table.add_row(label, value)
    console.print(table)
    if result.errors:
        console.print(f"[yellow]{len(result.errors)} issue(s)[/yellow]")
        for err in result.errors[:5]:
            console.print(f"  [dim]{err}[/dim]")


_EMAIL_OPT = typer.Option("me@example.com", "--user", help="User to generate for")
_GITHUB_OPT = typer.Option("", "--github", help="GitHub login; saved for later runs")
_TOPICS_OPT = typer.Option("", "--topics", help="Comma-separated; saved for later runs")
_STORIES_OPT = typer.Option(None, "--stories", help="Stories per brief (default 6)")
_MINUTES_OPT = typer.Option(None, "--minutes", help="Target length (default 10)")
_LANG_OPT = typer.Option(None, "--language", help="en or hi; saved for later runs")
_STYLE_OPT = typer.Option(None, "--style", help="solo_anchor or two_host")
_NO_AUDIO_OPT = typer.Option(False, "--no-audio", help="Skip synthesis")


@brief_app.command("generate")
def brief_generate(
    email: str = _EMAIL_OPT,
    github: str = _GITHUB_OPT,
    topics: str = _TOPICS_OPT,
    stories: int | None = _STORIES_OPT,
    minutes: int | None = _MINUTES_OPT,
    language: str | None = _LANG_OPT,
    style: str | None = _STYLE_OPT,
    no_audio: bool = _NO_AUDIO_OPT,
) -> None:
    """Generate one episode from what is already ingested."""
    from voicebrief.db import session_scope
    from voicebrief.pipeline.daily import NoFreshStories, generate_brief

    request = _request(email, github, topics, stories, minutes, language, style, no_audio)
    try:
        with session_scope() as session:
            result = generate_brief(session, request).result
    except NoFreshStories as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    _print_episode(result)
    if result.status != "ready":
        # Scripts and the scheduled task rely on the exit code to tell a good brief from
        # one that needs a look.
        raise typer.Exit(2 if result.status == "degraded" else 1)


@app.command("daily")
def daily_command(
    email: str = _EMAIL_OPT,
    github: str = _GITHUB_OPT,
    topics: str = _TOPICS_OPT,
    stories: int | None = _STORIES_OPT,
    minutes: int | None = _MINUTES_OPT,
    language: str | None = _LANG_OPT,
    style: str | None = _STYLE_OPT,
    no_audio: bool = _NO_AUDIO_OPT,
    force: bool = typer.Option(False, "--force", help="Generate even if today's brief exists"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Use what is already crawled"),
    trigger: str = typer.Option("cli", "--trigger", help="schedule or cli; recorded in history"),
) -> None:
    """Ingest, enrich, generate and index today's brief. Safe to run repeatedly.

    Exit codes: 0 done or nothing to do, 2 a brief was made but is degraded
    (for example silent audio), 1 failure.
    """
    from voicebrief.pipeline.daily import run_daily

    request = _request(email, github, topics, stories, minutes, language, style, no_audio)
    result = run_daily(request, force=force, skip_ingest=skip_ingest, trigger=trigger)

    if result.outcome == "locked":
        console.print("[yellow]Another daily run is already in progress; nothing to do.[/yellow]")
    elif result.outcome == "already_done":
        console.print(
            f"[green]Today's brief already exists[/green] ({result.status}): "
            f"{result.title}\n  episode {result.episode_id}  (use --force to make another)"
        )
    elif result.outcome == "failed":
        console.print(f"[red]Daily run failed:[/red] {result.error}")
    else:
        table = Table(title=result.title or "Today's brief")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for label, value in (
            ("episode id", str(result.episode_id)),
            ("status", result.status or ""),
            ("new items", str(result.items_ingested)),
            ("clusters", str(result.clusters)),
            ("duration", f"{result.duration_seconds / 60:.1f} min"),
            ("cost", f"INR {result.cost_inr:.2f}"),
            (
                "attribution",
                "-" if result.attribution_rate is None else f"{result.attribution_rate:.3f}",
            ),
        ):
            table.add_row(label, value)
        console.print(table)
        for issue in result.issues[:5]:
            console.print(f"  [yellow]{issue}[/yellow]")
    raise typer.Exit(result.exit_code)


personal_app = typer.Typer(no_args_is_help=True, help="Personal context from ProjectMind")
app.add_typer(personal_app, name="personal")


@personal_app.command("show")
def personal_show(
    email: str = _EMAIL_OPT,
    refresh: bool = typer.Option(False, "--refresh", help="Ask ProjectMind again now"),
) -> None:
    """What ProjectMind has told VoiceBrief about how you work (cached for a day)."""
    from sqlalchemy import select

    from voicebrief.db import session_scope
    from voicebrief.db.models import User
    from voicebrief.personalization.projectmind import configured_command
    from voicebrief.pipeline.daily import resolve_personal

    command = configured_command()
    console.print(
        "ProjectMind command: "
        + (
            " ".join(command)
            if command
            else "[yellow]not configured[/yellow] "
            "(set VB_PROJECTMIND_CMD, or put `projectmind` on PATH)"
        )
    )
    with session_scope() as session:
        user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            console.print("[red]No such user yet. Run `voicebrief daily` first.[/red]")
            raise typer.Exit(1)
        context = resolve_personal(session, user, refresh=refresh)

    if context is None or context.is_empty:
        console.print("[yellow]No personal context available.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"[bold]fetched[/bold] {context.fetched_at:%Y-%m-%d %H:%M} UTC, "
        f"{context.queries_answered} answers"
    )
    console.print("[bold]how you work[/bold]")
    for statement in context.statements[:15]:
        console.print(f"  - {statement}")
    console.print(f"[bold]interests[/bold]  {', '.join(context.interests) or '-'}")
    console.print(f"[bold]moved away from[/bold]  {', '.join(context.avoid) or '-'}")
    console.print(f"[bold]recent projects[/bold]  {', '.join(context.projects) or '-'}")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="0.0.0.0 to reach it from a phone"),
    port: int = typer.Option(8000, "--port"),
    no_auth: bool = typer.Option(False, "--no-auth", help="Allow network access without a token"),
) -> None:
    """Serve the web app and API together on one port."""
    import uvicorn

    from voicebrief.api.site import build_site, default_dist, is_loopback

    settings = get_settings()
    if not is_loopback(host) and not settings.api_token and not no_auth:
        console.print(
            "[red]Refusing to expose the app to the network without a token.[/red]\n"
            "Anyone who can reach it could generate briefs (which spend money) and play your "
            "audio. Set VB_API_TOKEN in .env, or pass --no-auth if you really mean it."
        )
        raise typer.Exit(1)

    if not (default_dist() / "index.html").is_file():
        console.print(
            "[yellow]The web app is not built.[/yellow] Run: cd web && npm ci && npm run build"
            "\nThe API still works at /api/docs."
        )
    where = "localhost" if is_loopback(host) else host
    console.print(f"[green]VoiceBrief at http://{where}:{port}[/green]")
    if settings.api_token:
        console.print("[dim]Protected by VB_API_TOKEN: open the page and enter it once.[/dim]")
    uvicorn.run(build_site(), host=host, port=port, log_level="info")


history_app = typer.Typer(no_args_is_help=True, help="Review what the system did and was asked")
app.add_typer(history_app, name="history")


@history_app.command("show")
def history_show(
    day: str = typer.Option("", "--date", help="YYYY-MM-DD in your timezone; default today"),
    details: bool = typer.Option(False, "--details", help="Include skipped stories"),
) -> None:
    """A timeline of one day: runs, what was picked and why, model calls, questions asked."""
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    from voicebrief.db import session_scope
    from voicebrief.history import timeline

    settings = get_settings()
    chosen = date.fromisoformat(day) if day else datetime.now(ZoneInfo(settings.timezone)).date()
    with session_scope() as session:
        data = timeline(session, chosen, timezone=settings.timezone)

    console.print(f"[bold]{data['date']}[/bold]")
    if not (data["runs"] or data["episodes"] or data["interactions"]):
        console.print("[dim]nothing recorded[/dim]")
        return

    for run in data["runs"]:
        colour = {"ok": "green", "skipped": "dim", "degraded": "yellow"}.get(run["status"], "red")
        summary = run["summary"] or {}
        bits = []
        if "ingest" in summary:
            bits.append(f"{summary['ingest']['new_items']} new items")
        if "candidates" in summary:
            c = summary["candidates"]
            bits.append(f"{c['considered']} candidates ({c['already_aired']} already aired)")
        console.print(
            f"  run [{colour}]{run['kind']}/{run['trigger']} {run['status']}[/{colour}] "
            f"{run['started_at'][11:19]}Z  {'; '.join(bits)}  {run['error'] or ''}"
        )

    for episode in data["episodes"]:
        console.print(
            f"\n[bold]{episode['title']}[/bold]  [{episode['status']}]  "
            f"INR {episode['cost_inr']:.2f}  {len(episode['llm_calls'])} model calls"
        )
        if episode["error"]:
            console.print(f"  [yellow]{episode['error']}[/yellow]")
        table = Table(show_header=True, header_style="dim")
        for col in ("", "story", "source", "score", "signals"):
            table.add_column(col)
        for entry in episode["ranking"]:
            if entry["outcome"] != "picked" and not details:
                continue
            mark = {"picked": "[green]*[/green]", "aired": "[dim]heard[/dim]"}.get(
                entry["outcome"], "[dim]-[/dim]"
            )
            top = sorted((entry["signals"] or {}).items(), key=lambda kv: -abs(kv[1]))[:3]
            table.add_row(
                mark,
                entry["title"][:70],
                entry["source"] or "",
                "" if entry["score"] is None else f"{entry['score']:.2f}",
                ", ".join(f"{k} {v:+.1f}" for k, v in top)
                + ("  stack: " + ", ".join(entry["stack_match"]) if entry["stack_match"] else ""),
            )
        console.print(table)

    for item in data["interactions"]:
        request = item["request"] or {}
        asked = request.get("question") or request.get("q") or request.get("vote") or request
        console.print(
            f"  {item['at'][11:19]}Z [cyan]{item['kind']}[/cyan] ({item['channel']}) {asked}"
        )


@history_app.command("export")
def history_export(
    since: str = typer.Option("", "--since", help="YYYY-MM-DD; default the last 30 days"),
    out: Path = typer.Option(Path("history.jsonl"), "--out"),
) -> None:
    """Write every record (runs, ranking decisions, model calls, questions, thumbs) as JSONL."""
    import json
    from datetime import UTC, datetime, timedelta

    from voicebrief.db import session_scope
    from voicebrief.history import export_records

    start = (
        datetime.fromisoformat(since).replace(tzinfo=UTC)
        if since
        else datetime.now(UTC) - timedelta(days=30)
    )
    counts: dict[str, int] = {}
    with session_scope() as session, out.open("w", encoding="utf-8") as handle:
        for record in export_records(session, start):
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            counts[record["type"]] = counts.get(record["type"], 0) + 1

    console.print(f"[green]Wrote {sum(counts.values())} records to {out}[/green]")
    for kind, n in sorted(counts.items()):
        console.print(f"  {kind}: {n}")


tts_app = typer.Typer(no_args_is_help=True, help="Speech engine setup")
app.add_typer(tts_app, name="tts")


@tts_app.command("setup")
def tts_setup() -> None:
    """Download the Kokoro voice model (about 350 MB, once) to the per-user model folder."""
    import httpx
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    from voicebrief.tts.engines import (
        ONNX_MODEL_FILE,
        ONNX_RELEASE,
        ONNX_VOICES_FILE,
        model_dir,
    )

    folder = model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]model folder: {folder}[/dim]")

    for name in (ONNX_MODEL_FILE, ONNX_VOICES_FILE):
        target = folder / name
        if target.is_file() and target.stat().st_size > 1_000_000:
            console.print(f"[green]{name}[/green] already present")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        with (
            httpx.stream("GET", f"{ONNX_RELEASE}/{name}", follow_redirects=True, timeout=60) as r,
            Progress(
                TextColumn(name),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as bar,
        ):
            r.raise_for_status()
            task = bar.add_task("dl", total=int(r.headers.get("content-length", 0)) or None)
            with partial.open("wb") as handle:
                for chunk in r.iter_bytes(1 << 16):
                    handle.write(chunk)
                    bar.update(task, advance=len(chunk))
        partial.replace(target)
    console.print("[green]Voice model ready.[/green] Try: voicebrief tts test")


@tts_app.command("test")
def tts_test(
    text: str = typer.Option(
        "VoiceBrief is working. This is a short test of the voice you will hear each morning.",
        "--text",
    ),
    voice: str = typer.Option("", "--voice", help="Defaults to the configured voice"),
    out: Path = typer.Option(Path("tts-test.wav"), "--out"),
) -> None:
    """Synthesize one sentence and report speed, to confirm speech works."""
    import time

    from voicebrief.tts.engines import build_engine

    settings = get_settings()
    engine = build_engine(settings.tts_engine)
    if engine.name == "null":
        console.print("[red]No speech engine available.[/red] Run: voicebrief tts setup")
        raise typer.Exit(1)

    started = time.perf_counter()
    clip = engine.synthesize(text, voice=voice or settings.tts_voice_en, out_path=out)
    took = time.perf_counter() - started
    console.print(
        f"[green]{engine.name}[/green] wrote {out} - {clip.duration_seconds:.1f}s of audio "
        f"in {took:.1f}s ({clip.duration_seconds / took:.1f}x realtime)"
    )


if __name__ == "__main__":
    app()
