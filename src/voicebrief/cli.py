"""Operator CLI.

Every long-running action in VoiceBrief is reachable from here as well as from the
API, so a cron entry and a developer at a terminal drive the exact same code path.
"""

from __future__ import annotations

import asyncio
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
    configure_logging(get_settings().log_level)


@sources_app.command("sync")
def sources_sync(
    path: Path = typer.Option(Path("config/sources.yaml"), "--file", "-f", exists=True),
) -> None:
    """Upsert the source registry from YAML. Idempotent; safe to re-run."""
    from voicebrief.db import session_scope
    from voicebrief.db.repository import upsert_source

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = spec.get("sources", [])

    with session_scope() as session:
        for entry in entries:
            upsert_source(session, entry)

    console.print(f"[green]Synced {len(entries)} sources from {path}[/green]")


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


@brief_app.command("generate")
def brief_generate(
    email: str = typer.Option("me@example.com", "--user", help="User to generate for"),
    github: str = typer.Option("", "--github", help="GitHub login for the stack profile"),
    topics: str = typer.Option("agentic-ai,tooling", "--topics"),
    stories: int = typer.Option(6, "--stories"),
    minutes: int = typer.Option(10, "--minutes"),
    language: str = typer.Option("en", "--language", help="en or hi"),
    style: str = typer.Option("solo_anchor", "--style"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip synthesis"),
) -> None:
    """Run the full pipeline and store an episode."""
    import asyncio as _asyncio

    from sqlalchemy import select

    from voicebrief.db import session_scope
    from voicebrief.db.models import Language, User
    from voicebrief.pipeline.candidates import build_candidates
    from voicebrief.pipeline.episodes import generate_episode

    profile = None
    if github:
        from voicebrief.personalization.github_profile import GitHubProfileBuilder

        profile = _asyncio.run(GitHubProfileBuilder().build(github))
        console.print(
            f"[dim]stack profile: {len(profile.dependencies)} dependencies "
            f"across {len(profile.repos)} repos[/dim]"
        )

    with session_scope() as session:
        user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            user = User(email=email, language=Language(language), github_login=github or None)
            session.add(user)
            session.flush()

        candidates = build_candidates(session)
        if not candidates:
            console.print("[red]No clusters found. Run `ingest run` then `enrich` first.[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]{len(candidates)} candidate stories[/dim]")

        result = generate_episode(
            session,
            user_id=user.id,
            candidates=candidates,
            profile=profile,
            declared_topics={t.strip() for t in topics.split(",") if t.strip()},
            language=language,
            style=style,
            max_stories=stories,
            target_minutes=minutes,
            render_audio=not no_audio,
        )

    table = Table(title=result.title or "Episode")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, value in (
        ("episode id", str(result.episode_id)),
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
        console.print(f"[yellow]{len(result.errors)} non-fatal issue(s)[/yellow]")
        for err in result.errors[:5]:
            console.print(f"  [dim]{err}[/dim]")


if __name__ == "__main__":
    app()
