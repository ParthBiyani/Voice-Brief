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


if __name__ == "__main__":
    app()
