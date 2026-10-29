"""history and audit trail

Revision ID: b7d3e5a1c9f2
Revises: a1f4c2d9b7e3
Create Date: 2026-10-30 02:10:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b7d3e5a1c9f2"
down_revision: str | None = "a1f4c2d9b7e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    op.add_column("cost_entry", sa.Column("prompt", sa.Text(), nullable=True))
    op.add_column("cost_entry", sa.Column("system_prompt", sa.Text(), nullable=True))
    op.add_column("cost_entry", sa.Column("response", sa.Text(), nullable=True))
    op.add_column("cost_entry", sa.Column("stop_reason", sa.String(length=24), nullable=True))
    op.add_column("cost_entry", sa.Column("run_id", sa.UUID(), nullable=True))

    op.create_table(
        "run_log",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("trigger", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("episode_id", sa.UUID(), nullable=True),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["user_id"], ["app_user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["episode_id"], ["episode.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_log_started", "run_log", ["started_at"])

    op.create_table(
        "ranking_snapshot",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("episode_id", sa.UUID(), nullable=False),
        sa.Column("cluster_id", sa.UUID(), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("source_slug", sa.String(length=96), nullable=True),
        sa.Column("cluster_size", sa.Integer(), nullable=True),
        sa.Column("heuristic_score", sa.Float(), nullable=True),
        sa.Column("llm_score", sa.Float(), nullable=True),
        sa.Column("final_score", sa.Float(), nullable=True),
        sa.Column("signals", postgresql.JSONB(), nullable=False),
        sa.Column("matched_dependencies", postgresql.JSONB(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["episode_id"], ["episode.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ranking_snapshot_episode", "ranking_snapshot", ["episode_id", "position"])

    op.create_table(
        "interaction",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("channel", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("episode_id", sa.UUID(), nullable=True),
        sa.Column("segment_id", sa.UUID(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["user_id"], ["app_user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["episode_id"], ["episode.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["segment_id"], ["segment.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_interaction_kind_created", "interaction", ["kind", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_interaction_kind_created", table_name="interaction")
    op.drop_table("interaction")
    op.drop_index("ix_ranking_snapshot_episode", table_name="ranking_snapshot")
    op.drop_table("ranking_snapshot")
    op.drop_index("ix_run_log_started", table_name="run_log")
    op.drop_table("run_log")
    for column in ("run_id", "stop_reason", "response", "system_prompt", "prompt"):
        op.drop_column("cost_entry", column)
