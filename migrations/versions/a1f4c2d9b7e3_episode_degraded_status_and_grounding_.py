"""episode degraded status and grounding report

Revision ID: a1f4c2d9b7e3
Revises: e762a88f5f39
Create Date: 2026-10-30 01:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1f4c2d9b7e3"
down_revision: str | None = "e762a88f5f39"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A new enum value cannot be used in the transaction that adds it, so the ALTER
    # runs outside the migration's own transaction.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE episode_status ADD VALUE IF NOT EXISTS 'degraded'")
    op.add_column("episode", sa.Column("grounding", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    # Postgres cannot drop an enum value; 'degraded' simply stays unused.
    op.drop_column("episode", "grounding")
