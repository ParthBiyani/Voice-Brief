"""add the hinglish language option

Revision ID: d9a3b7c1e5f2
Revises: c4e8a2b6d1f7
Create Date: 2026-11-11 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "d9a3b7c1e5f2"
down_revision: str | None = "c4e8a2b6d1f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A new enum value cannot be used in the transaction that adds it.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE language ADD VALUE IF NOT EXISTS 'hinglish'")


def downgrade() -> None:
    # Postgres cannot drop an enum value; rows using it would have to be rewritten first.
    pass
