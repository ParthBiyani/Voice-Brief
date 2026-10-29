"""projectmind personal context cache

Revision ID: c4e8a2b6d1f7
Revises: b7d3e5a1c9f2
Create Date: 2026-10-30 03:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c4e8a2b6d1f7"
down_revision: str | None = "b7d3e5a1c9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("stack_profile", sa.Column("personal", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("stack_profile", "personal")
