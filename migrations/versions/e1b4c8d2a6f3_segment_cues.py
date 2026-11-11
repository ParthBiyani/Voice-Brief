"""sentence timings per segment, for click-to-play transcripts

Revision ID: e1b4c8d2a6f3
Revises: d9a3b7c1e5f2
Create Date: 2026-11-11 19:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1b4c8d2a6f3"
down_revision: str | None = "d9a3b7c1e5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "segment",
        sa.Column("cues", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )


def downgrade() -> None:
    op.drop_column("segment", "cues")
