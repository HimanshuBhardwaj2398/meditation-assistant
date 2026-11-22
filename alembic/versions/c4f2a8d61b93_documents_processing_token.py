"""documents.processing_token: one ingestion run owns a document at a time

A run claims a document by setting processing_token, and only while the
column is NULL or the claim is stale (updated_at older than the lease). Every
status write and the publish transaction check the token, so a run that lost
its claim can no longer write. Nullable and unindexed: it is only ever read or
written together with the primary key.

Revision ID: c4f2a8d61b93
Revises: b7c9d1e3f5a7
Create Date: 2026-11-22
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4f2a8d61b93"
down_revision: Union[str, Sequence[str], None] = "b7c9d1e3f5a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "documents",
        sa.Column(
            "processing_token",
            sa.Text(),
            nullable=True,
            comment=(
                "Claim held by the ingestion run processing this document; NULL when idle. "
                "Every status write refreshes updated_at, which is the claim's heartbeat."
            ),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "processing_token")
