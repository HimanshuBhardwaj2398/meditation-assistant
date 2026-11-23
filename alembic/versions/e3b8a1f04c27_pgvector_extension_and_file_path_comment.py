"""pgvector extension owned by migrations; documents.file_path comment catches up

Until now the `vector` extension was created by Database.init_db(), which also
built tables with create_all() outside Alembic. init_db() is gone: Alembic
alone owns the schema, so it creates the extension too. IF NOT EXISTS makes
this a no-op on databases that already have it (Neon).

The file_path comment was updated in db/schema.py when the unique index landed
(b7c9d1e3f5a7) but never migrated; `alembic check` flagged the drift.

Revision ID: e3b8a1f04c27
Revises: c4f2a8d61b93
Create Date: 2026-11-23
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e3b8a1f04c27"
down_revision: Union[str, Sequence[str], None] = "c4f2a8d61b93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_COMMENT = "Original source: URL or file path"
_NEW_COMMENT = "Original source: URL or file path (unique — the DB enforces the dup guard)"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.alter_column(
        "documents",
        "file_path",
        existing_type=sa.Text(),
        existing_nullable=True,
        comment=_NEW_COMMENT,
        existing_comment=_OLD_COMMENT,
    )


def downgrade() -> None:
    """Downgrade schema.

    The extension is deliberately left in place: langchain_pg_embedding's
    `vector` column depends on it, and that table is not ours to drop.
    """
    op.alter_column(
        "documents",
        "file_path",
        existing_type=sa.Text(),
        existing_nullable=True,
        comment=_OLD_COMMENT,
        existing_comment=_NEW_COMMENT,
    )
