"""Tests for the injectable Database (dependency-inverted persistence)."""

from sqlalchemy import text

from config.settings import DatabaseSettings
from db.database import Database


def test_database_session_scope_yields_working_session():
    """A Database can be built from any settings and hand out working sessions —
    no module-level global, no environment variable, no real Postgres required."""
    db = Database(DatabaseSettings(url="sqlite+pysqlite:///:memory:"))

    with db.session_scope() as session:
        assert session.execute(text("SELECT 1")).scalar() == 1


def test_database_does_not_build_engine_until_first_use():
    """Constructing a Database must not connect or build an engine (lazy)."""
    db = Database(DatabaseSettings(url="postgresql://u:p@ep-x.neon.tech/db?sslmode=require"))

    assert db._engine is None


def test_set_default_database_overrides_the_process_default():
    """A composition root (e.g. an ingestion CLI) can point the default DB elsewhere."""
    import db.database as db_module
    from db.database import get_database, set_default_database

    original = db_module._default_database
    try:
        db = Database(DatabaseSettings(url="sqlite+pysqlite:///:memory:"))
        set_default_database(db)
        assert get_database() is db
    finally:
        db_module._default_database = original


# --- schema gate: Alembic owns the schema; the app only checks it is current ---

import pytest  # noqa: E402

from core.exceptions import SchemaValidationError  # noqa: E402


def _file_db(tmp_path) -> Database:
    return Database(DatabaseSettings(url=f"sqlite+pysqlite:///{tmp_path / 'gate.db'}"))


def _stamp(db: Database, revision: str) -> None:
    with db.engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version VALUES (:r)"), {"r": revision})


def test_require_current_schema_passes_at_head(tmp_path):
    from db.database import alembic_head

    db = _file_db(tmp_path)
    _stamp(db, alembic_head())

    db.require_current_schema()  # does not raise


def test_require_current_schema_rejects_a_database_alembic_never_met(tmp_path):
    """No alembic_version table: e.g. tables built by create_all. Must not pass."""
    db = _file_db(tmp_path)

    with pytest.raises(SchemaValidationError, match="alembic upgrade head"):
        db.require_current_schema()


def test_require_current_schema_rejects_a_database_behind_head(tmp_path):
    db = _file_db(tmp_path)
    _stamp(db, "1367bf7a9116")  # the initial revision

    with pytest.raises(SchemaValidationError, match="1367bf7a9116"):
        db.require_current_schema()
