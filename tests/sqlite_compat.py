"""Let the real ``db.schema`` tables run on SQLite, for tests only.

The schema is Postgres-only in four places. On the ``sqlite`` dialect alone:

- ``JSONB`` and ``ARRAY`` columns become ``JSON`` (lists are stored as JSON text);
- ``TSVECTOR`` becomes ``TEXT``, and the generated ``to_tsvector`` column becomes
  a plain, always-NULL column (SQLite has no ``to_tsvector``);
- ``BigInteger`` primary keys become ``INTEGER``, the only type SQLite
  auto-increments.

Postgres compilation is untouched. The price: a green test here does not prove
the SQL runs on Postgres. Full-text search in particular cannot be tested this way.
"""

import json

from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateColumn

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    @compiles(JSONB, "sqlite")
    def _jsonb(type_, compiler, **kw):
        return "JSON"

    @compiles(ARRAY, "sqlite")
    def _array(type_, compiler, **kw):
        return "JSON"

    @compiles(TSVECTOR, "sqlite")
    def _tsvector(type_, compiler, **kw):
        return "TEXT"

    @compiles(BigInteger, "sqlite")
    def _bigint(type_, compiler, **kw):
        return "INTEGER"

    @compiles(CreateColumn, "sqlite")
    def _computed_as_plain(element, compiler, **kw):
        column = element.element
        if column.computed is not None:
            return f"{column.name} TEXT"
        return compiler.visit_create_column(element, **kw)

    pg_bind, pg_result = ARRAY.bind_processor, ARRAY.result_processor

    def bind_processor(self, dialect):
        if dialect.name == "sqlite":
            return lambda value: None if value is None else json.dumps(value)
        return pg_bind(self, dialect)

    def result_processor(self, dialect, coltype):
        if dialect.name == "sqlite":
            return lambda value: None if value is None else json.loads(value)
        return pg_result(self, dialect, coltype)

    ARRAY.bind_processor = bind_processor  # type: ignore[method-assign]
    ARRAY.result_processor = result_processor  # type: ignore[method-assign]
