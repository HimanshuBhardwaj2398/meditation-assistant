"""Request-scoped dependencies."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from db.database import Database, get_database


def get_session(request: Request) -> Iterator[Session]:
    """One ``session_scope()`` per request: the route is the caller that owns the
    transaction (CRUD methods flush, never commit)."""
    database: Database = request.app.state.database or get_database()
    with database.session_scope() as session:
        yield session


# scope="function": commit/rollback runs when the route returns, before the
# response is sent, so a failed commit becomes a 500 instead of a 200 that lied.
SessionDep = Annotated[Session, Depends(get_session, scope="function")]
