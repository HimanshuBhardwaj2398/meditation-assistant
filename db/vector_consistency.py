"""Do chunk rows and vectors still match one to one?

Read-only checks over ``documents``, ``chunks`` and ``langchain_pg_embedding``,
plus the one repair this module offers: deleting vectors the pipeline wrote
whose chunk row is gone. Postgres only (JSONB operators). Callers own the
transaction; nothing here commits.

A vector written by the pipeline carries its document's id in ``cmetadata``.
Vectors without one were not written by this pipeline; they are counted but
never deleted.
"""

from dataclasses import dataclass, field
from typing import List, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

_ORPHAN_VECTORS = """
    FROM langchain_pg_embedding e
    WHERE e.cmetadata ? 'document_id'
      AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.uuid = e.custom_id)
"""

_CHUNKS_WITHOUT_VECTORS = """
    FROM chunks c
    WHERE NOT EXISTS (SELECT 1 FROM langchain_pg_embedding e WHERE e.custom_id = c.uuid)
"""


@dataclass
class ConsistencyReport:
    """Counts of what does not match, with a few examples of each."""

    vectors: int
    chunks: int
    # Pipeline vectors with no chunk row (repair: delete_orphan_vectors).
    orphan_vectors: int
    # Chunk rows with no vector (repair: reprocess the document from_embedding).
    chunks_without_vectors: int
    # Pipeline vectors naming a document that no longer exists.
    vectors_of_missing_documents: int
    # Vectors with no document_id: not this pipeline's, left alone.
    foreign_vectors: int
    orphan_vector_sample: List[str] = field(default_factory=list)
    documents_missing_vectors: List[int] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not (
            self.orphan_vectors or self.chunks_without_vectors or self.vectors_of_missing_documents
        )


def check_vector_consistency(session: Session, sample: int = 10) -> ConsistencyReport:
    """Count chunk rows and vectors that have lost their partner. Read-only."""

    def count(sql: str) -> int:
        return int(session.execute(text(sql)).scalar_one())

    def first(sql: str) -> list:
        return list(session.execute(text(sql), {"n": sample}).scalars())

    return ConsistencyReport(
        vectors=count("SELECT count(*) FROM langchain_pg_embedding"),
        chunks=count("SELECT count(*) FROM chunks"),
        orphan_vectors=count("SELECT count(*)" + _ORPHAN_VECTORS),
        chunks_without_vectors=count("SELECT count(*)" + _CHUNKS_WITHOUT_VECTORS),
        vectors_of_missing_documents=count(
            """
            SELECT count(*) FROM langchain_pg_embedding e
            WHERE e.cmetadata ? 'document_id'
              AND NOT EXISTS (
                SELECT 1 FROM documents d WHERE d.id::text = e.cmetadata->>'document_id'
              )
            """
        ),
        foreign_vectors=count(
            "SELECT count(*) FROM langchain_pg_embedding e WHERE NOT (e.cmetadata ? 'document_id')"
        ),
        orphan_vector_sample=first(
            "SELECT e.custom_id" + _ORPHAN_VECTORS + " ORDER BY e.custom_id LIMIT :n"
        ),
        documents_missing_vectors=first(
            "SELECT DISTINCT c.document_id"
            + _CHUNKS_WITHOUT_VECTORS
            + " ORDER BY c.document_id LIMIT :n"
        ),
    )


def delete_orphan_vectors(session: Session) -> int:
    """Delete pipeline vectors whose chunk row is gone. Returns how many."""
    result = cast(CursorResult, session.execute(text("DELETE" + _ORPHAN_VECTORS)))
    return result.rowcount
