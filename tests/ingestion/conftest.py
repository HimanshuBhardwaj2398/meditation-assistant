"""Offline stand-ins for the ingestion pipeline's external boundaries.

The orchestrator, the stages and the CRUD layer run for real against a
throwaway SQLite database. Only the source (parser) and the vector store are
faked. The fake vector store writes its rows through the caller's session, so a
rolled-back publication rolls its vectors back too, as it does on pgvector.
"""

from typing import Any, Dict, List, Optional

import pytest
from sqlalchemy import text

from config.settings import DatabaseSettings
from core.interfaces import ParseResult
from db import database as database_module
from db.database import Database
from db.schema import Base
from ingestion.chunking import Config as ChunkingConfig
from ingestion.embed import VectorStoreConfig
from ingestion.orchestrator import IngestionOrchestrator
from tests import sqlite_compat

sqlite_compat.install()

SOURCE = "sc:mn1/sujato"
TITLE = "MN 1 The Root of All Things"
SUTTA_MARKDOWN = """# MN 1 The Root of All Things

## The Setting

So I have heard. At one time the Buddha was staying near Ukkattha, in the Subhaga
Forest at the root of a magnificent sal tree.

## The Teaching

There the Buddha addressed the mendicants: an unlearned ordinary person perceives
earth as earth, and having perceived earth as earth, conceives it.
"""
PROVENANCE = {
    "source": "suttacentral",
    "uid": "mn1",
    "author_uid": "sujato",
    "lang": "en",
    "nikaya": "mn",
    "nikaya_name": "majjhima_nikaya",
}

# Small sizes so the two sections become separate chunks; no model download.
TEST_CHUNKING = ChunkingConfig(
    max_size=400,
    min_size=100,
    tiny_chunk_threshold=20,
    enable_semantic=False,
    enable_parallel=False,
)


class FakeParser:
    """Stands in for ParserFactory: returns one fixed sutta, or raises."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.error: Optional[Exception] = None

    def parse(self, source: str) -> ParseResult:
        self.calls.append(source)
        if self.error is not None:
            raise self.error
        return ParseResult(
            content=SUTTA_MARKDOWN,
            title=TITLE,
            metadata={**PROVENANCE, "tags": ["buddhism", "mn"]},
        )


class SessionVectorStore:
    """Stands in for VectorStoreManager, keeping vectors in a SQLite table."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.embedded: List[List[str]] = []

    def validate_database(self, session: Any) -> None:
        pass

    def compute_embeddings(self, documents: List[Any]) -> List[List[float]]:
        self.embedded.append([doc.page_content for doc in documents])
        return [[float(len(doc.page_content))] for doc in documents]

    def stage_vectors(self, session: Any, documents: List[Any], vectors: List[Any]) -> None:
        for doc, _vector in zip(documents, vectors, strict=True):
            session.execute(
                text(
                    "INSERT INTO fake_vectors (custom_id, document_id, document) "
                    "VALUES (:custom_id, :document_id, :document)"
                ),
                {
                    "custom_id": doc.id,
                    "document_id": doc.metadata.get("document_id"),
                    "document": doc.page_content,
                },
            )

    def delete_by_ids(self, ids: List[str]) -> int:
        with self.db.session_scope() as session:
            for custom_id in ids:
                session.execute(
                    text("DELETE FROM fake_vectors WHERE custom_id = :id"), {"id": custom_id}
                )
        return len(ids)

    def rows(self) -> List[Dict[str, Any]]:
        with self.db.session_scope() as session:
            result = session.execute(
                text("SELECT custom_id, document_id, document FROM fake_vectors ORDER BY rowid")
            )
            return [dict(row._mapping) for row in result]


@pytest.fixture
def ingest_db(tmp_path, monkeypatch) -> Database:
    """A SQLite file holding the real schema, installed as the default database."""
    db = Database(DatabaseSettings(url=f"sqlite:///{tmp_path / 'ingest.db'}"))
    Base.metadata.create_all(db.engine)
    with db.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE fake_vectors "
                "(custom_id TEXT PRIMARY KEY, document_id INTEGER, document TEXT)"
            )
        )
    monkeypatch.setattr(database_module, "_default_database", db)
    yield db
    db.engine.dispose()


@pytest.fixture
def parser() -> FakeParser:
    return FakeParser()


@pytest.fixture
def vectors(ingest_db) -> SessionVectorStore:
    return SessionVectorStore(ingest_db)


@pytest.fixture
def orchestrator(parser, vectors) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        VectorStoreConfig(db_url="sqlite://"),
        chunking_config=TEST_CHUNKING,
        parser_factory=parser,
        vector_store_manager=vectors,
    )
