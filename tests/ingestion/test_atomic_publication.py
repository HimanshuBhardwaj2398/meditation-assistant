"""Run against disposable pgvector with INGESTION_TEST_POSTGRES_URL set."""

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from config.settings import DatabaseSettings
from core.exceptions import DatabaseError, EmbeddingError
from core.interfaces import ParseResult
from db import database
from db.crud import DocumentCRUD
from db.schema import Base
from ingestion import stages
from ingestion.embed import VectorStoreConfig, VectorStoreManager
from ingestion.orchestrator import IngestionOrchestrator


class BatchEmbeddings(Embeddings):
    fail_second = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_second and "Second chunk" in texts:
            raise EmbeddingError("provider interrupted")
        return [self.embed_query(value) for value in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0] if text == "Second chunk" else [1.0, 0.0]


def test_embedding_batch_failure_does_not_initialize_vector_store(monkeypatch):
    manager = VectorStoreManager(VectorStoreConfig(db_url="postgresql://test/test", batch_size=1))
    embeddings = BatchEmbeddings()
    embeddings.fail_second = True
    monkeypatch.setattr(manager, "_embeddings", embeddings)

    def unexpected_store():
        pytest.fail("Computing embeddings must not open the vector store")

    monkeypatch.setattr(manager, "_create_vector_store", unexpected_store)
    with pytest.raises(EmbeddingError, match="batch 2 failed: provider interrupted"):
        manager.compute_embeddings(
            [Document(page_content="First chunk"), Document(page_content="Second chunk")]
        )


@pytest.fixture
def postgres_database(monkeypatch):
    url = os.environ.get("INGESTION_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set INGESTION_TEST_POSTGRES_URL to a disposable pgvector test database")

    admin = create_engine(url)
    schema = f"ingestion_test_{uuid4().hex}"
    with admin.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    isolated_url = make_url(url).update_query_dict({"options": f"-csearch_path={schema},public"})
    db = database.Database(DatabaseSettings(url=isolated_url.render_as_string(hide_password=False)))
    try:
        Base.metadata.create_all(db.engine)
        monkeypatch.setattr(database, "_default_database", db)
        yield db
    finally:
        db.engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.mark.integration
async def test_interrupted_ingestion_can_retry_without_orphan_vectors(
    postgres_database, monkeypatch
):
    db = postgres_database
    orchestrator = IngestionOrchestrator(
        VectorStoreConfig(db_url=db.engine.url.render_as_string(hide_password=False), batch_size=1)
    )
    provider = BatchEmbeddings()
    monkeypatch.setattr(orchestrator.vector_store_manager, "_embeddings", provider)
    monkeypatch.setattr(orchestrator.parser_factory, "parse", lambda source: ParseResult("text"))

    class TwoChunkSplitter:
        def __init__(self, **kwargs):
            pass

        async def chunk(self):
            chunks = [Document(page_content="First chunk"), Document(page_content="Second chunk")]
            stats = SimpleNamespace(total_chunks=2, processing_time=0.0, avg_chunk_size=2)
            return chunks, stats

    monkeypatch.setattr(stages, "MarkdownChunker", TwoChunkSplitter)

    def row_counts(session):
        return (
            session.scalar(text("SELECT count(*) FROM langchain_pg_embedding")),
            session.scalar(text("SELECT count(*) FROM chunks")),
        )

    provider.fail_second = True
    first_attempt = await orchestrator.process("sc:mn1/sujato")
    assert first_attempt["success"] is False
    assert "provider interrupted" in first_attempt["errors"]["embedding"]
    document_id = first_attempt["document_id"]
    assert not inspect(db.engine).has_table("langchain_pg_embedding")
    with db.session_scope() as session:
        assert session.scalar(text("SELECT count(*) FROM chunks")) == 0

    def interrupt_publication(self, document_id):
        assert row_counts(self.db) == (2, 2)
        with db.session_scope() as observer:
            assert row_counts(observer) == (0, 0)
        raise DatabaseError("publication interrupted")

    provider.fail_second = False
    with monkeypatch.context() as publication:
        publication.setattr(DocumentCRUD, "clear_chunks", interrupt_publication)
        second_attempt = await orchestrator.process(document_id)
    assert second_attempt["success"] is False
    assert second_attempt["errors"]["database_persistence"] == "publication interrupted"
    with db.session_scope() as session:
        assert row_counts(session) == (0, 0)

    result = await orchestrator.process(document_id)
    assert result["success"] is True, result["errors"]
    with db.session_scope() as session:
        assert row_counts(session) == (2, 2)
        rows = session.execute(
            text("""
            SELECT c.uuid, v.cmetadata->>'uuid', c.chunk_text, v.document, d.status
            FROM chunks c JOIN langchain_pg_embedding v ON c.uuid = v.custom_id
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.chunk_index
        """)
        ).all()
    assert len(rows) == 2
    for uuid, metadata_uuid, chunk_text, vector_text, status in rows:
        assert uuid == metadata_uuid
        assert chunk_text == vector_text
        assert status == "completed"
    store = orchestrator.vector_store_manager.vector_store
    assert store.similarity_search("Second chunk", k=1)[0].page_content == "Second chunk"
