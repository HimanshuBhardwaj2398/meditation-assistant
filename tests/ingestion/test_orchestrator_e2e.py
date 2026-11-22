"""Whole pipeline runs on SQLite, with only the source and vector store faked.

These pin what IngestionOrchestrator.process leaves in the documents, chunks
and vector tables for a new source, a failure, a retry and each reprocess mode,
so the orchestrator can be restructured without changing that.
"""

from types import SimpleNamespace

from sqlalchemy import select

from core.exceptions import DatabaseError, ParsingError
from db.crud import DocumentCRUD
from db.schema import Chunk, Document, DocumentStatus
from tests.ingestion.conftest import PROVENANCE, SOURCE, SUTTA_MARKDOWN, TITLE


def snapshot(db, document_id):
    """Read a document and its chunk rows into plain values."""
    with db.session_scope() as session:
        doc = session.get(Document, document_id)
        chunks = session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        ).all()
        return SimpleNamespace(
            status=doc.status,
            title=doc.title,
            markdown=doc.markdown,
            tags=list(doc.tags or []),
            metadata=dict(doc.doc_metadata or {}),
            chunk_indexes=[c.chunk_index for c in chunks],
            uuids=[c.uuid for c in chunks],
            texts=[c.chunk_text for c in chunks],
        )


def assert_published_in_step(state, vectors, document_id):
    """Every chunk row has exactly one vector, and nothing else is stored."""
    rows = vectors.rows()
    assert sorted(r["custom_id"] for r in rows) == sorted(state.uuids)
    assert all(r["document_id"] == document_id for r in rows)
    assert {r["custom_id"]: r["document"] for r in rows} == dict(zip(state.uuids, state.texts))


async def test_new_source_is_published_completed(orchestrator, ingest_db, vectors):
    result = await orchestrator.process(SOURCE)

    assert result["success"] is True, result["errors"]
    document_id = result["document_id"]
    state = snapshot(ingest_db, document_id)
    assert state.status == DocumentStatus.COMPLETED
    assert state.markdown == SUTTA_MARKDOWN
    assert state.title == TITLE
    assert {k: state.metadata[k] for k in PROVENANCE} == PROVENANCE
    assert state.metadata["table_of_contents"]["entries"]
    assert "mn" in state.tags
    assert len(state.uuids) >= 2
    assert result["chunk_count"] == len(state.uuids)
    assert state.chunk_indexes == list(range(len(state.uuids)))
    assert_published_in_step(state, vectors, document_id)


async def test_parse_failure_marks_document_failed(orchestrator, ingest_db, parser, vectors):
    parser.error = ParsingError("source offline")

    result = await orchestrator.process(SOURCE)

    assert result["success"] is False
    assert "source offline" in result["errors"]["parsing"]
    state = snapshot(ingest_db, result["document_id"])
    assert state.status == DocumentStatus.FAILED
    assert state.uuids == []
    assert vectors.rows() == []
    assert vectors.embedded == []


async def test_failed_publication_rolls_back_then_retry_completes(
    orchestrator, ingest_db, vectors, monkeypatch
):
    def interrupt(self, document_id):
        raise DatabaseError("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(DocumentCRUD, "clear_chunks", interrupt)
        failed = await orchestrator.process(SOURCE)

    assert failed["success"] is False
    assert failed["errors"]["database_persistence"] == "publication interrupted"
    document_id = failed["document_id"]
    state = snapshot(ingest_db, document_id)
    assert state.status == DocumentStatus.FAILED
    assert state.uuids == []
    assert vectors.rows() == []

    retried = await orchestrator.process(document_id)

    assert retried["success"] is True, retried["errors"]
    state = snapshot(ingest_db, document_id)
    assert state.status == DocumentStatus.COMPLETED
    assert_published_in_step(state, vectors, document_id)


async def test_reprocess_from_chunking_reuses_markdown(orchestrator, ingest_db, parser, vectors):
    first = await orchestrator.process(SOURCE)
    document_id = first["document_id"]
    before = snapshot(ingest_db, document_id)

    result = await orchestrator.process(document_id, reprocess_mode="from_chunking")

    assert result["success"] is True, result["errors"]
    assert parser.calls == [SOURCE]
    after = snapshot(ingest_db, document_id)
    assert after.status == DocumentStatus.COMPLETED
    assert after.texts == before.texts
    assert after.title == TITLE
    assert after.metadata["table_of_contents"]["entries"]
    assert_published_in_step(after, vectors, document_id)


async def test_reprocess_from_embedding_reuses_chunk_rows(orchestrator, ingest_db, parser, vectors):
    first = await orchestrator.process(SOURCE)
    document_id = first["document_id"]
    before = snapshot(ingest_db, document_id)

    result = await orchestrator.process(document_id, reprocess_mode="from_embedding")

    assert result["success"] is True, result["errors"]
    assert parser.calls == [SOURCE]
    after = snapshot(ingest_db, document_id)
    assert after.status == DocumentStatus.COMPLETED
    assert after.texts == before.texts
    assert vectors.embedded[-1] == before.texts
    assert_published_in_step(after, vectors, document_id)


async def test_reprocess_full_parses_again(orchestrator, ingest_db, parser, vectors):
    first = await orchestrator.process(SOURCE)
    document_id = first["document_id"]

    result = await orchestrator.process(document_id, reprocess_mode="full")

    assert result["success"] is True, result["errors"]
    assert parser.calls == [SOURCE, SOURCE]
    state = snapshot(ingest_db, document_id)
    assert state.status == DocumentStatus.COMPLETED
    assert_published_in_step(state, vectors, document_id)


async def test_status_trail_for_a_new_source(orchestrator, monkeypatch):
    """Every status written to documents.status during one clean run, in order."""
    trail = []
    original = DocumentCRUD.update_status

    def recording(self, document_id, status, status_details=None):
        trail.append(status)
        return original(self, document_id, status, status_details)

    monkeypatch.setattr(DocumentCRUD, "update_status", recording)

    result = await orchestrator.process(SOURCE)

    assert result["success"] is True, result["errors"]
    assert trail == [
        DocumentStatus.PARSING,
        DocumentStatus.PARSED,
        DocumentStatus.CHUNKING,
        DocumentStatus.CHUNKED,
        DocumentStatus.EMBEDDING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.COMPLETED,
    ]
