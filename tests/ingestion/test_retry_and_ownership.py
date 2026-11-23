"""Repeated submissions, retries, competing runs and failed replacements.

Runs the real orchestrator on SQLite (see conftest). SQLite has no row locks,
so the claim's race-freedom under real concurrency is Postgres's job; these
tests pin the rules: who may start a run, what a retry reuses, and what a
failed reprocess leaves behind.
"""

from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import select, text

from core.exceptions import (
    DatabaseError,
    DocumentBusyError,
    EmbeddingError,
    ProcessingLeaseLostError,
)
from core.interfaces import PipelineContext
from db.crud import DocumentCRUD
from db.schema import Chunk, Document, DocumentStatus
from ingestion.admission import admit
from ingestion.stages import DatabasePersistenceStage, ParsingStage
from tests.ingestion.conftest import SOURCE
from tests.ingestion.test_orchestrator_e2e import assert_published_in_step, snapshot


def hold_claim(db, document_id, token, *, age=timedelta(0)):
    """Make it look as if another run holds the document's claim."""
    with db.session_scope() as session:
        doc = session.get(Document, document_id)
        doc.processing_token = token
        doc.updated_at = datetime.now(timezone.utc) - age


def claim_of(db, document_id):
    with db.session_scope() as session:
        return session.get(Document, document_id).processing_token


async def test_resubmitting_a_completed_source_runs_nothing(orchestrator, parser, vectors):
    first = await orchestrator.process(SOURCE)

    again = await orchestrator.process("https://suttacentral.net/MN1/en/sujato")
    by_id = await orchestrator.process(first["document_id"])

    for result in (again, by_id):
        assert result["success"] is True
        assert result["already_completed"] is True
        assert result["document_id"] == first["document_id"]
        assert result["chunk_count"] == first["chunk_count"]
    assert first["already_completed"] is False
    assert parser.calls == [SOURCE]
    assert len(vectors.embedded) == 1


async def test_a_retry_resumes_the_same_document_from_its_stored_markdown(
    orchestrator, ingest_db, parser, vectors
):
    vectors.fail_next = EmbeddingError("provider down")
    failed = await orchestrator.process(SOURCE)
    assert failed["success"] is False
    assert snapshot(ingest_db, failed["document_id"]).status == DocumentStatus.FAILED
    assert claim_of(ingest_db, failed["document_id"]) is None

    retried = await orchestrator.process("sc:MN1/sujato")

    assert retried["success"] is True, retried["errors"]
    assert retried["document_id"] == failed["document_id"]
    assert parser.calls == [SOURCE]
    state = snapshot(ingest_db, retried["document_id"])
    assert state.status == DocumentStatus.COMPLETED
    assert_published_in_step(state, vectors, retried["document_id"])


async def test_a_document_another_run_is_processing_is_busy(orchestrator, ingest_db, parser):
    admission = admit(SOURCE)
    hold_claim(ingest_db, admission.document_id, "other-run")

    with pytest.raises(DocumentBusyError):
        await orchestrator.process(admission.document_id)
    with pytest.raises(DocumentBusyError):
        await orchestrator.process(SOURCE)

    assert parser.calls == []
    assert claim_of(ingest_db, admission.document_id) == "other-run"


async def test_a_dead_runs_claim_lapses_after_the_lease(orchestrator, ingest_db):
    admission = admit(SOURCE)
    hold_claim(
        ingest_db,
        admission.document_id,
        "dead-run",
        age=timedelta(seconds=orchestrator.lease_seconds + 60),
    )

    result = await orchestrator.process(admission.document_id)

    assert result["success"] is True, result["errors"]
    assert claim_of(ingest_db, admission.document_id) is None


async def test_a_run_that_is_taken_over_stops_without_writing(orchestrator, ingest_db, vectors):
    admission = admit(SOURCE)

    def another_run_takes_over():
        hold_claim(ingest_db, admission.document_id, "newer-run")

    vectors.before_embedding = another_run_takes_over

    with pytest.raises(ProcessingLeaseLostError):
        await orchestrator.process(SOURCE)

    state = snapshot(ingest_db, admission.document_id)
    assert state.uuids == []
    assert vectors.rows() == []
    assert claim_of(ingest_db, admission.document_id) == "newer-run"


async def test_publishing_requires_the_runs_claim(ingest_db, vectors):
    admission = admit(SOURCE)
    hold_claim(ingest_db, admission.document_id, "current-owner")
    chunk = LangchainDocument(
        id="chunk-1",
        page_content="Text",
        metadata={"uuid": "chunk-1", "document_id": admission.document_id},
    )
    context = PipelineContext(
        document_id=admission.document_id,
        chunks=[chunk],
        vectors=[[1.0]],
        processing_token="superseded-run",
    )

    with pytest.raises(ProcessingLeaseLostError):
        await DatabasePersistenceStage(vectors).execute(context)

    assert snapshot(ingest_db, admission.document_id).uuids == []
    assert vectors.rows() == []


async def test_storing_parsed_markdown_requires_the_runs_claim(ingest_db, parser):
    admission = admit(SOURCE)
    hold_claim(ingest_db, admission.document_id, "current-owner")
    context = PipelineContext(
        document_id=admission.document_id,
        source=SOURCE,
        processing_token="superseded-run",
    )

    with pytest.raises(ProcessingLeaseLostError):
        await ParsingStage(parser).execute(context)

    with ingest_db.session_scope() as session:
        doc = session.get(Document, admission.document_id)
        assert doc.markdown is None
        assert doc.title == "Untitled"


async def test_a_failed_reprocess_keeps_the_published_version(
    orchestrator, ingest_db, vectors, monkeypatch
):
    first = await orchestrator.process(SOURCE)
    document_id = first["document_id"]
    before = snapshot(ingest_db, document_id)
    vectors_before = vectors.rows()

    def interrupt(self, document_id):
        raise DatabaseError("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(DocumentCRUD, "clear_chunks", interrupt)
        result = await orchestrator.process(document_id, reprocess_mode="from_chunking")

    assert result["success"] is False
    assert result["errors"]["database_persistence"] == "publication interrupted"
    after = snapshot(ingest_db, document_id)
    assert after.uuids == before.uuids
    assert after.texts == before.texts
    assert after.metadata["table_of_contents"] == before.metadata["table_of_contents"]
    assert vectors.rows() == vectors_before
    assert after.status == DocumentStatus.COMPLETED
    assert claim_of(ingest_db, document_id) is None
    with ingest_db.session_scope() as session:
        details = session.get(Document, document_id).status_details
    assert "previous version is still published" in details
    assert "publication interrupted" in details


async def test_reprocess_from_embedding_keeps_chunk_ids(orchestrator, ingest_db, vectors):
    first = await orchestrator.process(SOURCE)
    document_id = first["document_id"]
    before = snapshot(ingest_db, document_id)

    result = await orchestrator.process(document_id, reprocess_mode="from_embedding")

    assert result["success"] is True, result["errors"]
    after = snapshot(ingest_db, document_id)
    assert after.uuids == before.uuids
    assert after.texts == before.texts
    assert_published_in_step(after, vectors, document_id)


async def test_a_reprocess_removes_vectors_older_runs_left_behind(orchestrator, ingest_db, vectors):
    first = await orchestrator.process(SOURCE)
    document_id = first["document_id"]
    with ingest_db.session_scope() as session:
        session.execute(
            text(
                "INSERT INTO fake_vectors (custom_id, document_id, document) "
                "VALUES ('orphan-from-an-old-run', :document_id, 'stale text')"
            ),
            {"document_id": document_id},
        )

    result = await orchestrator.process(document_id, reprocess_mode="from_chunking")

    assert result["success"] is True, result["errors"]
    state = snapshot(ingest_db, document_id)
    assert_published_in_step(state, vectors, document_id)
    with ingest_db.session_scope() as session:
        assert sorted(session.scalars(select(Chunk.uuid)).all()) == sorted(state.uuids)
