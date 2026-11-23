"""Reprocessing and parsing contracts, on the SQLite fixtures (see conftest)."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.interfaces import ParseResult, PipelineContext, StageStatus
from db.schema import Document as DbDocument
from ingestion.admission import admit
from ingestion.claim import ReprocessMode
from ingestion.stages import ParsingStage
from tests.ingestion.conftest import PROVENANCE, SOURCE


@pytest.mark.parametrize("mode", [ReprocessMode.FROM_CHUNKING, ReprocessMode.FROM_EMBEDDING])
async def test_reprocessing_preserves_source_metadata_in_vectors(
    orchestrator, parser, vectors, monkeypatch, mode
):
    first = await orchestrator.ingest(SOURCE)
    embedded: list = []
    compute = vectors.compute_embeddings

    def capture(documents):
        embedded.extend(documents)
        return compute(documents)

    monkeypatch.setattr(vectors, "compute_embeddings", capture)

    result = await orchestrator.reprocess(first["document_id"], mode)

    assert result["success"] is True
    assert parser.calls == [SOURCE]  # the reprocess did not parse again
    assert embedded
    for chunk in embedded:
        for key, value in PROVENANCE.items():
            assert chunk.metadata[key] == value
        assert "table_of_contents" not in chunk.metadata


@pytest.mark.parametrize(
    ("supplied", "extracted", "expected"),
    [
        ("My title", "Extracted", "My title"),
        (None, "Extracted", "Extracted"),
        ("My title", None, "My title"),
        (None, None, "Untitled"),
    ],
)
async def test_parsing_title_is_consistent_in_context_and_database(
    ingest_db, supplied, extracted, expected
):
    document_id = admit(SOURCE, database=ingest_db).document_id
    with ingest_db.session_scope() as session:
        session.get(DbDocument, document_id).processing_token = "run"
    factory = MagicMock()
    factory.parse.return_value = ParseResult(content="Content", title=extracted)

    result = await ParsingStage(factory, ingest_db).execute(
        PipelineContext(
            document_id=document_id, source=SOURCE, title=supplied, processing_token="run"
        )
    )

    assert result.stage_results["parsing"] == StageStatus.COMPLETED
    with ingest_db.session_scope() as session:
        stored_title = session.get(DbDocument, document_id).title
    assert result.title == stored_title == expected


@pytest.mark.parametrize(
    ("pipeline_result", "expected_success", "expected_count", "expected_error"),
    [
        (
            {"success": False, "chunk_count": 3, "errors": {"embedding": "provider unavailable"}},
            False,
            0,
            "embedding: provider unavailable",
        ),
        ({"success": False, "chunk_count": 3, "errors": {}}, False, 0, "Pipeline reported failure"),
        ({"success": True, "chunk_count": 3, "errors": {}}, True, 3, None),
    ],
)
async def test_collection_reprocess_propagates_pipeline_outcome(
    monkeypatch,
    pipeline_result,
    expected_success,
    expected_count,
    expected_error,
):
    from services import collection_service as service

    document_crud = MagicMock()
    document_crud.get_document_by_id.return_value = SimpleNamespace(id=7, title="Title")
    chunk_crud = MagicMock()
    chunk_crud.get_chunks_by_document.return_value = []
    monkeypatch.setattr(service, "session_scope", lambda: nullcontext(MagicMock()))
    monkeypatch.setattr(service, "DocumentCRUD", lambda session: document_crud)
    monkeypatch.setattr(service, "ChunkCRUD", lambda session: chunk_crud)
    orchestrator = SimpleNamespace(reprocess=AsyncMock(return_value=pipeline_result))
    monkeypatch.setattr(service, "IngestionOrchestrator", lambda **kwargs: orchestrator)

    result = await service.CollectionService(db_url="postgresql://test/test").reprocess_document(7)

    assert result.success is expected_success
    assert result.new_chunks_created == expected_count
    assert result.error == expected_error
