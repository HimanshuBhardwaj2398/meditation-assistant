"""Exercise real orchestration/stages with offline database and provider boundaries."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.schema import Document

from core.interfaces import ParseResult, PipelineContext, StageStatus
from db.schema import DocumentStatus
from ingestion.embed import VectorStoreConfig
from ingestion.orchestrator import IngestionOrchestrator
from ingestion.stages import ParsingStage


@pytest.mark.parametrize("mode", ["from_chunking", "from_embedding"])
async def test_reprocessing_preserves_source_metadata_in_vectors(monkeypatch, mode):
    from db import crud, database
    from ingestion import stages

    metadata = {
        "source": "suttacentral",
        "uid": "mn1",
        "author_uid": "sujato",
        "lang": "en",
        "nikaya": "mn",
        "nikaya_name": "majjhima_nikaya",
    }
    doc = SimpleNamespace(
        id=7,
        title="Stored title",
        markdown="Stored markdown",
        file_path="sc:mn1/sujato",
        doc_metadata={**metadata, "table_of_contents": {"entries": []}},
        chunks=None,
        status=DocumentStatus.COMPLETED,
    )
    document_crud = MagicMock()
    document_crud.get_document_by_id.return_value = doc
    chunk_crud = MagicMock()
    chunk_crud.get_chunks_by_document.return_value = [
        SimpleNamespace(uuid="old-id", chunk_text="Stored chunk", chunk_metadata={})
    ]
    monkeypatch.setattr(database, "session_scope", lambda: nullcontext(MagicMock()))
    monkeypatch.setattr(crud, "DocumentCRUD", lambda session: document_crud)
    monkeypatch.setattr(crud, "ChunkCRUD", lambda session: chunk_crud)
    chunker = MagicMock()
    chunker.chunk = AsyncMock(
        return_value=(
            [Document(page_content="Regenerated chunk")],
            SimpleNamespace(total_chunks=1, processing_time=0, avg_chunk_size=2),
        )
    )
    monkeypatch.setattr(stages, "MarkdownChunker", lambda **kwargs: chunker)

    async def persist(self, context):
        return context.mark_stage_completed(self.name)

    monkeypatch.setattr(stages.DatabasePersistenceStage, "execute", persist)
    orchestrator = IngestionOrchestrator(VectorStoreConfig(db_url="postgresql://test/test"))
    orchestrator.parser_factory = MagicMock()
    vector_store = MagicMock()
    vector_store.compute_embeddings.side_effect = lambda docs: [[1.0] for _ in docs]
    orchestrator.vector_store_manager = vector_store

    result = await orchestrator.process(7, reprocess_mode=mode)

    assert result["success"] is True
    orchestrator.parser_factory.parse.assert_not_called()
    embedded = vector_store.compute_embeddings.call_args.args[0]
    assert len(embedded) == 1
    for key, value in metadata.items():
        assert embedded[0].metadata[key] == value
    assert "table_of_contents" not in embedded[0].metadata


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
    monkeypatch,
    supplied,
    extracted,
    expected,
):
    from db import crud, database

    doc = SimpleNamespace(title="Untitled", tags=[], doc_metadata={}, processing_token="run")
    document_crud = MagicMock()
    document_crud.get_document_for_update.return_value = doc
    monkeypatch.setattr(database, "session_scope", lambda: nullcontext(MagicMock()))
    monkeypatch.setattr(crud, "DocumentCRUD", lambda session: document_crud)
    factory = MagicMock()
    factory.parse.return_value = ParseResult(content="Content", title=extracted)

    result = await ParsingStage(factory).execute(
        PipelineContext(document_id=7, source="source", title=supplied, processing_token="run")
    )

    assert result.stage_results["parsing"] == StageStatus.COMPLETED
    assert result.title == doc.title == expected


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
    orchestrator = SimpleNamespace(process=AsyncMock(return_value=pipeline_result))
    monkeypatch.setattr(service, "IngestionOrchestrator", lambda **kwargs: orchestrator)

    result = await service.CollectionService(db_url="postgresql://test/test").reprocess_document(7)

    assert result.success is expected_success
    assert result.new_chunks_created == expected_count
    assert result.error == expected_error
