"""Tests for embedding-stage integrity checks."""

import pytest
from langchain.schema import Document

from core.exceptions import EmbeddingError
from core.interfaces import PipelineContext, StageStatus
from ingestion.stages import EmbeddingStage


class RecordingEmbeddingManager:
    """Fake embedding provider that records the chunks passed to it."""

    def __init__(self):
        self.last_documents = []

    def compute_embeddings(self, documents):
        self.last_documents = documents
        return [[1.0, 0.0] for _ in documents]


class MissingOneIdManager:
    """Fake vector store that drops one embedding ID."""

    def embed_documents(self, documents):
        return [doc.id for doc in documents[1:]]


class DifferentIdsManager:
    """Fake vector store that returns non-matching IDs."""

    def embed_documents(self, documents):
        return [f"vec-{idx}" for idx, _ in enumerate(documents)]


@pytest.mark.asyncio
async def test_embedding_stage_sets_uuid_and_document_metadata():
    manager = RecordingEmbeddingManager()
    stage = EmbeddingStage(manager)
    context = PipelineContext(
        document_id=42,
        title="Meditation Manual",
        chunks=[
            Document(page_content="chunk one", metadata={}),
            Document(page_content="chunk two", metadata={}),
        ],
    )

    result = await stage.execute(context)

    assert result.stage_results["embedding"] == StageStatus.COMPLETED
    assert result.vectors == [[1.0, 0.0], [1.0, 0.0]]
    for doc in manager.last_documents:
        assert doc.id is not None
        assert doc.metadata["uuid"] == doc.id
        assert doc.metadata["document_id"] == 42
        assert doc.metadata["original_doc_id"] == 42
        assert doc.metadata["source_title"] == "Meditation Manual"


@pytest.mark.asyncio
async def test_embedding_stage_fails_when_ids_missing():
    stage = EmbeddingStage(MissingOneIdManager(), vector_only=True)
    context = PipelineContext(
        chunks=[
            Document(page_content="chunk one", metadata={}),
            Document(page_content="chunk two", metadata={}),
        ]
    )

    with pytest.raises(EmbeddingError, match="Stored vector IDs do not match"):
        await stage.execute(context)


@pytest.mark.asyncio
async def test_embedding_stage_fails_when_ids_do_not_match():
    stage = EmbeddingStage(DifferentIdsManager(), vector_only=True)
    context = PipelineContext(
        chunks=[
            Document(page_content="chunk one", metadata={}),
            Document(page_content="chunk two", metadata={}),
        ]
    )

    with pytest.raises(EmbeddingError, match="Stored vector IDs do not match"):
        await stage.execute(context)
