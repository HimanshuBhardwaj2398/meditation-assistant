"""One exception hierarchy, rooted in core.exceptions.

ingestion/embed.py historically defined its own VectorStoreError /
EmbeddingError / DatabaseConnectionError. The local EmbeddingError
shadowed core's: EmbeddingStage caught the core class while the vector
store manager raised the embed one, so the specific handler never fired.
These tests pin the unification.
"""

import pytest
from langchain.schema import Document

from core.interfaces import PipelineContext, PipelineStage, StageStatus
from ingestion.stages import EmbeddingStage


def test_embed_embedding_error_is_core_embedding_error():
    from core.exceptions import EmbeddingError as CoreEmbeddingError
    from ingestion.embed import EmbeddingError as EmbedEmbeddingError

    assert EmbedEmbeddingError is CoreEmbeddingError


def test_vector_store_errors_are_rooted_in_core():
    from core.exceptions import MeditationDBError, VectorStoreError
    from ingestion.embed import VectorStoreError as EmbedVectorStoreError

    assert EmbedVectorStoreError is VectorStoreError
    assert issubclass(VectorStoreError, MeditationDBError)


def test_database_connection_error_is_a_vector_store_error():
    from core.exceptions import DatabaseConnectionError, VectorStoreError

    assert issubclass(DatabaseConnectionError, VectorStoreError)


async def test_embedding_stage_lets_manager_errors_through():
    """The manager's EmbeddingError is core's class, so it reaches the pipeline intact."""
    from core.exceptions import EmbeddingError as CoreEmbeddingError

    class ExplodingManager:
        def compute_embeddings(self, documents):
            from ingestion.embed import EmbeddingError

            raise EmbeddingError("vector store down")

    stage = EmbeddingStage(ExplodingManager())
    context = PipelineContext(chunks=[Document(page_content="x", metadata={})])

    with pytest.raises(CoreEmbeddingError, match="vector store down"):
        await stage.execute(context)


async def test_pipeline_keeps_domain_messages_and_names_unexpected_errors():
    """A MeditationDBError is recorded verbatim; anything else with its type."""
    from core.exceptions import EmbeddingError
    from ingestion.pipeline import PipelineOrchestrator

    class RaisingStage(PipelineStage):
        def __init__(self, stage_name, error):
            self._name = stage_name
            self._error = error

        @property
        def name(self):
            return self._name

        async def execute(self, context):
            raise self._error

    pipeline = PipelineOrchestrator(
        [
            RaisingStage("embedding", EmbeddingError("vector store down")),
            RaisingStage("other", ValueError("boom")),
        ]
    )

    result = await pipeline.execute(PipelineContext())

    assert result.stage_results["embedding"] == StageStatus.FAILED
    assert result.error_messages == {
        "embedding": "vector store down",
        "other": "Unexpected ValueError: boom",
    }
