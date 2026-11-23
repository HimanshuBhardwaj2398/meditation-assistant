"""Reprocess modes: parsing them at the boundary, and the clear_markdown rule."""

from unittest.mock import MagicMock

import pytest

from core.exceptions import PipelineError
from ingestion.embed import VectorStoreConfig
from ingestion.orchestrator import IngestionOrchestrator, ReprocessMode


def _build_orchestrator() -> IngestionOrchestrator:
    """Create orchestrator without touching live DB/vector connections."""
    return IngestionOrchestrator(
        vector_store_config=VectorStoreConfig(
            collection_name="test_collection",
            db_url="postgresql://user:pass@localhost:5432/test_db",
        ),
        vector_store_manager=MagicMock(),
        database=MagicMock(),
    )


@pytest.mark.parametrize("mode", list(ReprocessMode))
def test_parse_accepts_each_mode_name(mode):
    assert ReprocessMode.parse(mode.value) is mode


def test_parse_rejects_an_unknown_mode_listing_the_allowed_ones():
    with pytest.raises(PipelineError, match="Invalid reprocess mode 'not_a_mode'.*from_chunking"):
        ReprocessMode.parse("not_a_mode")


@pytest.mark.parametrize("mode", [ReprocessMode.FROM_CHUNKING, ReprocessMode.FROM_EMBEDDING])
async def test_clear_markdown_requires_full_mode(mode):
    with pytest.raises(PipelineError, match="clear_markdown is only supported"):
        await _build_orchestrator().reprocess(1, mode, clear_markdown=True)


async def test_the_service_rejects_clear_markdown_without_a_mode(monkeypatch):
    from services import ingestion_service

    monkeypatch.setattr(ingestion_service, "get_orchestrator", lambda name: _build_orchestrator())

    with pytest.raises(PipelineError, match="clear_markdown is only supported"):
        await ingestion_service.process_document_by_id_async(1, clear_markdown=True)
