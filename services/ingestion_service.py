"""
Ingestion service - business logic for document processing.

Handles document ingestion, processing, and orchestration.
"""

import asyncio
from typing import Callable, Optional

from core.interfaces import StageStatus
from ingestion.embed import VectorStoreConfig
from ingestion.orchestrator import IngestionOrchestrator, IngestionResult


def get_orchestrator(collection_name: Optional[str] = None) -> IngestionOrchestrator:
    """Initialize ingestion orchestrator with vector store config.

    Args:
        collection_name: Name of the collection to store embeddings in;
            None means the configured default (VECTOR_COLLECTION_NAME)

    Returns:
        Configured IngestionOrchestrator instance
    """
    config = (
        VectorStoreConfig(collection_name=collection_name)
        if collection_name
        else VectorStoreConfig()
    )
    return IngestionOrchestrator(vector_store_config=config)


async def ingest_document_async(
    source: str,
    title: str,
    description: str,
    doc_type: str,
    category: str,
    tags: list,
    collection_name: Optional[str] = None,
    on_stage_update: Optional[Callable[[str, StageStatus], None]] = None,
) -> IngestionResult:
    """
    Ingest document with metadata.

    Args:
        source: URL or file path to the document
        title: Document title
        description: Document description
        doc_type: Type of document (e.g., ancient_text, lecture)
        category: Primary category (e.g., buddhism)
        tags: List of tags for searchability
        collection_name: Name of the vector store collection
        on_stage_update: Optional callback(stage_name, status) for progress updates

    Returns:
        The orchestrator's result. A source already in the corpus is not
        processed again: the result has ``already_completed=True`` and the
        existing document's id. A source that failed before is resumed.

    Raises:
        DocumentBusyError: Another run is processing the same source
    """
    # Build metadata
    metadata = {}
    if doc_type:
        metadata["type"] = doc_type
    if category:
        metadata["category"] = category

    # Combine tags
    all_tags = tags.copy() if tags else []
    if doc_type:
        all_tags.append(f"type:{doc_type}")
    if category:
        all_tags.append(f"category:{category}")

    orchestrator = get_orchestrator(collection_name)

    # Find or create the document; the metadata (and the collection) apply
    # only to a new one. An existing document keeps its own collection.
    admission = orchestrator.admit(
        source,
        title=title,
        description=description,
        tags=all_tags,
        doc_metadata=metadata,
    )
    return await orchestrator.process(
        source=admission.document_id,
        title=title,
        on_stage_update=on_stage_update,
    )


def ingest_document(
    source: str,
    title: str,
    description: str,
    doc_type: str,
    category: str,
    tags: list,
    collection_name: Optional[str] = None,
    on_stage_update: Optional[Callable[[str, StageStatus], None]] = None,
) -> IngestionResult:
    """Synchronous wrapper for async ingestion."""
    return asyncio.run(
        ingest_document_async(
            source,
            title,
            description,
            doc_type,
            category,
            tags,
            collection_name,
            on_stage_update=on_stage_update,
        )
    )


async def process_document_by_id_async(
    doc_id: int,
    collection_name: Optional[str] = None,
    on_stage_update: Optional[Callable[[str, StageStatus], None]] = None,
    reprocess_mode: Optional[str] = None,
    clear_markdown: bool = False,
) -> IngestionResult:
    """
    Process/resume a document by its ID.

    Args:
        doc_id: Database ID of the document to process
        collection_name: Name of the vector store collection
        reprocess_mode: Optional reprocessing mode for existing docs
        clear_markdown: Clear stored markdown before reprocessing (full mode only)

    Returns:
        dict: Result from orchestrator
    """
    orchestrator = get_orchestrator(collection_name)
    result = await orchestrator.process(
        source=doc_id,
        on_stage_update=on_stage_update,
        reprocess_mode=reprocess_mode,
        clear_markdown=clear_markdown,
    )
    return result


def process_document_by_id(
    doc_id: int,
    collection_name: Optional[str] = None,
    on_stage_update: Optional[Callable[[str, StageStatus], None]] = None,
    reprocess_mode: Optional[str] = None,
    clear_markdown: bool = False,
) -> IngestionResult:
    """Synchronous wrapper for async document processing."""
    return asyncio.run(
        process_document_by_id_async(
            doc_id,
            collection_name,
            on_stage_update=on_stage_update,
            reprocess_mode=reprocess_mode,
            clear_markdown=clear_markdown,
        )
    )
