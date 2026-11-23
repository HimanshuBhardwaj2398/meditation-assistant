"""
Pipeline stages for document ingestion.

Each stage declares its dependencies. Stages receive a PipelineContext and
return an updated context. A stage that cannot finish raises; the pipeline
orchestrator is the one place that turns an exception into a FAILED stage.
"""

import asyncio
import logging
import uuid as uuid_lib
from abc import abstractmethod
from typing import List

from langchain_core.documents import Document as LangchainDocument

from core.exceptions import (
    ChunkingError,
    DatabaseError,
    DocumentNotFoundError,
    EmbeddingError,
    ParsingError,
    ProcessingLeaseLostError,
)
from core.interfaces import PipelineContext, PipelineStage, StageName
from db.crud import ChunkCRUD, DocumentCRUD
from db.database import Database
from db.schema import Document, DocumentStatus
from ingestion.chunking import ChunkingConfig, MarkdownChunker
from ingestion.embed import VectorStoreManager
from ingestion.markdown_utils import UNTITLED
from ingestion.metadata import CHUNK_UUID_KEY, chunk_tags, source_provenance
from ingestion.parsing import ParserFactory
from ingestion.toc import build_table_of_contents

logger = logging.getLogger(__name__)


def _load_owned_document(doc_crud: DocumentCRUD, context: PipelineContext) -> Document:
    """Lock the run's document row and check the run still holds its claim.

    Every stage that writes to the document goes through this, so a run that
    another run took over cannot change it. The row lock keeps the check and
    the writes after it in one step.

    Raises:
        DocumentNotFoundError: No such document.
        ProcessingLeaseLostError: Another run has taken the document over.
    """
    assert context.document_id is not None
    doc = doc_crud.get_document_for_update(context.document_id)
    if doc is None:
        raise DocumentNotFoundError(f"Document {context.document_id} not found")
    if context.processing_token is None or doc.processing_token != context.processing_token:
        raise ProcessingLeaseLostError(
            f"Document {context.document_id}: this run no longer holds its processing "
            "claim; another run took it over"
        )
    return doc


class _BlockingStage(PipelineStage):
    """A stage whose work blocks: network calls, models, database round trips.

    ``execute`` runs ``run`` in a worker thread, so the event loop stays free
    and documents a caller runs concurrently do not wait on each other.
    """

    async def execute(self, context: PipelineContext) -> PipelineContext:
        return await asyncio.to_thread(self.run, context)

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Do the stage's work; raise if it cannot finish."""


# ============================================================================
# STAGE 1: PARSING
# ============================================================================


class ParsingStage(_BlockingStage):
    """
    Stage 1: Parse source document into markdown.

    Dependencies: None
    Input: context.source (file path or URL)
    Output: context.parsed_content, context.title
    """

    def __init__(self, parser_factory: ParserFactory, database: Database):
        self.parser_factory = parser_factory
        self.database = database

    @property
    def name(self) -> str:
        return StageName.PARSING

    @property
    def required_stages(self) -> List[str]:
        return []  # No dependencies

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Parse source document and update context.

        Args:
            context: Pipeline context with source

        Returns:
            Context with parsed_content and title populated

        Raises:
            ParsingError: If parsing fails
        """
        if not context.source:
            raise ParsingError("No source provided in context")

        logger.info("Parsing source: %s", context.source)

        # Use ParserFactory to automatically select parser
        result = self.parser_factory.parse(context.source)

        logger.info("Successfully parsed: %s (title: %s)", context.source, result.title)

        metadata = result.metadata or {}
        title = context.title or result.title or UNTITLED
        tags = metadata.get("tags")
        structured = source_provenance(metadata)

        # Store the markdown so a retry can resume from it, with the parsed
        # title and source tags/metadata.
        if context.document_id:
            with self.database.session_scope() as session:
                doc = _load_owned_document(DocumentCRUD(session), context)
                doc.markdown = result.content
                doc.title = title
                if tags:
                    existing = list(doc.tags or [])
                    doc.tags = existing + [t for t in tags if t not in existing]
                if structured:
                    doc.doc_metadata = {**(doc.doc_metadata or {}), **structured}
            logger.info("Stored markdown for document %s", context.document_id)

        # Update context with parsed results (source_metadata tags the chunks later)
        return context.with_update(
            parsed_content=result.content,
            title=title,
            source_metadata=metadata,
        ).mark_stage_completed(self.name)


# ============================================================================
# STAGE 2: CHUNKING
# ============================================================================


class ChunkingStage(_BlockingStage):
    """
    Stage 2: Chunk parsed markdown into semantic segments.

    Dependencies: parsing
    Input: context.parsed_content, context.title
    Output: context.chunks
    """

    def __init__(self, chunking_config: ChunkingConfig):
        """
        Initialize chunking stage.

        Args:
            chunking_config: Configuration for chunking behavior
        """
        self.chunking_config = chunking_config

    @property
    def name(self) -> str:
        return StageName.CHUNKING

    @property
    def required_stages(self) -> List[str]:
        return [StageName.PARSING]

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Chunk parsed content into semantic segments.

        Args:
            context: Pipeline context with parsed content

        Returns:
            Context with chunks populated

        Raises:
            ChunkingError: If chunking fails
        """
        if not context.parsed_content:
            raise ChunkingError("No parsed content available for chunking")

        logger.info("Chunking document: %s", context.title)

        chunker = MarkdownChunker(
            text=context.parsed_content, config=self.chunking_config, title=context.title
        )
        chunks, stats = chunker.chunk()

        logger.info(
            "Chunking complete: %s chunks in %.2fs (avg %.0f words)",
            stats.total_chunks,
            stats.processing_time,
            stats.avg_chunk_size,
        )

        return context.with_update(chunks=chunks).mark_stage_completed(self.name)


# ============================================================================
# STAGE 3: EMBEDDING
# ============================================================================


class EmbeddingStage(_BlockingStage):
    """
    Stage 3: Compute vectors for chunks.

    Dependencies: chunking
    Input: context.chunks, context.document_id, context.title
    Output: Chunks with UUIDs and vectors ready for persistence
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    @property
    def name(self) -> str:
        return StageName.EMBEDDING

    @property
    def required_stages(self) -> List[str]:
        return [StageName.CHUNKING]

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Compute vectors and attach the chunk IDs used by both tables.

        Args:
            context: Pipeline context with chunks

        Returns:
            Context marked as embedding completed

        Raises:
            EmbeddingError: If embedding fails
        """
        if not context.chunks:
            raise EmbeddingError("No chunks available for embedding")

        logger.info("Embedding %s chunks", len(context.chunks))

        # Enrich copies, not the shared chunk objects — the input context
        # must stay unchanged (PipelineContext immutability is shallow).
        tags = chunk_tags(context.document_id, context.title, context.source_metadata)
        enriched_chunks: List[LangchainDocument] = []
        for chunk in context.chunks:
            # Keys the chunk already has win (a reprocess keeps stored values).
            metadata = {**tags, **chunk.metadata}
            # Keep a chunk's existing UUID (a from_embedding reprocess re-embeds
            # the stored chunks), so chunk IDs and eval labels keyed on them
            # survive a re-embed. New chunks get a fresh one.
            chunk_uuid = metadata.get(CHUNK_UUID_KEY) or str(uuid_lib.uuid4())
            metadata[CHUNK_UUID_KEY] = chunk_uuid
            enriched_chunks.append(
                LangchainDocument(
                    # PGVector uses Document.id as the persisted vector ID.
                    id=chunk_uuid,
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        vectors = self.vector_store_manager.compute_embeddings(enriched_chunks)

        logger.info("Successfully embedded %s chunks", len(enriched_chunks))
        return context.with_update(chunks=enriched_chunks, vectors=vectors).mark_stage_completed(
            self.name
        )


# ============================================================================
# STAGE 4: DATABASE PERSISTENCE
# ============================================================================


class DatabasePersistenceStage(_BlockingStage):
    """
    Stage 4: Publish vectors and relational metadata together.

    Dependencies: embedding
    Input: context.document_id, context.chunks, context.vectors
    Output: Chunks saved to database, document status updated to COMPLETED
    """

    def __init__(self, vector_store_manager: VectorStoreManager, database: Database):
        self.vector_store_manager = vector_store_manager
        self.database = database

    @property
    def name(self) -> str:
        return StageName.DATABASE_PERSISTENCE

    @property
    def required_stages(self) -> List[str]:
        return [StageName.EMBEDDING]

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Publish the document: its vectors, chunk rows, table of contents and
        COMPLETED status, in one transaction.

        Whatever the document published before (chunk rows and vectors) is
        deleted in the same transaction, so a reprocess is a replacement: if
        it fails, the previous version stays searchable.

        The run's claim is checked under a row lock first. A run that another
        run took over cannot publish.

        Args:
            context: Pipeline context with embedded chunks

        Returns:
            Context marked as persistence completed

        Raises:
            DatabaseError: If database operations fail
            ProcessingLeaseLostError: If this run no longer holds the claim
        """
        if not context.document_id:
            raise DatabaseError("No document_id in context")

        if not context.chunks:
            raise DatabaseError("No chunks to persist")

        logger.info("Persisting %s chunks to database", len(context.chunks))

        with self.database.session_scope() as session:
            doc_crud = DocumentCRUD(session)
            chunk_crud = ChunkCRUD(session)

            doc = _load_owned_document(doc_crud, context)

            # Prepare chunk data for database
            chunks_data = []
            for idx, chunk in enumerate(context.chunks):
                chunk_uuid = chunk.id
                if not chunk_uuid or chunk.metadata.get(CHUNK_UUID_KEY) != chunk_uuid:
                    raise DatabaseError(f"Chunk {idx} has a missing or inconsistent UUID")

                chunks_data.append(
                    {
                        "uuid": chunk_uuid,
                        "chunk_text": chunk.page_content,
                        "chunk_index": idx,
                        "chunk_metadata": {
                            k: v for k, v in chunk.metadata.items() if k != CHUNK_UUID_KEY
                        },
                    }
                )

            # Replace the previous version, if there is one.
            previous_ids = [row.uuid for row in chunk_crud.get_chunks_by_document(doc.id)]
            removed_vectors = self.vector_store_manager.delete_document_vectors(
                session, doc.id, previous_ids
            )
            removed_chunks = chunk_crud.delete_chunks_by_document(doc.id)
            if removed_vectors or removed_chunks:
                logger.info(
                    "Replacing document %s: removed %s chunk rows and %s vectors",
                    doc.id,
                    removed_chunks,
                    removed_vectors,
                )

            self.vector_store_manager.stage_vectors(
                session, context.chunks, context.vectors, context.vector_collection
            )

            # Save chunks to database
            created_chunks = chunk_crud.create_chunks_batch(
                document_id=context.document_id,
                chunks_data=chunks_data,
            )

            logger.info("Saved %s chunks to database", len(created_chunks))

            # Build the table of contents from chunk headers, replacing any old one
            toc = build_table_of_contents(context.chunks)
            doc_metadata = dict(doc.doc_metadata or {})
            if toc["entries"]:
                doc_metadata["table_of_contents"] = toc
                logger.info("Saved table of contents with %s entries", len(toc["entries"]))
            else:
                doc_metadata.pop("table_of_contents", None)
            doc_crud.update_doc_metadata(context.document_id, doc_metadata, merge=False)

            # Update document status to COMPLETED and release the claim
            doc_crud.update_status(
                document_id=context.document_id,
                status=DocumentStatus.COMPLETED,
                status_details=f"Successfully processed {len(created_chunks)} chunks",
            )
            doc.processing_token = None

            # Clear temporary chunk storage
            doc_crud.clear_chunks(context.document_id)

        logger.info(
            "Document %s committed with %s chunks", context.document_id, len(created_chunks)
        )
        return context.with_update(vectors=[]).mark_stage_completed(self.name)
