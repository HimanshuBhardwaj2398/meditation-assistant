"""
Pipeline stages for document ingestion.

Each stage declares its dependencies. Stages receive a PipelineContext and
return an updated context. A stage that cannot finish raises; the pipeline
orchestrator is the one place that turns an exception into a FAILED stage.
"""

import logging
import uuid as uuid_lib
from typing import TYPE_CHECKING, List

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
from ingestion.chunking import Config as ChunkingConfig
from ingestion.chunking import MarkdownChunker
from ingestion.embed import VectorStoreManager
from ingestion.parsing import ParserFactory

if TYPE_CHECKING:
    from db.crud import DocumentCRUD
    from db.schema import Document

logger = logging.getLogger(__name__)

# Source fields stored on the document (doc_metadata).
PROVENANCE_KEYS = (
    "source",
    "uid",
    "author_uid",
    "lang",
    "nikaya",
    "nikaya_name",
    "nikaya_english",
    "reading_url",
)

# The subset copied onto every chunk and vector, for retrieval filtering.
CHUNK_TAG_KEYS = ("source", "uid", "author_uid", "lang", "nikaya", "nikaya_name")


def source_provenance(metadata: dict) -> dict:
    """Source fields shared by freshly parsed and reprocessed documents."""
    return {key: metadata[key] for key in PROVENANCE_KEYS if key in metadata}


def _load_owned_document(doc_crud: "DocumentCRUD", context: PipelineContext) -> "Document":
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


# ============================================================================
# STAGE 1: PARSING
# ============================================================================


class ParsingStage(PipelineStage):
    """
    Stage 1: Parse source document into markdown.

    Dependencies: None
    Input: context.source (file path or URL)
    Output: context.parsed_content, context.title
    """

    def __init__(self, parser_factory: ParserFactory):
        """
        Initialize parsing stage.

        Args:
            parser_factory: Factory for creating appropriate parsers
        """
        self.parser_factory = parser_factory

    @property
    def name(self) -> str:
        return StageName.PARSING

    @property
    def required_stages(self) -> List[str]:
        return []  # No dependencies

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Parse source document and update context.

        Args:
            context: Pipeline context with source

        Returns:
            Context with parsed_content and title populated

        Raises:
            ParsingError: If parsing fails
        """
        from db.crud import DocumentCRUD
        from db.database import session_scope

        if not context.source:
            raise ParsingError("No source provided in context")

        logger.info(f"Parsing source: {context.source}")

        # Use ParserFactory to automatically select parser
        result = self.parser_factory.parse(context.source)

        logger.info(f"Successfully parsed: {context.source} (title: {result.title})")

        metadata = result.metadata or {}
        title = context.title or result.title or "Untitled"
        tags = metadata.get("tags")
        structured = source_provenance(metadata)

        # Store the markdown so a retry can resume from it, with the parsed
        # title and source tags/metadata.
        if context.document_id:
            with session_scope() as session:
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


class ChunkingStage(PipelineStage):
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

    async def execute(self, context: PipelineContext) -> PipelineContext:
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

        logger.info(f"Chunking document: {context.title}")

        chunker = MarkdownChunker(
            text=context.parsed_content, config=self.chunking_config, title=context.title
        )
        chunks, stats = await chunker.chunk()

        logger.info(
            f"Chunking complete: {stats.total_chunks} chunks in "
            f"{stats.processing_time:.2f}s (avg {stats.avg_chunk_size:.0f} words)"
        )

        return context.with_update(chunks=chunks).mark_stage_completed(self.name)


# ============================================================================
# STAGE 3: EMBEDDING
# ============================================================================


class EmbeddingStage(PipelineStage):
    """
    Stage 3: Compute vectors for chunks.

    Dependencies: chunking
    Input: context.chunks, context.document_id, context.title
    Output: Chunks with UUIDs and vectors ready for persistence
    """

    def __init__(self, vector_store_manager: VectorStoreManager, *, vector_only: bool = False):
        self.vector_store_manager = vector_store_manager
        self.vector_only = vector_only

    @property
    def name(self) -> str:
        return StageName.EMBEDDING

    @property
    def required_stages(self) -> List[str]:
        return [StageName.CHUNKING]

    async def execute(self, context: PipelineContext) -> PipelineContext:
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

        logger.info(f"Embedding {len(context.chunks)} chunks")

        # Enrich copies, not the shared chunk objects — the input context
        # must stay unchanged (PipelineContext immutability is shallow).
        enriched_chunks: List[LangchainDocument] = []
        for chunk in context.chunks:
            metadata = dict(chunk.metadata)
            if context.document_id is not None:
                metadata.setdefault("document_id", context.document_id)
                metadata.setdefault("original_doc_id", context.document_id)
            if context.title:
                metadata.setdefault("source_title", context.title)
            # Tag each chunk with source/nikaya metadata for retrieval filtering
            for key in CHUNK_TAG_KEYS:
                if key in context.source_metadata:
                    metadata.setdefault(key, context.source_metadata[key])
            # Keep a chunk's existing UUID (a from_embedding reprocess re-embeds
            # the stored chunks), so chunk IDs and eval labels keyed on them
            # survive a re-embed. New chunks get a fresh one.
            chunk_uuid = metadata.get("uuid") or str(uuid_lib.uuid4())
            metadata["uuid"] = chunk_uuid
            enriched_chunks.append(
                LangchainDocument(
                    # PGVector uses Document.id as the persisted vector ID.
                    id=chunk_uuid,
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        if self.vector_only:
            vector_ids = self.vector_store_manager.embed_documents(enriched_chunks)
            expected_ids = {chunk.id for chunk in enriched_chunks}
            if len(vector_ids) != len(enriched_chunks) or set(vector_ids) != expected_ids:
                raise EmbeddingError("Stored vector IDs do not match the chunk IDs")
            vectors = []
        else:
            vectors = self.vector_store_manager.compute_embeddings(enriched_chunks)

        logger.info("Successfully embedded %s chunks", len(enriched_chunks))
        return context.with_update(chunks=enriched_chunks, vectors=vectors).mark_stage_completed(
            self.name
        )


# ============================================================================
# STAGE 4: DATABASE PERSISTENCE
# ============================================================================


class DatabasePersistenceStage(PipelineStage):
    """
    Stage 4: Publish vectors and relational metadata together.

    Dependencies: embedding
    Input: context.document_id, context.chunks, context.vectors
    Output: Chunks saved to database, document status updated to COMPLETED
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    @property
    def name(self) -> str:
        return StageName.DATABASE_PERSISTENCE

    @property
    def required_stages(self) -> List[str]:
        return [StageName.EMBEDDING]

    def _build_table_of_contents(self, chunks: List) -> dict:
        """
        Build table of contents from chunk header paths.

        Reads all_header_paths and header_level_map from chunk metadata
        to construct a hierarchical TOC.

        Args:
            chunks: List of LangChain Documents with header path metadata

        Returns:
            Dict with 'entries' (structured list) and 'text' (indented string)
        """
        entries = []
        seen_paths = {}  # path_from_root -> entry_id
        entry_count = 0

        for idx, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            paths = metadata.get("all_header_paths", [])
            level_map = metadata.get("header_level_map", {})

            for path in paths:
                segments = [s.strip() for s in path.split(" > ") if s.strip()]
                if not segments:
                    continue

                parent_path = ""
                for depth, segment in enumerate(segments, start=1):
                    current_path = f"{parent_path} > {segment}" if parent_path else segment

                    if current_path in seen_paths:
                        parent_path = current_path
                        continue

                    level = level_map.get(segment, depth)
                    entry_id = f"h{level}_{entry_count}"
                    parent_id = seen_paths.get(parent_path)

                    entries.append(
                        {
                            "id": entry_id,
                            "level": level,
                            "text": segment,
                            "parent_id": parent_id,
                            "chunk_index": idx,
                            "path_from_root": current_path,
                        }
                    )

                    seen_paths[current_path] = entry_id
                    parent_path = current_path
                    entry_count += 1

        text_lines = []
        for entry in entries:
            indent = "  " * (entry["level"] - 1)
            text_lines.append(f"{indent}{entry['text']}")

        logger.debug(f"Built TOC with {len(entries)} entries")
        return {
            "entries": entries,
            "text": "\n".join(text_lines),
        }

    async def execute(self, context: PipelineContext) -> PipelineContext:
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
        from db.crud import ChunkCRUD, DocumentCRUD
        from db.database import session_scope
        from db.schema import DocumentStatus

        if not context.document_id:
            raise DatabaseError("No document_id in context")

        if not context.chunks:
            raise DatabaseError("No chunks to persist")

        logger.info(f"Persisting {len(context.chunks)} chunks to database")

        with session_scope() as session:
            doc_crud = DocumentCRUD(session)
            chunk_crud = ChunkCRUD(session)

            doc = _load_owned_document(doc_crud, context)

            # Prepare chunk data for database
            chunks_data = []
            for idx, chunk in enumerate(context.chunks):
                chunk_uuid = chunk.id
                if not chunk_uuid or chunk.metadata.get("uuid") != chunk_uuid:
                    raise DatabaseError(f"Chunk {idx} has a missing or inconsistent UUID")

                chunks_data.append(
                    {
                        "uuid": chunk_uuid,
                        "chunk_text": chunk.page_content,
                        "chunk_index": idx,
                        "chunk_metadata": {k: v for k, v in chunk.metadata.items() if k != "uuid"},
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

            logger.info(f"Saved {len(created_chunks)} chunks to database")

            # Build the table of contents from chunk headers, replacing any old one
            toc = self._build_table_of_contents(context.chunks)
            doc_metadata = dict(doc.doc_metadata or {})
            if toc["entries"]:
                doc_metadata["table_of_contents"] = toc
                logger.info(f"Saved table of contents with {len(toc['entries'])} entries")
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
