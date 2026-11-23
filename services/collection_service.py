"""
Collection management service - coordinates app DB and vector store operations.

Ensures atomic consistency between document/chunk tables and vector embeddings.
This service should be used for all delete operations to prevent orphaned embeddings.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from config.settings import DatabaseSettings, VectorSettings
from core.exceptions import DocumentNotFoundError
from db.crud import ChunkCRUD, DocumentCRUD, get_all_collections
from db.database import session_scope
from ingestion.embed import VectorStoreConfig, VectorStoreManager
from ingestion.orchestrator import IngestionOrchestrator, ReprocessMode

logger = logging.getLogger(__name__)


@dataclass
class DeleteResult:
    """Result of a document delete operation."""

    document_id: int
    title: str
    chunks_deleted: int
    embeddings_deleted: int
    success: bool
    error: Optional[str] = None


@dataclass
class ReprocessResult:
    """Result of a document reprocess operation."""

    document_id: int
    title: str
    old_chunks_deleted: int
    old_embeddings_deleted: int
    new_chunks_created: int
    success: bool
    error: Optional[str] = None


class CollectionService:
    """
    Service for managing document collections across app DB and vector store.

    This service ensures atomic operations between:
    - Application database (documents, chunks tables)
    - Vector database (langchain_pg_embedding table)

    Always use this service for delete operations to maintain consistency.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        db_url: Optional[str] = None,
    ):
        """
        Initialize collection service.

        Args:
            collection_name: Name of the vector store collection (defaults to
                VECTOR_COLLECTION_NAME)
            db_url: Database URL (defaults to database settings)
        """
        self.collection_name = collection_name or VectorSettings().collection_name
        self.db_url = db_url or DatabaseSettings().url

        self._vector_store_manager: Optional[VectorStoreManager] = None

    @property
    def vector_store_manager(self) -> VectorStoreManager:
        """Lazy-load vector store manager."""
        if self._vector_store_manager is None:
            config = VectorStoreConfig(
                collection_name=self.collection_name,
                db_url=self.db_url,
            )
            self._vector_store_manager = VectorStoreManager(config)
        return self._vector_store_manager

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    async def delete_document_complete(
        self,
        document_id: int,
    ) -> DeleteResult:
        """
        Delete a document, its chunk rows and all its vectors, in one transaction.

        The document row is locked first, so a run publishing this document at
        the same moment finishes before the delete reads what to remove.
        Vectors are matched by chunk UUID and by the document id they carry,
        which also removes any an older pipeline left without chunk rows. If any
        step fails, nothing is deleted.

        Args:
            document_id: ID of document to delete

        Returns:
            DeleteResult with operation details

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        doc_title = "Unknown"
        try:
            with session_scope() as session:
                doc_crud = DocumentCRUD(session)
                doc = doc_crud.get_document_for_update(document_id)
                if not doc:
                    raise DocumentNotFoundError(f"Document {document_id} not found")
                doc_title = doc.title

                chunk_uuids = [
                    chunk.uuid for chunk in ChunkCRUD(session).get_chunks_by_document(document_id)
                ]
                embeddings_deleted = self.vector_store_manager.delete_document_vectors(
                    session, document_id, chunk_uuids
                )
                doc_crud.delete_document(document_id)  # chunk rows cascade
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Delete failed for document {document_id}; nothing was deleted: {e}")
            return DeleteResult(
                document_id=document_id,
                title=doc_title,
                chunks_deleted=0,
                embeddings_deleted=0,
                success=False,
                error=f"Delete failed; nothing was deleted: {e}",
            )

        logger.info(
            f"Deleted document {document_id} '{doc_title}': "
            f"{len(chunk_uuids)} chunks, {embeddings_deleted} embeddings"
        )
        return DeleteResult(
            document_id=document_id,
            title=doc_title,
            chunks_deleted=len(chunk_uuids),
            embeddings_deleted=embeddings_deleted,
            success=True,
        )

    def delete_document_complete_sync(self, document_id: int) -> DeleteResult:
        """
        Synchronous wrapper for delete_document_complete.

        Args:
            document_id: ID of document to delete

        Returns:
            DeleteResult with operation details
        """
        return asyncio.run(self.delete_document_complete(document_id))

    async def delete_documents_batch(
        self,
        document_ids: List[int],
    ) -> List[DeleteResult]:
        """
        Delete multiple documents atomically.

        Processes each document independently, collecting results.
        Partial failures don't affect other deletions.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            List of DeleteResult for each document
        """
        results = []

        for doc_id in document_ids:
            try:
                result = await self.delete_document_complete(doc_id)
                results.append(result)
            except DocumentNotFoundError:
                results.append(
                    DeleteResult(
                        document_id=doc_id,
                        title="Unknown",
                        chunks_deleted=0,
                        embeddings_deleted=0,
                        success=False,
                        error=f"Document {doc_id} not found",
                    )
                )
            except Exception as e:
                logger.error(f"Unexpected error deleting document {doc_id}: {e}")
                results.append(
                    DeleteResult(
                        document_id=doc_id,
                        title="Unknown",
                        chunks_deleted=0,
                        embeddings_deleted=0,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    def delete_documents_batch_sync(
        self,
        document_ids: List[int],
    ) -> List[DeleteResult]:
        """Synchronous wrapper for delete_documents_batch."""
        return asyncio.run(self.delete_documents_batch(document_ids))

    # =========================================================================
    # REPROCESS OPERATIONS
    # =========================================================================

    async def reprocess_document(
        self,
        document_id: int,
    ) -> ReprocessResult:
        """
        Reprocess a document from its stored markdown: re-chunk and re-embed it.

        Useful when:
        - Chunking strategy has changed
        - Embedding model has been updated
        - Document processing failed partway through

        The run replaces the document in its publish transaction, which deletes
        the old chunk rows and vectors and writes the new ones together. If the
        run fails, the old version stays published, and nothing is deleted.

        Args:
            document_id: ID of document to reprocess

        Returns:
            ReprocessResult with operation details

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        with session_scope() as session:
            doc = DocumentCRUD(session).get_document_by_id(document_id)
            if not doc:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            doc_title = doc.title
            old_chunk_count = len(ChunkCRUD(session).get_chunks_by_document(document_id))

        logger.info(
            f"Reprocessing document {document_id} '{doc_title}' "
            f"with {old_chunk_count} existing chunks"
        )

        try:
            orchestrator = IngestionOrchestrator(
                vector_store_config=VectorStoreConfig(
                    collection_name=self.collection_name,
                    db_url=self.db_url,
                )
            )
            result = await orchestrator.reprocess(document_id, ReprocessMode.FROM_CHUNKING)

            success = bool(result.get("success"))
            new_chunk_count = result.get("chunk_count", 0) if success else 0
            error = None
            if success:
                logger.info(
                    f"Reprocessing complete for document {document_id}: "
                    f"{new_chunk_count} new chunks created"
                )
            else:
                error = (
                    "; ".join(
                        f"{stage}: {message}"
                        for stage, message in (result.get("errors") or {}).items()
                    )
                    or "Pipeline reported failure"
                )
                logger.error("Reprocessing failed for document %s: %s", document_id, error)

            # Old chunks and their vectors are replaced only when the run publishes.
            replaced = old_chunk_count if success else 0
            return ReprocessResult(
                document_id=document_id,
                title=doc_title,
                old_chunks_deleted=replaced,
                old_embeddings_deleted=replaced,
                new_chunks_created=new_chunk_count,
                success=success,
                error=error,
            )

        except Exception as e:
            logger.error(f"Reprocessing failed for document {document_id}: {e}")
            return ReprocessResult(
                document_id=document_id,
                title=doc_title,
                old_chunks_deleted=0,
                old_embeddings_deleted=0,
                new_chunks_created=0,
                success=False,
                error=f"Reprocessing failed: {e}",
            )

    def reprocess_document_sync(self, document_id: int) -> ReprocessResult:
        """
        Synchronous wrapper for reprocess_document.

        Args:
            document_id: ID of document to reprocess

        Returns:
            ReprocessResult with operation details
        """
        return asyncio.run(self.reprocess_document(document_id))

    # =========================================================================
    # COLLECTION OPERATIONS
    # =========================================================================

    def list_collections(self) -> List[str]:
        """
        List all available vector store collections.

        Returns:
            List of collection names sorted alphabetically
        """
        with session_scope() as session:
            return get_all_collections(session)
