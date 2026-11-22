"""CRUD operations for documents and chunks.

Transaction contract: methods flush (IDs materialize, constraint errors
surface early) but never commit — the enclosing ``session_scope`` owns the
transaction, so multi-step operations stay atomic.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from . import schema


def get_all_collections(session: Session) -> List[str]:
    """
    Fetch all collection names from langchain_pg_collection table.

    Args:
        session: Database session

    Returns:
        List of collection names sorted alphabetically
    """
    result = session.execute(text("SELECT name FROM langchain_pg_collection ORDER BY name"))
    return [row[0] for row in result.fetchall()]


class DocumentCRUD:
    """CRUD operations for Document model."""

    def __init__(self, db: Session):
        self.db = db

    def create_document(
        self,
        title: str,
        file_path: str,
        markdown: Optional[str] = None,
        tags: Optional[List[str]] = None,
        doc_metadata: Optional[Dict] = None,
        description: Optional[str] = None,
        status: schema.DocumentStatus = schema.DocumentStatus.PENDING,
    ) -> schema.Document:
        """
        Creates a new document with source tracking.

        Args:
            title: Document title
            file_path: Source file path or URL (required for pipeline resume)
            markdown: Parsed content (optional during creation)
            tags: List of tags
            doc_metadata: Additional metadata
            description: Document description
            status: Initial status (defaults to PENDING)

        Returns:
            Created document instance
        """
        db_document = schema.Document(
            title=title,
            file_path=file_path,
            markdown=markdown,
            tags=tags or [],
            doc_metadata=doc_metadata or {},
            description=description,
            status=status,
        )
        self.db.add(db_document)
        self.db.flush()
        return db_document

    def get_document_by_id(self, document_id: int) -> Optional[schema.Document]:
        """
        Get a document by its ID.

        Args:
            document_id: ID of the document

        Returns:
            Document instance or None if not found
        """
        return self.db.query(schema.Document).filter(schema.Document.id == document_id).first()

    def get_all_documents(self) -> List[schema.Document]:
        """
        Get all documents.

        Returns:
            List of all documents
        """
        return self.db.query(schema.Document).all()

    def update_status(
        self,
        document_id: int,
        status: schema.DocumentStatus,
        status_details: Optional[str] = None,
    ) -> Optional[schema.Document]:
        """
        Update document status with optional details.

        Args:
            document_id: ID of document to update
            status: New status value
            status_details: Optional error message or status info

        Returns:
            Updated document or None if not found
        """
        doc = self.get_document_by_id(document_id)
        if doc:
            doc.status = status
            doc.status_details = status_details
            self.db.flush()
        return doc

    def insert_if_absent(
        self,
        file_path: str,
        title: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        doc_metadata: Optional[Dict] = None,
    ) -> Optional[int]:
        """
        Insert a PENDING document unless one with this file_path exists.

        A single INSERT ... ON CONFLICT DO NOTHING, so two callers admitting
        the same source can never both insert, and neither gets an
        IntegrityError that would abort its transaction.

        Returns:
            The new document's id, or None when the source was already there
        """
        dialect = self.db.get_bind().dialect.name
        insert = sqlite.insert if dialect == "sqlite" else postgresql.insert
        statement = (
            insert(schema.Document)
            .values(
                title=title,
                file_path=file_path,
                description=description,
                tags=tags or [],
                doc_metadata=doc_metadata or {},
                status=schema.DocumentStatus.PENDING,
            )
            .on_conflict_do_nothing(index_elements=["file_path"])
            .returning(schema.Document.id)
        )
        return self.db.execute(statement).scalar_one_or_none()

    def claim_for_processing(
        self,
        document_id: int,
        token: str,
        stale_before: datetime,
        status: Optional[schema.DocumentStatus] = None,
        status_details: Optional[str] = None,
    ) -> bool:
        """
        Take a document's processing claim if no live run holds it.

        A claim is live while processing_token is set and the row was written
        after stale_before. Each status write by the owning run refreshes
        updated_at, so a claim lapses only once its run stops writing.

        Args:
            status: Also set the run's starting status, in the same UPDATE

        Returns:
            True if this call took the claim
        """
        claimed = (
            self.db.query(schema.Document)
            .filter(
                schema.Document.id == document_id,
                or_(
                    schema.Document.processing_token.is_(None),
                    schema.Document.updated_at < stale_before,
                ),
            )
            .update(
                {
                    schema.Document.processing_token: token,
                    schema.Document.updated_at: func.now(),
                    **(
                        {
                            schema.Document.status: status,
                            schema.Document.status_details: status_details,
                        }
                        if status is not None
                        else {}
                    ),
                },
                synchronize_session=False,
            )
        )
        return claimed == 1

    def set_status_if_owner(
        self,
        document_id: int,
        token: str,
        status: schema.DocumentStatus,
        status_details: Optional[str] = None,
        release: bool = False,
    ) -> bool:
        """
        Write a run's status only while that run still holds the claim.

        One UPDATE, no read first. It also refreshes updated_at, which keeps
        the claim alive.

        Args:
            release: Also drop the claim (the run is finished)

        Returns:
            False if another run has taken the document over
        """
        values: Dict[Any, Any] = {
            schema.Document.status: status,
            schema.Document.status_details: status_details,
            schema.Document.updated_at: func.now(),
        }
        if release:
            values[schema.Document.processing_token] = None
        updated = (
            self.db.query(schema.Document)
            .filter(
                schema.Document.id == document_id,
                schema.Document.processing_token == token,
            )
            .update(values, synchronize_session=False)
        )
        return updated == 1

    def get_document_for_update(self, document_id: int) -> Optional[schema.Document]:
        """
        Load a document and lock its row until the transaction ends.

        Publishing reads the claim token through this lock, so a run that
        takes the document over cannot slip in between the check and the
        writes. (SQLite, used in tests, has no row locks and ignores it.)
        """
        return (
            self.db.query(schema.Document)
            .filter(schema.Document.id == document_id)
            .with_for_update()
            .first()
        )

    def update_markdown(
        self,
        document_id: int,
        markdown: str,
    ) -> Optional[schema.Document]:
        """
        Update document markdown content after parsing.

        Args:
            document_id: ID of document
            markdown: Parsed markdown content

        Returns:
            Updated document or None if not found
        """
        doc = self.get_document_by_id(document_id)
        if doc:
            doc.markdown = markdown
            self.db.flush()
        return doc

    def update_doc_metadata(
        self,
        document_id: int,
        metadata_updates: Dict[str, Any],
        merge: bool = True,
    ) -> Optional[schema.Document]:
        """
        Update document metadata.

        Args:
            document_id: ID of document
            metadata_updates: Dictionary of metadata to add/update
            merge: If True, merge with existing metadata. If False, replace.

        Returns:
            Updated document or None if not found
        """
        doc = self.get_document_by_id(document_id)
        if doc:
            if merge and doc.doc_metadata:
                doc.doc_metadata = {**doc.doc_metadata, **metadata_updates}
            else:
                doc.doc_metadata = metadata_updates
            self.db.flush()
        return doc

    def clear_chunks(
        self,
        document_id: int,
    ) -> Optional[schema.Document]:
        """
        Clear temporary chunk storage after successful embedding.

        Args:
            document_id: ID of document

        Returns:
            Updated document or None if not found
        """
        doc = self.get_document_by_id(document_id)
        if doc:
            doc.chunks = None
            self.db.flush()
        return doc

    def get_documents_by_status(
        self,
        status: schema.DocumentStatus,
    ) -> List[schema.Document]:
        """
        Get all documents with specific status.

        Args:
            status: Status to filter by

        Returns:
            List of matching documents
        """
        return (
            self.db.query(schema.Document)
            .filter(schema.Document.status == status)
            .order_by(schema.Document.created_at.desc())
            .all()
        )

    def get_failed_documents(self) -> List[schema.Document]:
        """
        Get all documents with FAILED status.

        Returns:
            List of failed documents with status_details
        """
        return self.get_documents_by_status(schema.DocumentStatus.FAILED)

    def get_document_by_file_path(self, file_path: str) -> Optional[schema.Document]:
        """
        Get a document by its file path or URL.

        Args:
            file_path: Source file path or URL

        Returns:
            Document instance or None if not found
        """
        return self.db.query(schema.Document).filter(schema.Document.file_path == file_path).first()

    def check_duplicate(self, file_path: str) -> Optional[schema.Document]:
        """
        Check if a document with the same source already exists.

        Args:
            file_path: Source file path or URL to check

        Returns:
            Existing document if found, None otherwise
        """
        return self.get_document_by_file_path(file_path)

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and all its associated chunks.

        Args:
            document_id: ID of document to delete

        Returns:
            True if deleted, False if not found
        """
        doc = self.get_document_by_id(document_id)
        if doc:
            self.db.delete(doc)
            self.db.flush()
            return True
        return False


class ChunkCRUD:
    """CRUD operations for Chunk model."""

    def __init__(self, db: Session):
        self.db = db

    def create_chunks_batch(
        self,
        document_id: int,
        chunks_data: List[Dict[str, Any]],
    ) -> List[schema.Chunk]:
        """
        Create multiple chunks for a document.

        Args:
            document_id: Parent document ID
            chunks_data: List of dicts with chunk information:
                - uuid: UUID from LangChain embedding
                - chunk_text: Text content
                - chunk_index: Position in document
                - chunk_metadata: Optional metadata dict

        Returns:
            List of created Chunk instances
        """
        chunks = []
        for data in chunks_data:
            chunk = schema.Chunk(
                uuid=data["uuid"],
                document_id=document_id,
                chunk_text=data["chunk_text"],
                chunk_index=data["chunk_index"],
                chunk_metadata=data.get("chunk_metadata", {}),
            )
            chunks.append(chunk)

        self.db.add_all(chunks)
        self.db.flush()

        return chunks

    def get_chunks_by_document(
        self,
        document_id: int,
    ) -> List[schema.Chunk]:
        """
        Get all chunks for a document, ordered by index.

        Args:
            document_id: Parent document ID

        Returns:
            List of chunks ordered by chunk_index
        """
        return (
            self.db.query(schema.Chunk)
            .filter(schema.Chunk.document_id == document_id)
            .order_by(schema.Chunk.chunk_index)
            .all()
        )

    def count_chunks_by_document(self, document_id: int) -> int:
        """Number of chunk rows a document has."""
        return (
            self.db.query(func.count(schema.Chunk.id))
            .filter(schema.Chunk.document_id == document_id)
            .scalar()
        )

    def get_chunk_by_uuid(
        self,
        uuid: str,
    ) -> Optional[schema.Chunk]:
        """
        Get chunk by UUID (for linking with vector search results).

        Args:
            uuid: Chunk UUID

        Returns:
            Chunk instance or None
        """
        return self.db.query(schema.Chunk).filter(schema.Chunk.uuid == uuid).first()

    def update_chunk_metadata(
        self,
        chunk_id: int,
        metadata_updates: Dict[str, Any],
        merge: bool = True,
    ) -> Optional[schema.Chunk]:
        """
        Update metadata for a single chunk.

        Args:
            chunk_id: ID of chunk to update
            metadata_updates: Dictionary of metadata to add/update
            merge: If True, merge with existing metadata. If False, replace.

        Returns:
            Updated chunk or None if not found
        """
        chunk = self.db.query(schema.Chunk).filter(schema.Chunk.id == chunk_id).first()
        if chunk:
            if merge and chunk.chunk_metadata:
                chunk.chunk_metadata = {**chunk.chunk_metadata, **metadata_updates}
            else:
                chunk.chunk_metadata = metadata_updates
            self.db.flush()
        return chunk

    def delete_chunks_by_document(self, document_id: int) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Parent document ID

        Returns:
            Number of chunks deleted
        """
        deleted = (
            self.db.query(schema.Chunk).filter(schema.Chunk.document_id == document_id).delete()
        )
        self.db.flush()
        return deleted
