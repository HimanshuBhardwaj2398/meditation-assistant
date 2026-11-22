"""
Vector store operations for document embedding and retrieval.

This module provides utilities for creating vector stores, embedding documents,
and managing the connection to PostgreSQL with pgvector extension.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_voyageai import VoyageAIEmbeddings
from psycopg2 import errors as pg_errors
from sqlalchemy import or_
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from core.exceptions import DatabaseConnectionError, EmbeddingError, VectorStoreError

__all__ = [
    "VectorStoreConfig",
    "VectorStoreManager",
    "VectorStoreError",
    "DatabaseConnectionError",
    "EmbeddingError",
]

# Configure logging
logger = logging.getLogger(__name__)


def _default_voyage_model() -> str:
    from config.settings import EmbeddingSettings

    return EmbeddingSettings().voyage_model


def _default_batch_size() -> int:
    from config.settings import EmbeddingSettings

    return EmbeddingSettings().batch_size


def _default_collection() -> str:
    from config.settings import VectorSettings

    return VectorSettings().collection_name


@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations.

    Unset fields come from settings, the same ones retrieval reads, so stored
    vectors and query vectors cannot silently use different models.
    """

    model_name: str = field(default_factory=_default_voyage_model)
    collection_name: str = field(default_factory=_default_collection)
    db_url: Optional[str] = None
    use_jsonb: bool = True
    batch_size: int = field(default_factory=_default_batch_size)

    def __post_init__(self):
        """Resolve the database URL from settings when not passed explicitly."""
        if not self.db_url:
            from config.settings import DatabaseSettings

            self.db_url = DatabaseSettings().url


class VectorStoreManager:
    """
    Manages vector store operations including embedding and document storage.

    This class provides a high-level interface for working with vector stores,
    handling connection management, error handling, and batch processing.
    """

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store manager.

        Args:
            config: Configuration object containing vector store settings.
        """
        self.config = config
        self._embeddings: Optional[Embeddings] = None
        self._vector_store: Optional[PGVector] = None

    @property
    def embeddings(self) -> Embeddings:
        """Lazy-load embeddings client."""
        if self._embeddings is None:
            from config.settings import EmbeddingSettings

            # Settings read .env without exporting it, so pass the key along
            # rather than rely on VOYAGE_API_KEY being in os.environ.
            api_key = EmbeddingSettings().voyage_api_key
            key_kwargs = {"voyage_api_key": api_key} if api_key else {}
            try:
                self._embeddings = VoyageAIEmbeddings(model=self.config.model_name, **key_kwargs)
                logger.info(f"Initialized embeddings with model: {self.config.model_name}")
            except Exception as e:
                raise EmbeddingError(f"Failed to initialize embeddings: {e}") from e

        return self._embeddings

    @property
    def vector_store(self) -> PGVector:
        """Lazy-load vector store connection."""
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()

        return self._vector_store

    def _create_vector_store(self) -> PGVector:
        """
        Create and configure the vector store connection.

        Returns:
            Configured PGVector instance.

        Raises:
            DatabaseConnectionError: If connection to database fails.
        """
        try:
            # PGVector creates its tables and this collection when built, on an
            # engine of its own. Vector writes and deletes go through the app's
            # session, so that engine is only needed here: NullPool closes its
            # connection afterwards instead of keeping one open on Neon. The
            # pgvector extension is the migrations' job (alembic / init_db).
            vector_store = PGVector(
                connection_string=self.config.db_url,
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name,
                use_jsonb=self.config.use_jsonb,
                create_extension=False,
                engine_args={"poolclass": NullPool},
            )

            logger.info(f"Connected to vector store collection: {self.config.collection_name}")
            return vector_store

        except pg_errors.InvalidCatalogName as e:
            error_msg = (
                f"Database '{self.config.collection_name}' not found. Check connection string."
            )
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to connect to database: {e}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e

    def compute_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """Compute vectors in input order without writing any database rows."""
        if self.config.batch_size < 1:
            raise EmbeddingError("Embedding batch size must be positive")

        vectors: List[List[float]] = []
        for start in range(0, len(documents), self.config.batch_size):
            batch = documents[start : start + self.config.batch_size]
            batch_number = start // self.config.batch_size + 1
            try:
                batch_vectors = self.embeddings.embed_documents([doc.page_content for doc in batch])
            except Exception as exc:
                raise EmbeddingError(f"Embedding batch {batch_number} failed: {exc}") from exc
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"Embedding batch {batch_number} returned {len(batch_vectors)} vectors "
                    f"for {len(batch)} chunks"
                )
            vectors.extend(batch_vectors)
        return vectors

    def validate_database(self, session: Session) -> None:
        """Require one database target for document and vector persistence."""
        db_url = self.config.db_url
        if not db_url or session.get_bind().engine.url != make_url(db_url):
            raise VectorStoreError("Document and vector writes must use the same database URL")

    def stage_vectors(
        self,
        session: Session,
        documents: List[Document],
        vectors: List[List[float]],
        collection_name: Optional[str] = None,
    ) -> None:
        """Flush vector rows using the caller's transaction.

        Args:
            collection_name: The collection to write into: the document's own
                collection. Defaults to this manager's configured collection.
        """
        self.validate_database(session)
        ids = {doc.id for doc in documents}
        if None in ids or "" in ids or len(ids) != len(documents):
            raise VectorStoreError("Each vector must have a unique chunk ID")
        if not documents or len(documents) != len(vectors):
            raise VectorStoreError("A vector is required for each chunk")

        # The legacy store initializes its schema/collection independently. Its
        # models can then use the application's session for all document data.
        store = self.vector_store
        name = collection_name or self.config.collection_name
        collection = store.CollectionStore.get_by_name(session, name)
        if collection is None:
            raise VectorStoreError(f"Vector collection {name!r} not found")
        session.add_all(
            store.EmbeddingStore(
                document=doc.page_content,
                embedding=vector,
                cmetadata=doc.metadata,
                custom_id=doc.id,
                collection_id=collection.uuid,
            )
            for doc, vector in zip(documents, vectors, strict=True)
        )
        session.flush()

    def delete_document_vectors(
        self, session: Session, document_id: int, chunk_ids: List[str]
    ) -> int:
        """Delete a document's vectors using the caller's transaction.

        Matches the document's current chunk IDs and also any vector tagged
        with the document's id. The second catches vectors that earlier
        versions of the pipeline committed without their chunk rows, which a
        delete by chunk ID alone never finds. Not limited to one collection:
        every vector of the document is the document's.

        Returns:
            How many vector rows were deleted
        """
        self.validate_database(session)
        embedding = self.vector_store.EmbeddingStore
        match = embedding.cmetadata["document_id"].astext == str(document_id)
        if chunk_ids:
            match = or_(match, embedding.custom_id.in_(chunk_ids))
        return session.query(embedding).filter(match).delete(synchronize_session=False)

    def collection_of_document(self, session: Session, document_id: int) -> Optional[str]:
        """The collection a document's existing vectors are in, if they agree on one.

        Used for documents ingested before the collection was recorded on the
        document. Returns None when the document has no vectors, or when they
        are split across collections (logged; nothing to inherit then).
        """
        self.validate_database(session)
        store = self.vector_store
        embedding, collection = store.EmbeddingStore, store.CollectionStore
        names = [
            name
            for (name,) in session.query(collection.name)
            .join(embedding, embedding.collection_id == collection.uuid)
            .filter(embedding.cmetadata["document_id"].astext == str(document_id))
            .distinct()
        ]
        if len(names) > 1:
            logger.warning(
                "Document %s has vectors in several collections %s; none inherited",
                document_id,
                sorted(names),
            )
            return None
        return names[0] if names else None

    def embed_documents(self, documents: List[Document]) -> List[str]:
        """Compute and write vectors for callers without relational chunk persistence."""
        if not documents:
            return []
        vectors = self.compute_embeddings(documents)
        ids = [doc.id or str(uuid4()) for doc in documents]
        try:
            return self.vector_store.add_embeddings(
                texts=[doc.page_content for doc in documents],
                embeddings=vectors,
                metadatas=[doc.metadata for doc in documents],
                ids=ids,
            )
        except Exception as exc:
            raise EmbeddingError(f"Failed to store embeddings: {exc}") from exc

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary containing collection metadata.
        """
        return {
            "collection_name": self.config.collection_name,
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "use_jsonb": self.config.use_jsonb,
        }
