"""Response models: the API's contract, which is also what the TypeScript client
will be generated from at step 2.

Routes convert ORM rows into these models inside the request's session, so
nothing lazy-loads after the session has closed.
"""

from datetime import datetime

from pydantic import AnyHttpUrl, BaseModel, ValidationError

from db.schema import Chunk, Document, DocumentStatus
from retrieval.utils import extract_header_paths


class DocumentSummary(BaseModel):
    """One document, as it appears in a list."""

    id: int
    title: str
    uid: str | None  # SuttaCentral uid, e.g. "mn10"; None for PDFs and web pages
    nikaya: str | None  # "dn", "mn", "sn", "an", "kn"
    status: DocumentStatus

    @classmethod
    def from_row(cls, doc: Document) -> "DocumentSummary":
        meta = doc.doc_metadata or {}
        return cls(
            id=doc.id,
            title=doc.title,
            uid=meta.get("uid"),
            nikaya=meta.get("nikaya"),
            status=doc.status,
        )


class DocumentPage(BaseModel):
    """One page of the document collection."""

    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DocumentDetail(DocumentSummary):
    """The document fields the reader can show above its chunks."""

    description: str | None
    tags: list[str]
    source: str | None
    reading_url: AnyHttpUrl | None
    author_uid: str | None
    lang: str | None
    doc_type: str | None
    category: str | None
    created_at: datetime

    @classmethod
    def from_row(cls, doc: Document) -> "DocumentDetail":
        meta = doc.doc_metadata or {}

        def metadata_text(key: str) -> str | None:
            value = meta.get(key)
            if isinstance(value, str):
                return value.strip() or None
            return None

        reading_url = None
        if url := metadata_text("reading_url"):
            try:
                reading_url = AnyHttpUrl(url)
            except ValidationError:
                pass  # A malformed legacy link should not prevent reading the document.

        return cls(
            **DocumentSummary.from_row(doc).model_dump(),
            description=doc.description,
            tags=doc.tags or [],
            source=metadata_text("source"),
            reading_url=reading_url,
            author_uid=metadata_text("author_uid"),
            lang=metadata_text("lang"),
            doc_type=metadata_text("type"),
            category=metadata_text("category"),
            created_at=doc.created_at,
        )


class ChunkOut(BaseModel):
    """One chunk, in reading order."""

    uuid: str
    chunk_index: int
    text: str
    headers: list[str]

    @classmethod
    def from_row(cls, chunk: Chunk) -> "ChunkOut":
        return cls(
            uuid=chunk.uuid,
            chunk_index=chunk.chunk_index,
            text=chunk.chunk_text,
            headers=extract_header_paths(chunk.chunk_metadata),
        )


class ChunkList(BaseModel):
    """All chunks of one document. Not paged: even the longest sutta has few."""

    document_id: int
    items: list[ChunkOut]
