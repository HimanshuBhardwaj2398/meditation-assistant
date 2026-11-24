"""Response models: the API's contract, which is also what the TypeScript client
will be generated from at step 2.

Routes convert ORM rows into these models inside the request's session, so
nothing lazy-loads after the session has closed.
"""

from pydantic import BaseModel

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
