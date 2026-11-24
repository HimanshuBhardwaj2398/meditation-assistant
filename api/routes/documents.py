"""The document collection, its items, and each item's chunks.

Routes are plain ``def``, not ``async def``: SQLAlchemy here is synchronous, and
FastAPI runs ``def`` routes in a thread pool so a slow query doesn't block the
event loop.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from api.deps import SessionDep
from api.schemas import ChunkList, ChunkOut, DocumentPage, DocumentSummary
from db.crud import ChunkCRUD, DocumentCRUD

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.get("", response_model=DocumentPage)
def list_documents(
    session: SessionDep,
    nikaya: Annotated[str | None, Query(description="Collection code: dn, mn, sn, an, kn")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPage:
    rows, total = DocumentCRUD(session).list_documents(nikaya=nikaya, limit=limit, offset=offset)
    items = [DocumentSummary.from_row(row) for row in rows]
    return DocumentPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentSummary)
def get_document(document_id: int, session: SessionDep) -> DocumentSummary:
    doc = DocumentCRUD(session).get_document_by_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DocumentSummary.from_row(doc)


@router.get("/{document_id}/chunks", response_model=ChunkList)
def list_chunks(document_id: int, session: SessionDep) -> ChunkList:
    # Check the parent first: "no such document" (404) and "a document with no
    # chunks yet" (200, empty list) are different answers.
    if DocumentCRUD(session).get_document_by_id(document_id) is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    chunks = ChunkCRUD(session).get_chunks_by_document(document_id)
    return ChunkList(document_id=document_id, items=[ChunkOut.from_row(c) for c in chunks])
