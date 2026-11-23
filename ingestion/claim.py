"""
Claim a document for one pipeline run and load what the run can reuse.

Before any stage runs, the run takes the document's processing claim (a lease
token in ``documents``), so no other run works on it at the same time. In the
same transaction it decides where the run starts: a resume reuses stored
markdown, ``from_chunking`` reuses the markdown, ``from_embedding`` reuses the
stored chunks and their UUIDs, and ``full`` starts from the source. The result
is a ``PreparedRun`` the orchestrator turns into a ``PipelineContext``.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy.orm import Session

from core.exceptions import DocumentBusyError, DocumentNotFoundError, PipelineError
from core.interfaces import StageName, StageStatus
from db.crud import ChunkCRUD, DocumentCRUD
from db.database import Database
from db.schema import Document, DocumentStatus
from ingestion.admission import VECTOR_COLLECTION_KEY
from ingestion.embed import VectorStoreManager
from ingestion.stages import source_provenance
from ingestion.status import RUNNING_STATUS

logger = logging.getLogger(__name__)


class ReprocessMode(str, Enum):
    """Supported reprocessing entry points for existing documents."""

    FULL = "full"
    FROM_CHUNKING = "from_chunking"
    FROM_EMBEDDING = "from_embedding"

    @classmethod
    def parse(cls, value: str) -> "ReprocessMode":
        """The mode a CLI flag or UI choice names.

        Raises:
            PipelineError: For a value that names no mode.
        """
        try:
            return cls(value)
        except ValueError as e:
            allowed = ", ".join(mode.value for mode in cls)
            raise PipelineError(
                f"Invalid reprocess mode {value!r}. Allowed values: {allowed}"
            ) from e


@dataclass
class PreparedRun:
    """Everything the pipeline needs to start on one claimed document."""

    document_id: int
    source: Optional[str]
    title: Optional[str]
    token: str
    # The document was COMPLETED when the run started: a reprocess replaces it.
    replacing: bool = False
    parsed_content: Optional[str] = None
    chunks: List[LangchainDocument] = field(default_factory=list)
    stage_results: Dict[str, StageStatus] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    # The document's own vector collection; publishing writes there.
    vector_collection: Optional[str] = None
    # The status the claim set as the run started.
    start_status: Optional[DocumentStatus] = None


def claim_document(
    database: Database,
    document_id: int,
    *,
    title: Optional[str],
    mode: Optional[ReprocessMode],
    clear_markdown: bool,
    lease_seconds: int,
    configured_collection: str,
    vector_store_manager: VectorStoreManager,
) -> Optional[PreparedRun]:
    """Take the document's processing claim and load what the run reuses.

    Args:
        database: Where the document lives
        document_id: The document to claim
        title: Title given by the caller, if any
        mode: Reprocess entry point, or None to process or resume
        clear_markdown: Clear the stored markdown (``full`` reprocess only)
        lease_seconds: How old another run's claim must be before it lapses
        configured_collection: The collection the orchestrator is set up for
        vector_store_manager: Used to find the collection an older document's
            vectors are in

    Returns None, without claiming, when the document is COMPLETED and no
    reprocess was asked for.

    Raises:
        DocumentNotFoundError: No such document.
        DocumentBusyError: Another run holds a live claim on it.
    """
    token = uuid.uuid4().hex
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=lease_seconds)

    with database.session_scope() as session:
        crud = DocumentCRUD(session)
        doc = crud.get_document_by_id(document_id)
        if not doc:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        was_completed = doc.status == DocumentStatus.COMPLETED
        if was_completed and mode is None:
            return None

        run = PreparedRun(
            document_id=doc.id,
            source=doc.file_path,
            title=title,
            token=token,
            replacing=was_completed,
        )
        _seed(run, doc, ChunkCRUD(session), mode)

        # The claim also sets the status of the first stage that will run,
        # saving that stage a status write of its own.
        first_stage = next(name for name in RUNNING_STATUS if name not in run.stage_results)
        run.start_status = RUNNING_STATUS[first_stage]
        if not crud.claim_for_processing(
            document_id,
            token,
            stale_before,
            status=run.start_status,
            status_details=f"Stage '{first_stage}' running",
        ):
            raise DocumentBusyError(
                f"Document {document_id} is being processed by another run; "
                f"try again once it finishes (a dead run's claim lapses after "
                f"{lease_seconds}s)"
            )

        run.vector_collection = _resolve_collection(
            session, doc, configured_collection, vector_store_manager
        )
        if clear_markdown:
            doc.markdown = None

    logger.info("Document %s claimed for %s", document_id, mode.value if mode else "processing")
    return run


def _resolve_collection(
    session: Session,
    doc: Document,
    configured_collection: str,
    vector_store_manager: VectorStoreManager,
) -> str:
    """The document's own vector collection, recorded on it if it lacked one.

    Retries and reprocesses publish into the document's collection,
    whatever collection this orchestrator is configured for. A document
    ingested before collections were recorded inherits the collection its
    vectors are already in, or else gets the configured one.
    """
    metadata = dict(doc.doc_metadata or {})
    collection = metadata.get(VECTOR_COLLECTION_KEY)
    if collection:
        if collection != configured_collection:
            logger.info(
                "Document %s stays in its own collection %r (not %r)",
                doc.id,
                collection,
                configured_collection,
            )
        return collection

    collection = (
        vector_store_manager.collection_of_document(session, doc.id) or configured_collection
    )
    metadata[VECTOR_COLLECTION_KEY] = collection
    doc.doc_metadata = metadata
    return collection


def _seed(
    run: PreparedRun,
    doc: Document,
    chunk_crud: ChunkCRUD,
    mode: Optional[ReprocessMode],
) -> None:
    """Mark the stages a run can reuse as done, from what the document stores."""
    reuses_stored = mode in (ReprocessMode.FROM_CHUNKING, ReprocessMode.FROM_EMBEDDING)
    resumes_from_markdown = mode is None and bool(doc.markdown)

    if reuses_stored or resumes_from_markdown:
        run.title = run.title or doc.title
        run.source_metadata = source_provenance(doc.doc_metadata or {})
    elif doc.title != "Untitled":
        run.title = run.title or doc.title

    if mode == ReprocessMode.FROM_CHUNKING or resumes_from_markdown:
        if not doc.markdown:
            raise PipelineError(
                f"Document {doc.id} has no stored markdown; cannot start from chunking"
            )
        run.parsed_content = doc.markdown
        run.stage_results = {StageName.PARSING: StageStatus.COMPLETED}
    elif mode == ReprocessMode.FROM_EMBEDDING:
        existing_chunks = chunk_crud.get_chunks_by_document(doc.id)
        if not existing_chunks:
            raise PipelineError(
                f"Document {doc.id} has no stored chunks; cannot start from embedding"
            )
        run.stage_results = {
            StageName.PARSING: StageStatus.COMPLETED,
            StageName.CHUNKING: StageStatus.COMPLETED,
        }
        for chunk_row in existing_chunks:
            metadata = dict(chunk_row.chunk_metadata or {})
            metadata["uuid"] = chunk_row.uuid
            run.chunks.append(
                LangchainDocument(page_content=chunk_row.chunk_text, metadata=metadata)
            )

    if mode == ReprocessMode.FULL and not run.source:
        raise PipelineError(
            f"Document {doc.id} is missing file_path; cannot reprocess from parsing"
        )
