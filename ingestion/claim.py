"""
Claim a document for one pipeline run and load what the run can reuse.

Before any stage runs, the run takes the document's processing claim (a lease
token in ``documents``), so no other run works on it at the same time. In the
same transaction it decides where the run starts: a resume reuses stored
markdown, ``from_chunking`` reuses the markdown, ``from_embedding`` reuses the
stored chunks and their UUIDs, and ``full`` starts from the source. The result
is a ``PreparedRun``: the ``PipelineContext`` the stages start from, plus what
the orchestrator needs to record the run's progress.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy.orm import Session

from core.exceptions import DocumentBusyError, DocumentNotFoundError, PipelineError
from core.interfaces import PipelineContext, StageName, StageStatus
from db.crud import ChunkCRUD, DocumentCRUD
from db.database import Database
from db.schema import Document, DocumentStatus
from ingestion.admission import VECTOR_COLLECTION_KEY
from ingestion.embed import VectorStoreManager
from ingestion.markdown_utils import UNTITLED
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


@dataclass(frozen=True)
class PreparedRun:
    """One claimed document, ready for the pipeline."""

    # What the stages start from: the document, its claim token, its vector
    # collection, and whatever stored work the run reuses.
    context: PipelineContext
    # The document was COMPLETED when the run started: a reprocess replaces it.
    replacing: bool
    # The status the claim set as the run started.
    start_status: DocumentStatus

    @property
    def document_id(self) -> int:
        assert self.context.document_id is not None
        return self.context.document_id

    @property
    def token(self) -> str:
        assert self.context.processing_token is not None
        return self.context.processing_token


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

        context = _seed(doc, ChunkCRUD(session), mode, title)

        # The claim also sets the status of the first stage that will run,
        # saving that stage a status write of its own.
        first_stage = next(name for name in RUNNING_STATUS if name not in context.stage_results)
        start_status = RUNNING_STATUS[first_stage]
        if not crud.claim_for_processing(
            document_id,
            token,
            stale_before,
            status=start_status,
            status_details=f"Stage '{first_stage}' running",
        ):
            raise DocumentBusyError(
                f"Document {document_id} is being processed by another run; "
                f"try again once it finishes (a dead run's claim lapses after "
                f"{lease_seconds}s)"
            )

        context = context.with_update(
            processing_token=token,
            vector_collection=_resolve_collection(
                session, doc, configured_collection, vector_store_manager
            ),
        )
        if clear_markdown:
            doc.markdown = None

    logger.info("Document %s claimed for %s", document_id, mode.value if mode else "processing")
    return PreparedRun(context=context, replacing=was_completed, start_status=start_status)


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
    doc: Document,
    chunk_crud: ChunkCRUD,
    mode: Optional[ReprocessMode],
    title: Optional[str],
) -> PipelineContext:
    """The context a run starts from: the stages it can reuse are marked done,
    with the stored markdown or chunks they would have produced."""
    reuses_stored = mode in (ReprocessMode.FROM_CHUNKING, ReprocessMode.FROM_EMBEDDING)
    resumes_from_markdown = mode is None and bool(doc.markdown)

    if mode == ReprocessMode.FULL and not doc.file_path:
        raise PipelineError(
            f"Document {doc.id} is missing file_path; cannot reprocess from parsing"
        )

    context = PipelineContext(document_id=doc.id, source=doc.file_path, title=title)
    if reuses_stored or resumes_from_markdown:
        context = context.with_update(
            title=title or doc.title,
            source_metadata=source_provenance(doc.doc_metadata or {}),
        )
    elif doc.title != UNTITLED:
        context = context.with_update(title=title or doc.title)

    if mode == ReprocessMode.FROM_CHUNKING or resumes_from_markdown:
        if not doc.markdown:
            raise PipelineError(
                f"Document {doc.id} has no stored markdown; cannot start from chunking"
            )
        return context.with_update(
            parsed_content=doc.markdown,
            stage_results={StageName.PARSING: StageStatus.COMPLETED},
        )

    if mode == ReprocessMode.FROM_EMBEDDING:
        stored = chunk_crud.get_chunks_by_document(doc.id)
        if not stored:
            raise PipelineError(
                f"Document {doc.id} has no stored chunks; cannot start from embedding"
            )
        chunks: List[LangchainDocument] = [
            LangchainDocument(
                page_content=row.chunk_text,
                metadata={**(row.chunk_metadata or {}), "uuid": row.uuid},
            )
            for row in stored
        ]
        return context.with_update(
            chunks=chunks,
            stage_results={
                StageName.PARSING: StageStatus.COMPLETED,
                StageName.CHUNKING: StageStatus.COMPLETED,
            },
        )

    return context
