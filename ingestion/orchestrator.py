"""
Document ingestion orchestration.

``IngestionOrchestrator`` prepares one document (creates it, or loads it to
resume or reprocess), runs the stage pipeline on it (``ingestion.pipeline``),
and records progress in ``documents.status``.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

# LangChain Document for type hints
from langchain_core.documents import Document as LangchainDocument

from core.exceptions import (
    DocumentBusyError,
    DocumentNotFoundError,
    PipelineError,
    ProcessingLeaseLostError,
)
from core.interfaces import PipelineContext, PipelineStage, StageName, StageStatus
from ingestion.admission import VECTOR_COLLECTION_KEY, Admission, AdmissionOutcome, admit
from ingestion.chunking import Config as ChunkingConfig
from ingestion.embed import VectorStoreConfig, VectorStoreManager
from ingestion.parsing import ParserFactory
from ingestion.pipeline import PipelineOrchestrator
from ingestion.stages import (
    ChunkingStage,
    DatabasePersistenceStage,
    EmbeddingStage,
    ParsingStage,
    source_provenance,
)
from ingestion.status import RUNNING_STATUS, StatusRecorder

logger = logging.getLogger(__name__)


# ============================================================================
# REPROCESS MODES
# ============================================================================


class ReprocessMode(str, Enum):
    """Supported reprocessing entry points for existing documents."""

    FULL = "full"
    FROM_CHUNKING = "from_chunking"
    FROM_EMBEDDING = "from_embedding"


# ============================================================================
# HIGH-LEVEL INGESTION ORCHESTRATOR
# ============================================================================


class IngestionResult(TypedDict):
    """What ``IngestionOrchestrator.process`` reports for one document."""

    document_id: int
    source: Optional[str]
    title: Optional[str]
    chunk_count: int
    reprocess_mode: Optional[str]
    stage_results: Dict[str, str]
    errors: Dict[str, str]
    success: bool
    # True when the document was already COMPLETED and nothing was run.
    already_completed: bool


@dataclass
class _PreparedRun:
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
    start_status: Optional[Any] = None


class IngestionOrchestrator:
    """
    High-level orchestrator for document ingestion.

    For each document it decides what to do (admit a new source, resume an
    unfinished one, reprocess, or report one that is already complete), takes
    the document's processing claim so no other run works on it at the same
    time, runs the stage pipeline, and records progress in
    ``documents.status``.
    """

    def __init__(
        self,
        vector_store_config: VectorStoreConfig,
        chunking_config: Optional[ChunkingConfig] = None,
        enable_database_persistence: bool = True,
        *,
        parser_factory: Optional[ParserFactory] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
    ):
        """
        Initialize ingestion orchestrator.

        Args:
            vector_store_config: Configuration for vector store
            chunking_config: Configuration for chunking (optional)
            enable_database_persistence: Enable database stage (default True)
            parser_factory: Parser selection to use instead of the default
                ParserFactory (tests inject a fake)
            vector_store_manager: Vector adapter to use instead of one built
                from vector_store_config (tests inject a fake)
        """
        from config.settings import IngestionSettings

        self.vector_store_config = vector_store_config
        self.chunking_config = chunking_config or ChunkingConfig.from_settings()
        self.enable_database_persistence = enable_database_persistence
        self.lease_seconds = IngestionSettings().lease_seconds

        # Initialize components
        self.parser_factory = parser_factory or ParserFactory()
        self.vector_store_manager = vector_store_manager or VectorStoreManager(
            config=vector_store_config
        )

        logger.info("IngestionOrchestrator initialized with DAG pipeline")

    @staticmethod
    def _parse_reprocess_mode(
        reprocess_mode: Optional[Union[ReprocessMode, str]],
    ) -> Optional[ReprocessMode]:
        """Normalize and validate optional reprocess mode."""
        if reprocess_mode is None:
            return None
        if isinstance(reprocess_mode, ReprocessMode):
            return reprocess_mode
        try:
            return ReprocessMode(reprocess_mode)
        except ValueError as e:
            allowed = ", ".join(mode.value for mode in ReprocessMode)
            raise PipelineError(
                f"Invalid reprocess_mode '{reprocess_mode}'. Allowed values: {allowed}"
            ) from e

    async def process(
        self,
        source: Union[str, int],
        title: Optional[str] = None,
        on_stage_update: Optional[Callable[[str, StageStatus], None]] = None,
        reprocess_mode: Optional[Union[ReprocessMode, str]] = None,
        clear_markdown: bool = False,
    ) -> IngestionResult:
        """
        Process a document through the ingestion pipeline.

        A source string goes through admission first: a new source gets a
        document, and a source already in the corpus maps to its existing
        document. Then, for the document:

        - COMPLETED, no reprocess_mode: nothing runs; the result reports the
          published version with ``already_completed=True``.
        - Not completed: the run resumes, reusing stored markdown when the
          earlier attempt got that far.
        - With reprocess_mode: the run replaces the published version. If it
          fails, the previous chunks and vectors stay published.

        Args:
            source: File path, URL, or document ID
            title: Optional title (will be extracted if not provided)
            on_stage_update: Optional callback(stage_name, status) for UI updates
            reprocess_mode: Optional mode for existing documents:
                - "full": re-parse + chunk + embed + persist
                - "from_chunking": reuse stored markdown, rerun chunking+
                  embedding+persistence
                - "from_embedding": reuse stored chunks (and their UUIDs),
                  rerun embedding+persistence
            clear_markdown: With reprocess_mode="full", clear
                documents.markdown before the run.

        Returns:
            The run's IngestionResult.

        Raises:
            DocumentBusyError: Another run is processing the document.
            ProcessingLeaseLostError: Another run took the document over during
                this run, which stopped without writing anything.
        """
        mode = self._parse_reprocess_mode(reprocess_mode)
        if clear_markdown and mode != ReprocessMode.FULL:
            raise PipelineError("clear_markdown is only supported with reprocess_mode='full'")

        self._check_same_database()

        if isinstance(source, int):
            document_id = source
        else:
            if mode is not None:
                raise PipelineError(
                    "reprocess_mode can only be used when source is an existing document ID"
                )
            admission = self.admit(source, title=title)
            if admission.outcome == AdmissionOutcome.COMPLETED:
                return self._existing_result(admission.document_id)
            document_id = admission.document_id

        run = self._claim(document_id, title, mode, clear_markdown)
        if run is None:
            return self._existing_result(document_id)
        return await self._run_pipeline(run, mode, on_stage_update)

    def admit(
        self,
        source: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> Admission:
        """Find or create the document for a source (see ``ingestion.admission``).

        A new document records this orchestrator's vector collection as its
        own, so later retries and reprocesses publish there too.
        """
        metadata = {
            **(doc_metadata or {}),
            VECTOR_COLLECTION_KEY: self.vector_store_config.collection_name,
        }
        return admit(source, title=title, description=description, tags=tags, doc_metadata=metadata)

    # ------------------------------------------------------------------
    # Preparing a document
    # ------------------------------------------------------------------

    def _check_same_database(self) -> None:
        """Vectors and documents must be written to the same database."""
        if not self.enable_database_persistence:
            return
        from db.database import session_scope

        # Compares URLs only; no connection is opened.
        with session_scope() as session:
            self.vector_store_manager.validate_database(session)

    def _claim(
        self,
        document_id: int,
        title: Optional[str],
        mode: Optional[ReprocessMode],
        clear_markdown: bool,
    ) -> Optional[_PreparedRun]:
        """Take the document's processing claim and load what the run reuses.

        Returns None, without claiming, when the document is COMPLETED and no
        reprocess was asked for.

        Raises:
            DocumentNotFoundError: No such document.
            DocumentBusyError: Another run holds a live claim on it.
        """
        from db.crud import ChunkCRUD, DocumentCRUD
        from db.database import session_scope
        from db.schema import DocumentStatus

        token = uuid.uuid4().hex
        stale_before = datetime.now(timezone.utc) - timedelta(seconds=self.lease_seconds)

        with session_scope() as session:
            crud = DocumentCRUD(session)
            doc = crud.get_document_by_id(document_id)
            if not doc:
                raise DocumentNotFoundError(f"Document {document_id} not found")

            was_completed = doc.status == DocumentStatus.COMPLETED
            if was_completed and mode is None:
                return None

            run = _PreparedRun(
                document_id=doc.id,
                source=doc.file_path,
                title=title,
                token=token,
                replacing=was_completed,
            )
            self._seed(run, doc, ChunkCRUD(session), mode)

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
                    f"{self.lease_seconds}s)"
                )

            if self.enable_database_persistence:
                run.vector_collection = self._resolve_collection(session, doc)
            if clear_markdown:
                doc.markdown = None

        logger.info("Document %s claimed for %s", document_id, mode.value if mode else "processing")
        return run

    def _resolve_collection(self, session: Any, doc: Any) -> str:
        """The document's own vector collection, recorded on it if it lacked one.

        Retries and reprocesses publish into the document's collection,
        whatever collection this orchestrator is configured for. A document
        ingested before collections were recorded inherits the collection its
        vectors are already in, or else gets the configured one.
        """
        metadata = dict(doc.doc_metadata or {})
        collection = metadata.get(VECTOR_COLLECTION_KEY)
        if collection:
            if collection != self.vector_store_config.collection_name:
                logger.info(
                    "Document %s stays in its own collection %r (not %r)",
                    doc.id,
                    collection,
                    self.vector_store_config.collection_name,
                )
            return collection

        collection = (
            self.vector_store_manager.collection_of_document(session, doc.id)
            or self.vector_store_config.collection_name
        )
        metadata[VECTOR_COLLECTION_KEY] = collection
        doc.doc_metadata = metadata
        return collection

    @staticmethod
    def _seed(
        run: _PreparedRun,
        doc: Any,
        chunk_crud: Any,
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

    def _existing_result(self, document_id: int) -> IngestionResult:
        """Report a COMPLETED document without running anything."""
        from db.crud import ChunkCRUD, DocumentCRUD
        from db.database import session_scope

        with session_scope() as session:
            doc = DocumentCRUD(session).get_document_by_id(document_id)
            if not doc:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            result = IngestionResult(
                document_id=doc.id,
                source=doc.file_path,
                title=doc.title,
                chunk_count=ChunkCRUD(session).count_chunks_by_document(doc.id),
                reprocess_mode=None,
                stage_results={},
                errors={},
                success=True,
                already_completed=True,
            )
        logger.info("Document %s is already completed; nothing to run", document_id)
        return result

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def _build_stages(self) -> List[PipelineStage]:
        stages: List[PipelineStage] = [
            ParsingStage(self.parser_factory),
            ChunkingStage(self.chunking_config),
            EmbeddingStage(
                self.vector_store_manager, vector_only=not self.enable_database_persistence
            ),
        ]
        if self.enable_database_persistence:
            stages.append(DatabasePersistenceStage(self.vector_store_manager))
        return stages

    @property
    def _final_stage(self) -> StageName:
        """The stage whose completion means the document is done."""
        if self.enable_database_persistence:
            return StageName.DATABASE_PERSISTENCE
        return StageName.EMBEDDING

    async def _run_pipeline(
        self,
        run: _PreparedRun,
        mode: Optional[ReprocessMode],
        on_stage_update: Optional[Callable[[str, StageStatus], None]],
    ) -> IngestionResult:
        """Run the stages on a claimed document and report the outcome."""
        recorder = StatusRecorder(
            run.document_id, run.token, replacing=run.replacing, current=run.start_status
        )

        def _on_stage_update(stage_name: str, stage_status: StageStatus) -> None:
            recorder.record(stage_name, stage_status)
            if on_stage_update is not None:
                on_stage_update(stage_name, stage_status)

        pipeline = PipelineOrchestrator(self._build_stages())
        context = PipelineContext(
            document_id=run.document_id,
            source=run.source,
            title=run.title,
            parsed_content=run.parsed_content,
            chunks=run.chunks,
            stage_results=run.stage_results,
            source_metadata=run.source_metadata,
            processing_token=run.token,
            vector_collection=run.vector_collection,
        )

        try:
            logger.info(f"Starting pipeline for document {run.document_id}")
            final_context = await pipeline.execute(context, on_stage_update=_on_stage_update)

            success = final_context.stage_results.get(self._final_stage) == StageStatus.COMPLETED
            result = IngestionResult(
                document_id=run.document_id,
                source=run.source,
                title=final_context.title,
                chunk_count=len(final_context.chunks) if final_context.chunks else 0,
                reprocess_mode=mode.value if mode else None,
                stage_results={
                    name: status.value for name, status in final_context.stage_results.items()
                },
                errors=final_context.error_messages,
                success=success,
                already_completed=False,
            )

            if success:
                logger.info(f"Pipeline completed successfully for document {run.document_id}")
            else:
                logger.warning(f"Pipeline partially completed for document {run.document_id}")
                recorder.finish_failed(str(final_context.error_messages))

            return result

        except ProcessingLeaseLostError:
            logger.warning(
                "Document %s was taken over by another run; this run stopped", run.document_id
            )
            raise
        except Exception as e:
            logger.error(f"Pipeline failed for document {run.document_id}: {e}", exc_info=True)
            recorder.finish_failed(str(e))
            raise PipelineError(f"Pipeline execution failed: {e}") from e
