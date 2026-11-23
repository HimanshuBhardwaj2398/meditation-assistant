"""
Document ingestion orchestration.

``IngestionOrchestrator`` prepares one document (creates it, or loads it to
resume or reprocess), runs the stage pipeline on it (``ingestion.pipeline``),
and records progress in ``documents.status``.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TypedDict

from config.settings import IngestionSettings
from core.exceptions import DocumentNotFoundError, PipelineError, ProcessingLeaseLostError
from core.interfaces import PipelineContext, PipelineStage, StageName, StageStatus
from db.crud import ChunkCRUD, DocumentCRUD
from db.database import Database, get_database
from ingestion.admission import VECTOR_COLLECTION_KEY, Admission, AdmissionOutcome, admit
from ingestion.chunking import ChunkingConfig
from ingestion.claim import PreparedRun, ReprocessMode, claim_document
from ingestion.embed import VectorStoreConfig, VectorStoreManager
from ingestion.parsing import ParserFactory
from ingestion.pipeline import PipelineOrchestrator, StageCallback
from ingestion.stages import (
    ChunkingStage,
    DatabasePersistenceStage,
    EmbeddingStage,
    ParsingStage,
)
from ingestion.status import StatusRecorder

__all__ = ["IngestionOrchestrator", "IngestionResult", "ReprocessMode"]

logger = logging.getLogger(__name__)


class IngestionResult(TypedDict):
    """What ``ingest``, ``resume`` and ``reprocess`` report for one document."""

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
        *,
        parser_factory: Optional[ParserFactory] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        database: Optional[Database] = None,
    ):
        """
        Initialize ingestion orchestrator.

        Args:
            vector_store_config: Configuration for vector store
            chunking_config: Configuration for chunking (optional)
            parser_factory: Parser selection to use instead of the default
                ParserFactory (tests inject a fake)
            vector_store_manager: Vector adapter to use instead of one built
                from vector_store_config (tests inject a fake)
            database: The database documents and chunks are written to;
                defaults to the default database (``set_default_database``)
        """
        self.database = database or get_database()
        self.vector_store_config = vector_store_config
        self.chunking_config = chunking_config or ChunkingConfig.from_settings()
        self.lease_seconds = IngestionSettings().lease_seconds

        # Initialize components
        self.parser_factory = parser_factory or ParserFactory()
        self.vector_store_manager = vector_store_manager or VectorStoreManager(
            config=vector_store_config
        )

        logger.info("IngestionOrchestrator initialized with DAG pipeline")

    # ------------------------------------------------------------------
    # Entry points
    #
    # Admission, claiming and status writes are database round trips: they
    # run in worker threads, like the stages, to keep the event loop free.
    # ------------------------------------------------------------------

    async def ingest(
        self,
        source: str,
        *,
        title: Optional[str] = None,
        on_stage_update: Optional[StageCallback] = None,
    ) -> IngestionResult:
        """Ingest a source: a SuttaCentral reference, a URL or a PDF path.

        The source goes through admission (``ingestion.admission``). A new
        source gets a document and runs. A source whose document is
        COMPLETED runs nothing: the result reports the published version
        with ``already_completed=True``. A source that never completed
        resumes, as ``resume`` does.

        Args:
            source: The source to ingest
            title: Title for a new document (else the parsed one is used)
            on_stage_update: Optional callback(stage_name, status), sync or
                async, for progress updates

        Raises:
            DocumentBusyError: Another run is processing the document.
            ProcessingLeaseLostError: Another run took the document over during
                this run, which stopped without writing anything.
        """
        await asyncio.to_thread(self._check_same_database)
        admission = await asyncio.to_thread(self.admit, source, title=title)
        if admission.outcome == AdmissionOutcome.COMPLETED:
            return await asyncio.to_thread(self._existing_result, admission.document_id)
        return await self._claim_and_run(
            admission.document_id, title=title, on_stage_update=on_stage_update
        )

    async def resume(
        self,
        document_id: int,
        *,
        title: Optional[str] = None,
        on_stage_update: Optional[StageCallback] = None,
    ) -> IngestionResult:
        """Finish a document that never completed, reusing its stored markdown
        when an earlier attempt got that far. A COMPLETED document runs
        nothing and is reported with ``already_completed=True``.

        Raises:
            DocumentNotFoundError: No such document.
            DocumentBusyError: Another run is processing the document.
            ProcessingLeaseLostError: As for ``ingest``.
        """
        await asyncio.to_thread(self._check_same_database)
        return await self._claim_and_run(document_id, title=title, on_stage_update=on_stage_update)

    async def reprocess(
        self,
        document_id: int,
        mode: ReprocessMode,
        *,
        clear_markdown: bool = False,
        on_stage_update: Optional[StageCallback] = None,
    ) -> IngestionResult:
        """Rebuild a document, replacing its published version.

        If the run fails, the previous chunks and vectors stay published.

        Args:
            document_id: The document to rebuild
            mode: Where the rebuild starts:
                - FULL: from the source (parse, chunk, embed, publish)
                - FROM_CHUNKING: from the stored markdown
                - FROM_EMBEDDING: from the stored chunks, keeping their UUIDs
            clear_markdown: Clear the stored markdown first (FULL only)
            on_stage_update: As for ``ingest``

        Raises:
            PipelineError: clear_markdown with a mode other than FULL.
            DocumentNotFoundError, DocumentBusyError, ProcessingLeaseLostError:
                As for ``resume``.
        """
        if clear_markdown and mode != ReprocessMode.FULL:
            raise PipelineError("clear_markdown is only supported with ReprocessMode.FULL")
        await asyncio.to_thread(self._check_same_database)
        return await self._claim_and_run(
            document_id,
            mode=mode,
            clear_markdown=clear_markdown,
            on_stage_update=on_stage_update,
        )

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
        return admit(
            source,
            title=title,
            description=description,
            tags=tags,
            doc_metadata=metadata,
            database=self.database,
        )

    # ------------------------------------------------------------------
    # Before the pipeline
    # ------------------------------------------------------------------

    async def _claim_and_run(
        self,
        document_id: int,
        *,
        title: Optional[str] = None,
        mode: Optional[ReprocessMode] = None,
        clear_markdown: bool = False,
        on_stage_update: Optional[StageCallback] = None,
    ) -> IngestionResult:
        """Claim the document and run the pipeline on it, or report it as
        already completed when there is nothing to run."""
        run = await asyncio.to_thread(
            claim_document,
            self.database,
            document_id,
            title=title,
            mode=mode,
            clear_markdown=clear_markdown,
            lease_seconds=self.lease_seconds,
            configured_collection=self.vector_store_config.collection_name,
            vector_store_manager=self.vector_store_manager,
        )
        if run is None:
            return await asyncio.to_thread(self._existing_result, document_id)
        return await self._run_pipeline(run, mode, on_stage_update)

    def _check_same_database(self) -> None:
        """Vectors and documents must be written to the same database."""
        # Compares URLs only; no connection is opened.
        with self.database.session_scope() as session:
            self.vector_store_manager.validate_database(session)

    def _existing_result(self, document_id: int) -> IngestionResult:
        """Report a COMPLETED document without running anything."""
        with self.database.session_scope() as session:
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
        return [
            ParsingStage(self.parser_factory, self.database),
            ChunkingStage(self.chunking_config),
            EmbeddingStage(self.vector_store_manager),
            DatabasePersistenceStage(self.vector_store_manager, self.database),
        ]

    async def _run_pipeline(
        self,
        run: PreparedRun,
        mode: Optional[ReprocessMode],
        on_stage_update: Optional[StageCallback],
    ) -> IngestionResult:
        """Run the stages on a claimed document and report the outcome."""
        recorder = StatusRecorder(
            self.database,
            run.document_id,
            run.token,
            replacing=run.replacing,
            current=run.start_status,
        )

        async def _on_stage_update(stage_name: str, stage_status: StageStatus) -> None:
            await asyncio.to_thread(recorder.record, stage_name, stage_status)
            # The caller's callback stays on the event loop (Streamlit needs that).
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

            success = (
                final_context.stage_results.get(StageName.DATABASE_PERSISTENCE)
                == StageStatus.COMPLETED
            )
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
                await asyncio.to_thread(recorder.finish_failed, str(final_context.error_messages))

            return result

        except ProcessingLeaseLostError:
            logger.warning(
                "Document %s was taken over by another run; this run stopped", run.document_id
            )
            raise
        except Exception as e:
            logger.error(f"Pipeline failed for document {run.document_id}: {e}", exc_info=True)
            await asyncio.to_thread(recorder.finish_failed, str(e))
            raise PipelineError(f"Pipeline execution failed: {e}") from e
