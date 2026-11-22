"""Mirror a run's stage transitions into ``documents.status``.

The document row is the progress record: the Streamlit queue reads it today,
and API polling will read it later. There is no separate "persisting" status;
persistence reports under EMBEDDING until its own transaction marks the row
COMPLETED.

Every write is fenced by the run's claim token: it lands only while this run
still holds the document. The same write refreshes ``updated_at``, which is
what keeps the claim alive.
"""

import logging
from typing import Dict, Optional

from core.exceptions import ProcessingLeaseLostError
from core.interfaces import StageStatus
from db.schema import DocumentStatus

logger = logging.getLogger(__name__)

RUNNING_STATUS: Dict[str, DocumentStatus] = {
    "parsing": DocumentStatus.PARSING,
    "chunking": DocumentStatus.CHUNKING,
    "embedding": DocumentStatus.EMBEDDING,
    # No dedicated persistence enum; keep this under the embedding phase.
    "database_persistence": DocumentStatus.EMBEDDING,
}

COMPLETED_STATUS: Dict[str, DocumentStatus] = {
    "parsing": DocumentStatus.PARSED,
    "chunking": DocumentStatus.CHUNKED,
    # Keep EMBEDDING until database persistence marks COMPLETED.
    "embedding": DocumentStatus.EMBEDDING,
}


class StatusRecorder:
    """Writes one run's progress on one document, while the run holds its claim."""

    def __init__(self, document_id: int, token: str, *, replacing: bool = False) -> None:
        """
        Args:
            document_id: The document this run is processing
            token: The run's processing claim
            replacing: The document was COMPLETED when the run started, so a
                failure leaves the previous version published
        """
        self.document_id = document_id
        self.token = token
        self.replacing = replacing

    def record(self, stage_name: str, stage_status: StageStatus) -> None:
        """Mirror one stage transition.

        A failed stage writes nothing here; ``finish_failed`` writes the run's
        final state once. A database error is logged and the run goes on.

        Raises:
            ProcessingLeaseLostError: Another run has taken the document over,
                so this one must stop.
        """
        status = self._map(stage_name, stage_status)
        if status is None:
            return
        try:
            owned = self._write(status, f"Stage '{stage_name}' {stage_status.value}")
        except Exception as e:
            logger.warning(
                "Failed to persist stage status for document %s (%s=%s): %s",
                self.document_id,
                stage_name,
                stage_status.value,
                e,
            )
            return
        if not owned:
            raise ProcessingLeaseLostError(
                f"Document {self.document_id}: another run took over its processing claim"
            )
        logger.info(
            "Document %s status -> %s (%s: %s)",
            self.document_id,
            status.value,
            stage_name,
            stage_status.value,
        )

    def finish_failed(self, details: str) -> None:
        """End a failed run and release its claim.

        A document that was never published ends FAILED. A reprocess of a
        COMPLETED document ends COMPLETED again: its previous chunks and
        vectors are untouched and still published, and the details say what
        the failed reprocess hit.
        """
        if self.replacing:
            status = DocumentStatus.COMPLETED
            details = f"Reprocess failed; the previous version is still published. {details}"
        else:
            status = DocumentStatus.FAILED
        if not self._write(status, details, release=True):
            logger.warning(
                "Document %s: another run took it over; leaving its status alone",
                self.document_id,
            )

    @staticmethod
    def _map(stage_name: str, stage_status: StageStatus) -> Optional[DocumentStatus]:
        if stage_status == StageStatus.RUNNING:
            return RUNNING_STATUS.get(stage_name)
        if stage_status == StageStatus.COMPLETED:
            return COMPLETED_STATUS.get(stage_name)
        return None

    def _write(self, status: DocumentStatus, details: str, release: bool = False) -> bool:
        # Imported at call time so tests can swap the default database.
        from db.crud import DocumentCRUD
        from db.database import session_scope

        with session_scope() as session:
            return DocumentCRUD(session).set_status_if_owner(
                self.document_id,
                self.token,
                status,
                status_details=details,
                release=release,
            )
