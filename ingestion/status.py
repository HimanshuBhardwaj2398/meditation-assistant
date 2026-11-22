"""Mirror pipeline stage transitions into ``documents.status``.

The document row is the progress record: the Streamlit queue reads it today,
and API polling will read it later. There is no separate "persisting" status;
persistence reports under EMBEDDING until its own transaction marks the row
COMPLETED.
"""

import logging
from typing import Dict, Optional

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
    """Writes one document's stage transitions to ``documents.status``."""

    def __init__(self, document_id: int) -> None:
        self.document_id = document_id

    def record(self, stage_name: str, stage_status: StageStatus) -> None:
        """Mirror one stage transition. A failed write is logged, not raised."""
        status = self._map(stage_name, stage_status)
        if status is None:
            return
        try:
            self._write(status, f"Stage '{stage_name}' {stage_status.value}")
            logger.info(
                "Document %s status -> %s (%s: %s)",
                self.document_id,
                status.value,
                stage_name,
                stage_status.value,
            )
        except Exception as e:
            logger.warning(
                "Failed to persist stage status for document %s (%s=%s): %s",
                self.document_id,
                stage_name,
                stage_status.value,
                e,
            )

    def fail(self, details: str) -> None:
        """Mark the document FAILED with the run's errors. Raises if the write fails."""
        self._write(DocumentStatus.FAILED, details)

    @staticmethod
    def _map(stage_name: str, stage_status: StageStatus) -> Optional[DocumentStatus]:
        if stage_status == StageStatus.RUNNING:
            return RUNNING_STATUS.get(stage_name)
        if stage_status == StageStatus.COMPLETED:
            return COMPLETED_STATUS.get(stage_name)
        if stage_status == StageStatus.FAILED:
            return DocumentStatus.FAILED
        return None

    def _write(self, status: DocumentStatus, details: str) -> None:
        # Imported at call time so tests can swap the default database.
        from db.crud import DocumentCRUD
        from db.database import session_scope

        with session_scope() as session:
            DocumentCRUD(session).update_status(
                document_id=self.document_id,
                status=status,
                status_details=details,
            )
