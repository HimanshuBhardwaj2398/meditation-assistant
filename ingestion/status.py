"""Mirror a run's stage transitions into ``documents.status``.

The document row is the progress record: the Streamlit queue reads it today,
and API polling will read it later. There is no separate "persisting" status;
persistence reports under EMBEDDING until its own transaction marks the row
COMPLETED.

Every write is fenced by the run's claim token: it lands only while this run
still holds the document. The same write refreshes ``updated_at``, which is
what keeps the claim alive.

Each write is a Neon round trip or three, so only real changes are written:
the status a stage starts under, and only when it differs from the current
one. A stage finishing writes nothing (the next stage starting overwrites it a
moment later), and the claim already set the run's starting status. A clean
run writes CHUNKING and EMBEDDING here; the claim wrote PARSING before it, and
publishing writes COMPLETED.
"""

import logging
from typing import Dict, Optional

from core.exceptions import ProcessingLeaseLostError
from core.interfaces import StageName, StageStatus
from db.schema import DocumentStatus

logger = logging.getLogger(__name__)

RUNNING_STATUS: Dict[str, DocumentStatus] = {
    StageName.PARSING: DocumentStatus.PARSING,
    StageName.CHUNKING: DocumentStatus.CHUNKING,
    StageName.EMBEDDING: DocumentStatus.EMBEDDING,
    # No dedicated persistence enum; keep this under the embedding phase.
    StageName.DATABASE_PERSISTENCE: DocumentStatus.EMBEDDING,
}


class StatusRecorder:
    """Writes one run's progress on one document, while the run holds its claim."""

    def __init__(
        self,
        document_id: int,
        token: str,
        *,
        replacing: bool = False,
        current: Optional[DocumentStatus] = None,
    ) -> None:
        """
        Args:
            document_id: The document this run is processing
            token: The run's processing claim
            replacing: The document was COMPLETED when the run started, so a
                failure leaves the previous version published
            current: The status the document has now (the one the claim set)
        """
        self.document_id = document_id
        self.token = token
        self.replacing = replacing
        self.current = current

    def record(self, stage_name: str, stage_status: StageStatus) -> None:
        """Mirror one stage transition.

        Only a stage starting under a new status writes. A failed stage writes
        nothing here; ``finish_failed`` writes the run's final state once. A
        database error is logged and the run goes on.

        Raises:
            ProcessingLeaseLostError: Another run has taken the document over,
                so this one must stop.
        """
        status = self._map(stage_name, stage_status)
        if status is None or status == self.current:
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
        self.current = status
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
