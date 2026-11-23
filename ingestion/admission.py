"""Admission: what happens when a source is submitted for ingestion.

One rule, shared by the orchestrator, the services and the CLIs:

- a source not seen before gets a new PENDING document;
- a source whose document is COMPLETED is not processed again;
- a source whose document never completed is resumed.

Sources are compared in canonical form, so one SuttaCentral text submitted as
``sc:MN1/sujato`` and as its reading URL is one document, not two.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.exceptions import DatabaseError
from db.crud import DocumentCRUD
from db.database import Database, get_database
from db.schema import DocumentStatus
from ingestion.markdown_utils import UNTITLED
from ingestion.suttacentral import SuttaCentralParser, parse_sutta_ref

logger = logging.getLogger(__name__)

_SUTTACENTRAL = SuttaCentralParser()

# doc_metadata key naming the vector collection a document belongs to.
VECTOR_COLLECTION_KEY = "vector_collection"


def canonical_source(source: str) -> str:
    """The form a source is stored and compared in (``documents.file_path``).

    SuttaCentral references and reading URLs become ``sc:<uid>/<author>``, with
    ``/<lang>`` appended only when the language is not English; this matches
    how the existing corpus is stored. Other sources only lose surrounding
    whitespace.

    Raises:
        ParsingError: For a SuttaCentral reference too short to name a text, so
            nothing is created for it.
    """
    source = source.strip()
    if not _SUTTACENTRAL.can_parse(source):
        return source
    ref = parse_sutta_ref(source)
    lang = ref.lang.lower()
    suffix = "" if lang == "en" else f"/{lang}"
    return f"sc:{ref.uid.lower()}/{ref.author.lower()}{suffix}"


class AdmissionOutcome(str, Enum):
    """What admission found for a source."""

    CREATED = "created"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Admission:
    """The document a submitted source maps to, and what admission found."""

    document_id: int
    source: str
    outcome: AdmissionOutcome


def admit(
    source: str,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    doc_metadata: Optional[Dict[str, Any]] = None,
    database: Optional[Database] = None,
) -> Admission:
    """Find or create the document for a source, in one transaction.

    The optional fields only apply to a newly created document; an existing
    document keeps its own. ``database`` defaults to the default database.
    """
    source = canonical_source(source)
    with (database or get_database()).session_scope() as session:
        crud = DocumentCRUD(session)
        new_id = crud.insert_if_absent(
            file_path=source,
            title=title or UNTITLED,
            description=description,
            tags=tags,
            doc_metadata=doc_metadata,
        )
        if new_id is not None:
            admission = Admission(new_id, source, AdmissionOutcome.CREATED)
        else:
            doc = crud.get_document_by_file_path(source)
            if doc is None:
                raise DatabaseError(f"{source!r} conflicted on insert but no document has it")
            outcome = (
                AdmissionOutcome.COMPLETED
                if doc.status == DocumentStatus.COMPLETED
                else AdmissionOutcome.INCOMPLETE
            )
            admission = Admission(doc.id, source, outcome)

    logger.info(
        "Admitted %s as document %s (%s)",
        source,
        admission.document_id,
        admission.outcome.value,
    )
    return admission
