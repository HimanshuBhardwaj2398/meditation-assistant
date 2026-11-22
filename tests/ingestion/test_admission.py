"""Admission: one canonical form per source, and one rule for what a submission does."""

import pytest

from core.exceptions import ParsingError
from db.schema import Document, DocumentStatus
from ingestion.admission import AdmissionOutcome, admit, canonical_source


@pytest.mark.parametrize(
    ("submitted", "canonical"),
    [
        ("sc:mn1/sujato", "sc:mn1/sujato"),
        ("  sc:MN1/Sujato  ", "sc:mn1/sujato"),
        ("sc:mn1/sujato/en", "sc:mn1/sujato"),
        ("sc:mn1/sujato/", "sc:mn1/sujato"),
        ("sc:mn1/sujato/de", "sc:mn1/sujato/de"),
        ("https://suttacentral.net/mn1/en/sujato", "sc:mn1/sujato"),
        ("https://www.suttacentral.net/MN1/en/sujato?lang=en", "sc:mn1/sujato"),
        ("https://suttacentral.net/sn1.1/en/sujato", "sc:sn1.1/sujato"),
        ("https://suttacentral.net/mn1/pli/ms", "sc:mn1/ms/pli"),
        (" https://example.com/page ", "https://example.com/page"),
        ("Books/sutra.pdf", "Books/sutra.pdf"),
    ],
)
def test_canonical_source(submitted, canonical):
    assert canonical_source(submitted) == canonical


def test_a_suttacentral_ref_that_names_no_text_is_rejected():
    with pytest.raises(ParsingError):
        canonical_source("sc:mn1")


def test_admission_finds_one_document_however_the_source_is_spelled(ingest_db):
    first = admit("sc:mn1/sujato", title="MN 1", tags=["mn"])
    again = admit("https://suttacentral.net/MN1/en/sujato", title="ignored")

    assert first.outcome == AdmissionOutcome.CREATED
    assert again.outcome == AdmissionOutcome.INCOMPLETE
    assert again.document_id == first.document_id
    assert again.source == first.source == "sc:mn1/sujato"
    with ingest_db.session_scope() as session:
        doc = session.get(Document, first.document_id)
        assert doc.status == DocumentStatus.PENDING
        assert doc.title == "MN 1"
        assert list(doc.tags) == ["mn"]
        assert session.query(Document).count() == 1


def test_admission_reports_a_completed_document(ingest_db):
    first = admit("sc:mn1/sujato")
    with ingest_db.session_scope() as session:
        session.get(Document, first.document_id).status = DocumentStatus.COMPLETED

    again = admit("sc:mn1/sujato")

    assert again.outcome == AdmissionOutcome.COMPLETED
    assert again.document_id == first.document_id
