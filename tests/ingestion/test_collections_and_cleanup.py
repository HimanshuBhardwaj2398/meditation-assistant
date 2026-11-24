"""Each document keeps its own vector collection, and deleting one is a single transaction.

The SQLite tests use the fakes in conftest. The Postgres tests (skipped unless
INGESTION_TEST_POSTGRES_URL is set) run the real PGVector tables, where the
collection lookup and the consistency checks use JSONB operators.
"""

import pytest
from sqlalchemy import text

from core.exceptions import DocumentNotFoundError, EmbeddingError, VectorStoreError
from db.schema import Document
from db.vector_consistency import check_vector_consistency, delete_orphan_vectors
from ingestion.admission import VECTOR_COLLECTION_KEY, admit
from ingestion.claim import ReprocessMode
from ingestion.embed import VectorStoreConfig
from ingestion.orchestrator import IngestionOrchestrator
from services.collection_service import CollectionService
from tests.ingestion.conftest import SOURCE, TEST_CHUNKING
from tests.ingestion.test_orchestrator_e2e import snapshot


def orchestrator_for(collection, parser, vectors):
    return IngestionOrchestrator(
        VectorStoreConfig(collection_name=collection, db_url="sqlite://"),
        chunking_config=TEST_CHUNKING,
        parser_factory=parser,
        vector_store_manager=vectors,
    )


def recorded_collection(db, document_id):
    with db.session_scope() as session:
        return (session.get(Document, document_id).doc_metadata or {}).get(VECTOR_COLLECTION_KEY)


def collections_of(vectors):
    return {row["collection"] for row in vectors.rows()}


async def test_a_new_document_records_and_publishes_into_its_collection(ingest_db, parser, vectors):
    result = await orchestrator_for("primary", parser, vectors).ingest(SOURCE)

    assert result["success"] is True, result["errors"]
    assert recorded_collection(ingest_db, result["document_id"]) == "primary"
    assert collections_of(vectors) == {"primary"}


async def test_a_retry_from_elsewhere_stays_in_the_documents_collection(ingest_db, parser, vectors):
    vectors.fail_next = EmbeddingError("provider down")
    failed = await orchestrator_for("primary", parser, vectors).ingest(SOURCE)
    assert failed["success"] is False

    retried = await orchestrator_for("secondary", parser, vectors).ingest(SOURCE)

    assert retried["success"] is True, retried["errors"]
    assert recorded_collection(ingest_db, retried["document_id"]) == "primary"
    assert collections_of(vectors) == {"primary"}


async def test_a_reprocess_from_elsewhere_stays_in_the_documents_collection(
    ingest_db, parser, vectors
):
    first = await orchestrator_for("primary", parser, vectors).ingest(SOURCE)

    result = await orchestrator_for("secondary", parser, vectors).reprocess(
        first["document_id"], ReprocessMode.FROM_CHUNKING
    )

    assert result["success"] is True, result["errors"]
    assert collections_of(vectors) == {"primary"}


async def test_an_older_document_inherits_the_collection_its_vectors_are_in(
    ingest_db, parser, vectors
):
    legacy = admit(SOURCE)  # admitted without recording a collection
    vectors.existing_collections[legacy.document_id] = "legacy"

    result = await orchestrator_for("primary", parser, vectors).resume(legacy.document_id)

    assert result["success"] is True, result["errors"]
    assert recorded_collection(ingest_db, legacy.document_id) == "legacy"
    assert collections_of(vectors) == {"legacy"}


async def test_an_older_document_without_vectors_gets_the_configured_collection(
    ingest_db, parser, vectors
):
    legacy = admit(SOURCE)

    await orchestrator_for("primary", parser, vectors).resume(legacy.document_id)

    assert recorded_collection(ingest_db, legacy.document_id) == "primary"


def service_with(vectors):
    service = CollectionService(db_url="sqlite://")
    service._vector_store_manager = vectors
    return service


async def test_deleting_a_document_removes_its_chunks_and_every_vector(
    orchestrator, ingest_db, vectors
):
    first = await orchestrator.ingest(SOURCE)
    document_id = first["document_id"]
    with ingest_db.session_scope() as session:
        session.execute(
            text(
                "INSERT INTO fake_vectors (custom_id, document_id, document) "
                "VALUES ('left-by-an-old-run', :id, 'stale')"
            ),
            {"id": document_id},
        )

    result = await service_with(vectors).delete_document_complete(document_id)

    assert result.success is True, result.error
    assert result.chunks_deleted == first["chunk_count"]
    assert result.embeddings_deleted == first["chunk_count"] + 1
    assert vectors.rows() == []
    with ingest_db.session_scope() as session:
        assert session.get(Document, document_id) is None
        assert session.execute(text("SELECT count(*) FROM chunks")).scalar_one() == 0


async def test_a_failed_delete_deletes_nothing(orchestrator, ingest_db, vectors, monkeypatch):
    first = await orchestrator.ingest(SOURCE)
    document_id = first["document_id"]
    before = snapshot(ingest_db, document_id)
    vectors_before = vectors.rows()

    def broken(session, document_id, chunk_ids):
        raise VectorStoreError("vector store unavailable")

    monkeypatch.setattr(vectors, "delete_document_vectors", broken)

    result = await service_with(vectors).delete_document_complete(document_id)

    assert result.success is False
    assert "nothing was deleted" in result.error
    assert snapshot(ingest_db, document_id).uuids == before.uuids
    assert vectors.rows() == vectors_before


async def test_deleting_a_missing_document_says_so(ingest_db, vectors):
    with pytest.raises(DocumentNotFoundError):
        await service_with(vectors).delete_document_complete(404)


# ---------------------------------------------------------------------------
# Postgres: the real vector tables
# ---------------------------------------------------------------------------


class FixedEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def postgres_orchestrator(db, collection, monkeypatch):
    from core.interfaces import ParseResult
    from tests.ingestion.conftest import SUTTA_MARKDOWN

    orchestrator = IngestionOrchestrator(
        VectorStoreConfig(
            collection_name=collection,
            db_url=db.engine.url.render_as_string(hide_password=False),
        ),
        chunking_config=TEST_CHUNKING,
    )
    monkeypatch.setattr(orchestrator.vector_store_manager, "_embeddings", FixedEmbeddings())
    monkeypatch.setattr(
        orchestrator.parser_factory, "parse", lambda source: ParseResult(SUTTA_MARKDOWN)
    )
    return orchestrator


def vector_collections(session, document_id):
    return (
        session.execute(
            text(
                """
            SELECT DISTINCT c.name FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE e.cmetadata->>'document_id' = :id
            """
            ),
            {"id": str(document_id)},
        )
        .scalars()
        .all()
    )


@pytest.mark.integration
async def test_postgres_reprocess_keeps_the_documents_collection(postgres_database, monkeypatch):
    db = postgres_database
    alpha = postgres_orchestrator(db, "alpha", monkeypatch)
    beta = postgres_orchestrator(db, "beta", monkeypatch)
    _ = beta.vector_store_manager.vector_store  # creates the "beta" collection

    first = await alpha.ingest(SOURCE)
    assert first["success"] is True, first["errors"]
    document_id = first["document_id"]
    with db.session_scope() as session:
        assert alpha.vector_store_manager.collection_of_document(session, document_id) == "alpha"

    result = await beta.reprocess(document_id, ReprocessMode.FROM_CHUNKING)

    assert result["success"] is True, result["errors"]
    with db.session_scope() as session:
        assert vector_collections(session, document_id) == ["alpha"]


@pytest.mark.integration
async def test_postgres_consistency_report_and_orphan_repair(postgres_database, monkeypatch):
    db = postgres_database
    orchestrator = postgres_orchestrator(db, "alpha", monkeypatch)
    first = await orchestrator.ingest(SOURCE)
    assert first["success"] is True, first["errors"]
    document_id = first["document_id"]

    with db.session_scope() as session:
        assert check_vector_consistency(session).consistent

    store = orchestrator.vector_store_manager.vector_store
    with db.session_scope() as session:
        collection_id = store.get_collection(session).uuid

        def vector(custom_id, metadata):
            return store.EmbeddingStore(
                document="x",
                embedding=[1.0, 1.0],
                cmetadata=metadata,
                custom_id=custom_id,
                collection_id=collection_id,
            )

        session.add(vector("orphan-of-ours", {"document_id": document_id}))
        session.add(vector("not-ours", {"source": "some other tool"}))
        # A chunk row that lost its vector.
        session.execute(
            text(
                "DELETE FROM langchain_pg_embedding WHERE custom_id = "
                "(SELECT uuid FROM chunks ORDER BY chunk_index LIMIT 1)"
            )
        )

    with db.session_scope() as session:
        report = check_vector_consistency(session)
    assert report.orphan_vectors == 1
    assert report.orphan_vector_sample == ["orphan-of-ours"]
    assert report.chunks_without_vectors == 1
    assert report.documents_missing_vectors == [document_id]
    assert report.foreign_vectors == 1
    assert not report.consistent

    with db.session_scope() as session:
        assert delete_orphan_vectors(session) == 1
    with db.session_scope() as session:
        report = check_vector_consistency(session)
        remaining = (
            session.execute(
                text("SELECT custom_id FROM langchain_pg_embedding WHERE custom_id = 'not-ours'")
            )
            .scalars()
            .all()
        )
    assert report.orphan_vectors == 0
    assert remaining == ["not-ours"]


@pytest.mark.integration
async def test_postgres_delete_removes_every_vector_of_the_document(postgres_database, monkeypatch):
    # The service builds a PGVector store to delete through, and its Voyage
    # client insists on a key even though deleting embeds nothing. A
    # placeholder keeps the test from depending on a real key in .env.
    monkeypatch.setenv("VOYAGE_API_KEY", "placeholder-never-called")
    db = postgres_database
    orchestrator = postgres_orchestrator(db, "alpha", monkeypatch)
    first = await orchestrator.ingest(SOURCE)
    document_id = first["document_id"]
    service = CollectionService(
        collection_name="alpha", db_url=db.engine.url.render_as_string(hide_password=False)
    )

    result = await service.delete_document_complete(document_id)

    assert result.success is True, result.error
    with db.session_scope() as session:
        assert vector_collections(session, document_id) == []
        assert check_vector_consistency(session).consistent
