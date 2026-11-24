"""Fixtures for API tests: the real app on a throwaway SQLite file."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config.settings import DatabaseSettings
from db.database import Database
from db.schema import Base, Chunk, Document, DocumentStatus
from tests import sqlite_compat

sqlite_compat.install()

MN_COUNT = 45
DN_COUNT = 5


@pytest.fixture
def database(tmp_path) -> Database:
    # A file, not :memory: — TestClient serves requests on another thread, and an
    # in-memory SQLite database is private to the connection that made it.
    db = Database(DatabaseSettings(url=f"sqlite:///{tmp_path / 'api.db'}"))
    Base.metadata.create_all(db.engine)
    return db


@pytest.fixture
def corpus(database) -> dict:
    """45 MN suttas with 5 DN suttas interleaved, and three chunks on MN 10.

    Returns the ids, so tests never assume what the database assigned.
    """
    mn_ids: list[int] = []
    dn_ids: list[int] = []
    with database.session_scope() as session:
        mn_n = dn_n = 0
        for i in range(1, MN_COUNT + DN_COUNT + 1):
            if i % 10 == 0:
                dn_n += 1
                uid, nikaya, ids = f"dn{dn_n}", "dn", dn_ids
            else:
                mn_n += 1
                uid, nikaya, ids = f"mn{mn_n}", "mn", mn_ids
            doc = Document(
                title=uid.upper(),
                file_path=f"sc:{uid}/sujato",
                doc_metadata={"source": "suttacentral", "uid": uid, "nikaya": nikaya},
                status=DocumentStatus.COMPLETED,
                # Force a timestamp tie so paging must use the document IDs too.
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            session.add(doc)
            session.flush()
            ids.append(doc.id)

        mn10_id = mn_ids[9]
        # Inserted out of order on purpose: reading order must come from chunk_index.
        for index in (2, 0, 1):
            session.add(
                Chunk(
                    uuid=f"mn10-chunk-{index}",
                    document_id=mn10_id,
                    chunk_text=f"Text of chunk {index}.",
                    chunk_index=index,
                    chunk_metadata={"all_header_paths": [f"MN 10 > Section {index}"]},
                )
            )
    return {"mn_ids": mn_ids, "dn_ids": dn_ids, "mn10_id": mn10_id}


@pytest.fixture
def client(database) -> TestClient:
    with TestClient(create_app(database)) as test_client:
        yield test_client
