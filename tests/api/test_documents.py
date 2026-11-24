"""Step 1 of the corpus workbench: the document collection, paged, and its chunks.

The paging tests check the current page, the matching total, and the paging parameters.
"""

import logging

from db.schema import Document
from tests.api.conftest import DN_COUNT, MN_COUNT


def test_page_three_of_mn_and_whether_page_four_exists(client, corpus):
    # 45 MN suttas at 20 a page: pages hold 20, 20, 5. Page 3 starts after 40.
    page3 = client.get("/v1/documents", params={"nikaya": "mn", "limit": 20, "offset": 40})
    everything = client.get("/v1/documents", params={"nikaya": "mn", "limit": 100})

    assert page3.status_code == 200
    body = page3.json()
    assert body["total"] == MN_COUNT
    assert (body["limit"], body["offset"]) == (20, 40)
    assert [doc["id"] for doc in body["items"]] == [
        doc["id"] for doc in everything.json()["items"][40:]
    ]
    # Page 4 would start at 60; nothing is left after 40 + 5.
    assert 40 + len(body["items"]) >= body["total"]


def test_walking_the_pages_visits_every_mn_sutta_exactly_once(client, corpus):
    seen: list[int] = []
    offset = 0
    while True:
        body = client.get(
            "/v1/documents", params={"nikaya": "mn", "limit": 20, "offset": offset}
        ).json()
        seen += [doc["id"] for doc in body["items"]]
        offset += 20
        if offset >= body["total"]:
            break

    assert len(seen) == len(set(seen)), "a document appeared on two pages"
    assert seen == sorted(corpus["mn_ids"], reverse=True)


def test_without_a_filter_the_total_is_the_whole_corpus(client, corpus):
    body = client.get("/v1/documents", params={"limit": 5}).json()

    assert body["total"] == MN_COUNT + DN_COUNT
    assert len(body["items"]) == 5


def test_a_page_past_the_end_is_empty_not_an_error(client, corpus):
    response = client.get("/v1/documents", params={"nikaya": "mn", "offset": 1000})

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == MN_COUNT


def test_the_server_bounds_the_page_size(client, corpus):
    assert client.get("/v1/documents", params={"limit": 500}).status_code == 422
    assert client.get("/v1/documents", params={"limit": 0}).status_code == 422
    assert client.get("/v1/documents", params={"offset": -1}).status_code == 422


def test_one_document(client, corpus):
    response = client.get(f"/v1/documents/{corpus['mn10_id']}")

    assert response.status_code == 200
    assert response.json()["uid"] == "mn10"
    assert response.json()["nikaya"] == "mn"


def test_document_detail_exposes_reader_metadata_without_internal_fields(client, corpus, database):
    document_id = corpus["mn10_id"]
    with database.session_scope() as session:
        doc = session.get(Document, document_id)
        doc.description = "A discourse on the foundations of mindfulness."
        doc.tags = ["meditation"]
        doc.markdown = "This is fetched through the chunks endpoint."
        doc.status_details = "An internal status message."
        doc.doc_metadata = {
            **doc.doc_metadata,
            "type": "ancient_text",
            "author_uid": "sujato",
            "reading_url": "https://suttacentral.net/mn10/en/sujato",
            "internal_marker": "unused",
        }

    response = client.get(f"/v1/documents/{document_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["description"] == "A discourse on the foundations of mindfulness."
    assert body["tags"] == ["meditation"]
    assert body["doc_type"] == "ancient_text"
    assert body["author_uid"] == "sujato"
    assert body["reading_url"] == "https://suttacentral.net/mn10/en/sujato"
    assert {
        "file_path",
        "markdown",
        "status_details",
        "doc_metadata",
        "internal_marker",
    }.isdisjoint(body)


def test_a_document_with_an_invalid_stored_reading_url_can_still_be_read(client, corpus, database):
    document_id = corpus["mn10_id"]
    with database.session_scope() as session:
        doc = session.get(Document, document_id)
        doc.doc_metadata = {**doc.doc_metadata, "reading_url": "not a web URL"}

    response = client.get(f"/v1/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["reading_url"] is None


def test_chunks_come_back_in_reading_order(client, corpus):
    response = client.get(f"/v1/documents/{corpus['mn10_id']}/chunks")

    assert response.status_code == 200
    items = response.json()["items"]
    assert [chunk["chunk_index"] for chunk in items] == [0, 1, 2]
    assert items[0]["headers"] == ["MN 10 > Section 0"]


def test_an_unknown_document_is_a_404(client, corpus, caplog):
    assert client.get("/v1/documents/999999").status_code == 404
    assert client.get("/v1/documents/999999/chunks").status_code == 404
    assert not any(
        record.name == "db.database" and record.levelno >= logging.ERROR
        for record in caplog.records
    )
