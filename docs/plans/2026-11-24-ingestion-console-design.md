# Ingestion console: design

Status: approved 2026-11-24
Branch: `feature/corpus-workbench`

## Goal

Ingest PDFs from a browser. Pick the vector collection each one goes into, either an existing
one or a new one. Edit a document's metadata before and after ingest. Behind the screens is a
FastAPI corpus API that the `console/` React app talks to over HTTP only.

## Relation to the corpus-workbench arc

This feature overlaps steps 1–4 of
[the corpus-workbench arc](../learning/arcs/2026-11-17-corpus-workbench.md). The maintainer chose
on 2026-11-24 to have Claude build it end to end rather than as learner-written core. The arc
steps it covers become `given`: Claude wrote the core, and the learner still owes an explanation
of it. Step 3 (genre labels on chunks) and step 5 (retry and export) are not part of this design.

## Decisions

- **Stack.** FastAPI (the existing `api/` package) and Vite, React and TypeScript in `console/`.
  The client is generated from `/openapi.json`; the generator is picked from current docs at build
  time. TanStack Query handles fetching and polling; styling is Tailwind. The Streamlit pages stay
  as they are.
- **A collection is a pgvector collection** (`langchain_pg_collection`). A document records its
  collection in `doc_metadata` when it is admitted, and later retries and reprocesses publish
  there. **The collection cannot change after ingest:** moving one would mean moving its vectors,
  which is out of scope.
- **New collections need no endpoint of their own.** If an upload names a collection that doesn't
  exist, ingestion creates it. Names must match `^[A-Za-z0-9_-]{1,63}$`.
- **Editable fields are document fields only:** title, description, tags, doc type and category.
  Chunks are read-only, because chunk metadata is stored twice (`chunks.chunk_metadata` and the
  vector `cmetadata`).
- **The title is the one field copied into chunks.** It is stored as `source_title`
  (`ingestion/metadata.py`, `chunk_tags`), and citations read it from there. A title edit rewrites
  `source_title` in both chunk copies, in the same transaction as the document row. Description,
  tags, type and category are not copied, so editing them is a plain row update.
- **Uploads are stored durably** at `data/uploads/<sha256>.pdf`, not in a temp file. A resume or
  reprocess needs the file again. Because the name is the content hash, the same PDF re-uploaded
  under another filename maps to the same source, and admission reports it as already ingested.
- **Ingest returns `202` and the client polls.** The ingest runs as a plain `def` background task,
  in a worker thread, through `services/ingestion_service.py`. The pipeline's `async` stages
  block, so a thread keeps the polling `GET` responsive. If the API process dies mid-ingest, the
  document's claim lapses after `INGESTION_LEASE_SECONDS` and the next submission resumes it.
  Nothing re-runs it unattended; the Phase 2 job queue owns that.

## API

All routes are under `/v1`.

| Method and path | Behaviour |
|---|---|
| `GET /collections` | Collection names, each with its document count |
| `GET /documents?collection=&nikaya=&status=&limit=&offset=` | One page: `items`, `total`, `limit`, `offset`, ordered by `id` so pages stay stable |
| `GET /documents/{id}` | Full detail: status, `status_details`, collection, and the editable fields. The console polls this. `404` if unknown. |
| `GET /documents/{id}/chunks` | Chunks in reading order (exists already) |
| `POST /documents/uploads` | Multipart: a `file` (PDF) and a `metadata` JSON part with `title`, `description`, `tags`, `doc_type`, `category` and `collection`. Returns `202` with a `Location` header and `{id, status}`. Returns `409` with the existing id if the document is already completed. A non-PDF gets `415`, a file over 50 MB gets `413`, and bad metadata gets `422`. |
| `POST /documents` | JSON `{source, ...metadata}` for a SuttaCentral ref or URL. The same `202` and `409` contract. |
| `PATCH /documents/{id}` | Any subset of the editable fields. Sending `collection` gets `422`. A document that is being ingested (an unexpired claim) gets `409`. Returns the updated detail. |

A source that exists but never completed is resumed and answered with `202`. Another run
already holding the claim (`DocumentBusyError`) also gets `202`, since the client can poll.

## Console screens

1. **Upload.**
   - A drop zone takes one or more PDFs. Each file becomes a card, with its title prefilled from
     the filename, plus description, tags (chips), doc type and category.
   - A searchable collection picker lists existing collections with their counts, and ends with
     "+ New collection…", whose name field validates inline. "Apply to all" sets the collection on
     every card.
   - Submitting sends one request per file. Each card then shows its stage (parsing, chunking,
     embedding, completed), polled every 2 s. Polling stops at COMPLETED or FAILED, and FAILED
     shows `status_details`.
   - A `409` shows "Already in the corpus" with a link.
   - A "From reference" tab submits a SuttaCentral ref or URL.
2. **Documents.** A paged table with filters for collection, nikāya and status. Each row links to
   its detail page.
3. **Document detail.** An editable metadata form whose Save sends a `PATCH`. A `422` shows next
   to the field it is about. Collection is read-only. The form is disabled while the document is
   being ingested. The chunks are listed read-only below it.

Every screen shows an "API unreachable" banner when a request fails to connect.

In dev, a Vite proxy forwards `/v1` to `127.0.0.1:8000`, so no CORS setup is needed. There is no
auth and no deployment in v1.

## Errors

- Client errors (`404`, `409`, `413`, `415`, `422`) use FastAPI's `{"detail": ...}` shape. A `409`
  also carries `existing_id`.
- `Database.session_scope` currently logs an ERROR for every client error thrown back into the
  session dependency. It is fixed so that only real failures log at ERROR.
- An ingest failure doesn't fail the `202` request. It shows on the document as
  `status_details`, which the console displays.

## Testing

- **API tests** in `tests/api/`, on SQLite with the pipeline faked:
  - an upload returns `202` and the file is stored
  - a duplicate upload returns `409`
  - a non-PDF gets `415`, a bad collection name gets `422`, and an oversized file gets `413`
  - a title `PATCH` rewrites `source_title` in both chunk copies
  - a `PATCH` sending `collection` gets `422`, and one mid-ingest gets `409`
  - paging is stable
  - filters work
- **Postgres:** the title rewrite is JSONB on two tables, so its test also runs against local
  Docker Postgres. A green SQLite run doesn't prove the Postgres SQL.
- **Console:** `tsc --noEmit`, plus one Playwright smoke run against the live stack: upload a small
  PDF, see COMPLETED, edit the title, and reload to confirm the edit stuck.

## Out of scope

Moving a document between collections, editing chunk metadata, genre labelling (arc step 3),
retry and export (arc step 5), auth, deployment, and a durable job queue.
