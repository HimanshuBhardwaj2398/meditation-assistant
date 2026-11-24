# Arc: the corpus workbench

Status: in progress
Started: 2026-11-17
Journal entry: not written yet
Next: look back on Step 1 before starting the workbench UI. Explain how the client requests page 3 and decides whether there is a page 4.

Step 1 verified 2026-11-24: the API tests passed. A read-only PostgreSQL smoke check also covered the first three MN pages, MN 10 and its 15 ordered chunks, plus the 404 and 422 responses. Review removed the extra list query, added a unique paging tie-breaker, and stopped logging client errors as database errors.

The [Postman collection](../../../tests/postman/documents.postman_collection.json) passed 14 assertions across eight requests against the local API.

Status is one of: framing, in progress, writing up, done.

A FastAPI corpus API (documents, chunks, genre labels, ingestion), with a React and
TypeScript workbench on top of it. The learner asked for it on 2026-11-17, to learn API
creation and frontend basics while building something they would use.

This arc comes first in the v1 interface plan (`docs/ingestion-and-evals/README.md`),
ahead of the retrieval API arc (`2026-10-02-retrieval-api.md`). There are two
reasons. A documents and chunks API covers more of REST: collections, items, pages,
updates, a slow create and errors. And labelling chunks is useful now. The eval dataset arc
(`2026-10-03-eval-dataset.md`) needs the same "read a chunk,
judge it" screen.

## What the project has already decided

Decisions in the project's own docs that bear on this feature, each with where
it is written. Frame from these instead of deciding again.

- **Tools.** FastAPI for the REST API (`docs/ROADMAP.md`). React and TypeScript for the
  frontend, with a client generated from the API's OpenAPI schema rather than written by
  hand. The frontend speaks only HTTP and imports nothing from Python (plan README, design
  rule 6 and Phase 3). The proposed code locations are an `api/` package and a `console/`
  folder at the repo root.
- **Ingestion is slow.** It took about 19 s a document, most of it Neon round trips
  (measured 2026-08-30). The ingestion hardening of 2026-11-22
  (`docs/ingestion-and-evals/ingestion-roadmap.md`) cut a new document from about 47 round
  trips to about 32. That is still seconds, so ingest returns `202` and the client polls.
- **Submitting a source has one rule.** `IngestionOrchestrator.admit` (`ingestion/admission.py`)
  compares sources in canonical form, so `sc:MN1/sujato` and its suttacentral.net URL are one
  document. A source already COMPLETED comes back with `already_completed=True` and its id;
  step 4 decides whether the API answers that with `409` or `200`. One that never completed
  resumes, reusing its stored markdown.
- **The document row is already a progress record.** A clean run shows PARSING (set when
  the run claims the document), CHUNKING, EMBEDDING, then COMPLETED. There is no
  "persisting" state, and PARSED and CHUNKED are no longer written. A failed reprocess ends
  COMPLETED again, with the error in `status_details`, because the old version is still the
  published one.
- **One run per document, already built.** A run claims a document through
  `documents.processing_token` (migrated on Neon 2026-11-22). A second run gets
  `DocumentBusyError`, and a dead run's claim lapses after `INGESTION_LEASE_SECONDS`
  (30 minutes). This is a lease, which the October plan made the learner's core in
  Phase 2.2. On 2026-11-22 the learner chose to have Claude build it now. Phase 2.2 still owns
  the job queue itself (claiming jobs with `FOR UPDATE SKIP LOCKED`, retries); for the lease,
  it can critique `ingestion/orchestrator.py` (`_claim`) and `ingestion/status.py` instead of
  writing them.
- **Chunk metadata is stored twice.** At persistence, `ingestion/orchestrator.py` (around
  line 333) copies `chunk_metadata` into the vector store's `cmetadata`. Dense results read
  that copy, and full-text results read `chunks.chunk_metadata` (`retrieval/query.py`,
  around line 186). Editing `chunk_metadata` in place would make the two search paths
  disagree about the same chunk.
- **Genre already has a planned home.** The rumination plan
  (`docs/plans/2026-09-03-rumination-retrieval-enrichment-plan.md`, Task 6) gives genre its
  own `chunk_genre` table, versioned by whatever produced the label. That table is not built
  yet. The enrichment design (`docs/plans/enrichment-layer-design.md`) says derived data is
  a cache, not the truth.
- **Gold labels exist outside any app.** `data/labels/genre_gold_v2.jsonl` holds 482 labels
  (`genre_gold_v1.jsonl` holds 185). Each label has a mode (narrative, doctrine, practice,
  and so on) plus `is_verse` and `has_simile` flags. They are keyed by
  `(sutta_uid, chunk_index)`, not by chunk UUID. The scale-up notebook asks whether more
  labelling would help (`retrieval/genre_classifier_scaleup.ipynb`, §5).
- **Nikāya is already on every document**, in `doc_metadata` and `documents.tags`
  (`nikaya_tags` in `ingestion/suttacentral.py`).
- **Nothing pages yet.** `DocumentCRUD.get_all_documents` returns all 4,050 documents.
- **Deletes and reprocessing go through `CollectionService`**
  (`services/collection_service.py`). A delete removes the document, its chunks and its vectors
  in one transaction. A reprocess replaces the old chunks and vectors only when it publishes,
  so a failed one changes nothing searchable.
- **Callers own transactions.** CRUD methods flush but never commit. A route wraps its
  writes in one `session_scope()` (`CLAUDE.md`; `tests/test_crud_transaction_ownership.py`).
- **Not in v1:** auth, deployment, or rebuilding the Streamlit pages (plan README).
- **Real things to ingest.** The Jataka (82 suttas) and Cariyapitaka (35) were left out of
  the corpus on purpose (`EXCLUDED_KN_PREFIXES` in `scripts/ingest_all.py`), so they are
  real candidates for the first ingest from the browser.

## Demo script

Written before any design. This is what "working" means for the arc.

Claude drafted this script, and the learner chose it on 2026-11-17 without rewriting it.
It is not yet in the learner's own words, and they can still rewrite it.

- Who uses this, and to do what: the maintainer, to read the corpus, label chunk genre,
  and grow the corpus.
- How they use it:
  1. `npm run dev`, then filter the sutta list to MN.
  2. Open MN 10: its chunks appear in reading order, each with its headers and current
     label.
  3. On one chunk, set mode=practice and verse=no, then Save. After a page reload the
     label is still there.
  4. Ingest `sc:ja1`. It returns `202`, and the page shows parsing, then chunking, then
     embedding, then completed.
- What they see when it fails:
  - Ingesting a sutta that is already in the corpus says "already in the corpus" and
    links to it.
  - An invalid mode shows an error on that field instead of crashing.
  - When the API is down, a clear banner shows instead of a blank page.

## Concepts

Two to four. One resource each.

| Concept | Lens | Where it already shows up in this repo | Resource | Read it to answer |
|---|---|---|---|---|
| The shape of a resource API: collections, items, pages | software | `DocumentCRUD.get_all_documents` returns every row; `SearchResult` in `retrieval/query.py` is the closest thing to a designed response | [Web API Design Best Practices](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design) (Microsoft), the sections on resources and on filtering and pagination | How does a client ask for page 3 of the MN suttas, and how does it know a page 4 exists? |
| A screen made of components and state | usability | `views/document_detail.py`, the Streamlit reading view, which reruns the whole page on every click | [Thinking in React](https://react.dev/learn/thinking-in-react) (react.dev), steps 3 and 4 | What is the smallest state the reading view needs, and what could be computed instead of stored? |
| Where an edit lives | software | Chunk metadata stored twice (orchestrator, around line 333; query, around line 186); the planned `chunk_genre` table | [Zalando RESTful API Guidelines](https://opensource.zalando.com/restful-api-guidelines/) (look-up), the rules on PUT and PATCH | Is saving a chunk's genre a PUT or a PATCH, and what happens when the same save arrives twice? |
| Slow work behind a fast API | software | `DocumentStatus`; the orchestrator's per-stage status writes | MDN: [HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status), the `202 Accepted` entry | What does `202` promise a client, and what does it leave unpromised? |

## Steps

Each step ends with something new that can be run by hand.

| # | What you can run when it is done | Concept | Kind | Status |
|---|---|---|---|---|
| 1 | `uvicorn api.main:app`. From `/docs`, `GET /v1/documents?nikaya=mn&limit=20` returns a page with a way to get the next one, `GET /v1/documents/{id}/chunks` returns the chunks in order, and an unknown id returns a 404 | The shape of a resource API | core: the response models and the paging rule | done |
| 2 | `npm run dev` shows the sutta list and a reading view, through a client typed from `/openapi.json`. An unknown id shows "not found", and a stopped API shows a banner | A screen made of components and state | core: the reading view's fetch, with its loading, empty and error states | todo |
| 3 | Label a chunk in the reading view with `PUT /v1/chunks/{uuid}/genre`, backed by the `chunk_genre` table. The label survives a reload, and a bad mode shows inline as a 422 | Where an edit lives | core: the write path and its validation | todo |
| 4 | Ingest `sc:ja1` from the browser. `POST /v1/documents` returns `202` and a location, the page polls `GET /v1/documents/{id}` through the stages, and a duplicate returns a 409 that links to the existing document | Slow work behind a fast API | core: the `202` contract, and the rule for when polling stops | todo |
| 5 | A "Failed" filter with a Retry button (`POST /v1/documents/{id}/reprocess` through `CollectionService`), and a labels export to gold JSONL keyed by chunk UUID | Briefing an agent | delegated | todo |

- Kind: `core` means the learner writes the lines that carry the concept.
  `delegated` means the learner writes the brief and reviews the diff.
- Status: `todo`, `in progress`, `done`, or `given` (Claude wrote the core; the
  learner still owes an explanation of it).
- Claude's plumbing:
  - Adding FastAPI, uvicorn and httpx.
  - The app factory, with the database injected through a FastAPI dependency so tests run
    on in-memory SQLite.
  - The Alembic migration for `chunk_genre`.
  - The Vite and React scaffold, and the OpenAPI client generator, chosen from current
    docs at step 2.
  - The background-task wiring for ingest.
  - A failing test first, for every step.
- A known limit of step 4: the ingest runs as a background task inside the API process. If
  that process dies, the document is no longer stuck for good: its claim lapses after 30
  minutes, and the next submission resumes it. Nothing re-runs it unattended, though; that
  is the October plan's Phase 2 job queue, and this arc records the limit rather than solving
  it.
- A second limit of step 4: the pipeline's stages are `async def` but block (HTTP, the
  database, the chunking model). Run the background task in a thread (a plain `def` task does
  this in FastAPI), or each ingest freezes every other request, the polling `GET` included.

## Notes

One block per step, filled in as the step happens. The learner's words, not
Claude's. No solution code.

### Step 1: the shape of a resource API

- Opened 2026-11-17 on branch `feature/corpus-workbench`. Reading and design question given.
- Prediction (verbatim): "I think to get page three we can have query paramter offet =3 and
  limit =20. I dont know how will client know that page 4 exists , may be we will return
  additional details about number of pages in response body and then on frontend client will
  see next button if they want to go more"
- Asked along the way: "why do we need sqlite". Answer given: tests need a real database
  without a server (CI fakes external services; no local Postgres; never test against Neon).
  The cost is that the schema is Postgres-only, so `tests/api/sqlite_compat.py` shims it, and
  a green SQLite run doesn't prove the Postgres SQL. The look-back runs against Neon to cover
  that. Offered: switch to a throwaway Postgres in Docker if they prefer.
- Scaffold (Claude): `fastapi[standard]` 0.163 added. `api/` holds the app factory, a
  per-request `session_scope` dependency (`scope="function"`), the response models and the
  routes. `GET /v1/documents/{id}` and `/{id}/chunks` and their 404s are done and checked
  against Neon (MN 10 is document 14, with 15 chunks in order). `tests/api/` runs the real
  schema on a SQLite file. Of 8 tests, 4 paging tests fail at the TODOs; a throwaway fill
  passed all 8 and was removed.
- Before review, 404 and 422 responses logged an ERROR "Session error" because
  `Database.session_scope` logged any exception, including client errors thrown back into
  the dependency.

## Ideas

Things that using the app made the learner want. Candidates for the next arc.

- **Tracing and profiling for ingestion and browsing latency** (learner, 2026-11-17). The
  plan is a short follow-on arc once this one is done, rather than a step inside it:
  - **It revises a July decision, for latency only.** On 2026-07-16 the project rejected
    OpenTelemetry in favour of Langfuse behind one tracer port
    (`docs/plans/2026-07-16-retrieval-observability-agentic-seam-design.md`). Langfuse
    stays for LLM and retrieval traces. OpenTelemetry's FastAPI and SQLAlchemy
    instrumentation is what shows each Neon round trip as a span.
  - **The ingest cause is already measured.** About 11 of the 19 s go to status writes.
    The tension is that step 4's polling reads those same writes.
  - **Browsing may be slow from step 2.** The engine is pooled with `pool_pre_ping=True`
    (`db/database.py`), so each request likely pays several round trips to us-east-1
    before its query runs. If step 2 feels slow, note it here as evidence.
  - **Sketch, to be framed properly when that arc starts:**
    1. Every request and ingest produces a trace you open locally. The ingest trace shows
       the four stages and every Neon round trip, linked to the `202` request.
    2. Cut ingest's round trips without losing the polled progress, with before and after
       traces as the proof.
    3. Fix what the trace of the list and reading view shows. Reach for a sampling
       profiler (py-spy) only when a span's time isn't explained by its children.
