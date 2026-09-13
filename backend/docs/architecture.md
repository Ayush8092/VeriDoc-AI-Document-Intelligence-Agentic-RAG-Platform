# Architecture

## Overview

Veridoc is a document-grounded Q&A backend. A document goes in (PDF, DOCX,
TXT, MD, PNG/JPG — scanned or native), gets parsed into a structured
representation, chunked, embedded, and stored in Pinecone; a question comes
in, gets answered strictly from retrieved chunks, with citations that carry
enough provenance (page, table cell, OCR confidence) to verify the answer
against the source.

```
                 ┌─────────────────────────── ingestion ───────────────────────────┐
 file  ──▶  documents/pipeline.py  ──▶  chunking.py  ──▶  embeddings  ──▶  Pinecone
 (pdf/       (dispatch by ext)          (structure-        (Gemini)         (vectors +
  docx/       │                          aware, table-                       metadata)
  txt/md/     ├─ parsers/text_parser.py  aware chunks)
  png/jpg)    ├─ parsers/docx_parser.py
              ├─ parsers/pdf_parser.py  ──▶ ocr/ (scanned pages)
              └─ parsers/image_parser.py──▶ ocr/ + tables/reconstruct.py
                                                            │
                                                            ▼
                                                   db/models.py (Document,
                                                   DocumentVersion metadata)
```

```
                                          ─────────────────────────── ask ───────────────────────────
question ──▶ rag/query_classifier.py ──▶ retrieval.py (hybrid) ──▶ rerank ──▶ rag/llm.py (grade/rewrite/answer)
                  │                        │        │                │              │
                  │                        │        └─ lexical_index.py (BM25,      ├─ citation validation
                  │                        │           corpus-wide) ──▶ rag/fusion.py│  (code, not LLM-trusted)
                  │                        └─ vectorstore.py (Pinecone, dense)       └─ rag/grounding.py
                  │                                 │                                  (claim-level, after
                  └── UNANSWERABLE short-circuits    └─ rag/cross_encoder.py             citation validation)
                      straight to refusal               (falls back to reranker.py/BM25 if unavailable)
```

## Module layout

- `app/core/` — settings (`config.py`), the single source of truth for
  every environment variable.
- `app/db/` — SQLAlchemy models + session for document/ingestion
  *metadata* (not vectors — those live in Pinecone). Defaults to local
  SQLite; set `DATABASE_URL` to a `postgresql+psycopg2://...` URL in
  production.
- `app/documents/` — the format-agnostic document model
  (`models.py`) and everything that produces it: `parsers/` (one module
  per format family), `ocr/` (Tesseract wrapper + scanned-PDF detection),
  `tables/` (OCR-based table reconstruction heuristic).
- `app/chunking.py` — walks an `ExtractedDocument`'s blocks into
  deterministic, context-rich `Chunk`s. Tables always get their own chunk.
- `app/clients.py`, `app/vectorstore.py`, `app/retrieval.py`,
  `app/reranker.py` — provider clients, Pinecone read/write, dense
  retrieval, lexical reranking. Ported from the baseline project largely
  unchanged (see MIGRATION_PLAN.md) — this logic was already
  domain-independent.
- `app/lexical_index.py` — corpus-wide BM25 retrieval, an
  independent retriever fused with dense search via `app/rag/fusion.py`
  (Reciprocal Rank Fusion) — distinct from `app/reranker.py`, which only
  reorders a small candidate set Pinecone already returned. The snapshot
  it reads (`data/lexical_index.json`) is a generated runtime artifact,
  never committed — see `app/services/ingestion_service.py`'s
  `ingest_corpus` (it's written last, only after Pinecone indexing for
  the same run has actually succeeded, so it can never describe chunks
  that were never really indexed).
- `app/rag/` — `llm.py` (grading/rewrite/answer prompts + parsing),
  `graph.py` (the LangGraph state machine, `DocumentQAService`),
  `query_classifier.py` (query understanding), `fusion.py`
  (Reciprocal Rank Fusion), `cross_encoder.py` (cross-encoder reranking,
  falls back to `app/reranker.py`'s BM25 when unavailable), `bm25.py`
  (the shared BM25 scorer both the reranker and the lexical index build
  on), `grounding.py` (claim-level grounding).
- `app/evaluation/` — deterministic evaluation metrics:
  `retrieval_metrics.py` (Recall@K/MRR/nDCG) and `metrics.py` (grounded
  answer rate, refusal accuracy, citation precision/recall, latency
  percentiles). See `evaluation/README.md` (repo-root-relative,
  outside `app/`) for the benchmark dataset and ablation runner that use
  these.
- `app/services/ingestion_service.py` — the one ingestion entry point
  both the CLI and `POST /documents/upload` call. Owns idempotency
  (content-hash skip), stale-vector cleanup, document-version
  bookkeeping in the DB, and writes the BM25 lexical index
  snapshot `app/lexical_index.py` reads at query time. The ingestion-work
  DB transaction and the `IngestionRun` audit-row transaction are
  deliberately separate (see the function's docstring) so a failed run
  can never leave partial `Document`/`DocumentVersion` rows committed.
- `app/api/` — FastAPI routers: `health.py`, `ask.py`, `documents.py`.
  `documents.py`'s upload handler stages files in a temp directory,
  validates them, and only moves them into `data/uploads/` immediately
  before indexing — reverted if indexing fails — so a half-failed upload
  can never leave a file that looks indexed but isn't (see the module
  docstring).
- `app/main.py` — assembles the FastAPI app, CORS, lifespan (background
  RAG-service warm-up), and a minimal static fallback page (the real UI
  is the Next.js app in `../frontend`).

## Idempotent, reconciled ingestion

Chunk IDs are deterministic
(`<source_root>::<filename>::<section-slug>::<index>`), so re-running
ingestion overwrites the same Pinecone vectors instead of creating
duplicates. Unchanged chunks (matched by content hash) are skipped before
the embedding call. Every run computes
`stale_ids = <every ID in the namespace> - <current corpus's chunk IDs>`
across *both* `data/corpus/` (immutable) and `data/uploads/` (mutable,
written by `POST /documents/upload`) together in one pass, and deletes
exactly those — reconciling the two directories separately would make
each run see the other's vectors as stale and delete them by mistake.

A file's own content hash (distinct from each chunk's hash) is used to
detect a new *document version*: `DocumentVersion.version_number` bumps
only when the source file's bytes actually change, giving every citation
a stable answer to "which version of this document was this extracted
from".

## Grounding and citation guarantees

The LLM never sees or returns a raw `chunk_id` — it sees short
`CANDIDATE_N`/`EVIDENCE_N` labels, and application code maps those back to
real chunk IDs. `validate_citations` then re-checks every cited ID against
the chunks *actually retrieved for this request* and builds citation
metadata directly from that chunk — never from model output — so a
fabricated citation cannot reach the API response regardless of what the
model claims.

`app/rag/grounding.py` goes one level deeper, after citation
validation: it splits the answer into short factual claims and checks
each one against the specific chunk(s) cited for it, so a citation that
points at a real, retrieved chunk but doesn't actually support what the
sentence next to it says gets caught (`citation_precision` in the API
response) rather than silently passing as "cited, therefore grounded".
This is additive metadata — it can never override `validate_citations`'s
refuse/answer decision, only annotate an already-valid answer.

## Query understanding and hybrid retrieval

`app/rag/query_classifier.py` runs before retrieval and assigns one of
ten query types (LOOKUP, MULTI_HOP, COMPARISON, EXTRACTION,
SUMMARIZATION, CROSS_DOCUMENT, TABLE_QUERY, OCR_QUERY, AMBIGUOUS,
UNANSWERABLE) using deterministic rules by default. Only UNANSWERABLE
short-circuits straight to refusal, skipping a retrieval round trip for
empty input/greetings/small talk; every other type still goes through
retrieval and grading as before — classification widens the retry
fan-out for types that inherently span more than one fact
(`app/rag/query_classifier.py`'s `WIDE_FANOUT_TYPES`) but never decides
answerability on its own.

Retrieval itself (`app/retrieval.py`'s `hybrid_retrieve_*` functions) runs
dense (Pinecone) and BM25 (`app/lexical_index.py`, corpus-wide, not just
reranking a candidate set) searches independently and fuses them with
Reciprocal Rank Fusion (`app/rag/fusion.py`) before the candidate set
reaches reranking — this is what actually surfaces lexically-exact
matches (SKU codes, dollar figures, proper nouns) a dense embedding alone
sometimes under-ranks. `HYBRID_RETRIEVAL_ENABLED=false` degrades cleanly
to the original dense-only behavior.

## Limitations

- **No font-based heading detection for native PDF text.** Headings are
  only detected from Markdown `#`/`##` syntax and DOCX "Heading N"
  styles; layout-based detection (font size/weight/position) is future
  work — see MIGRATION_PLAN.md, "Roadmap".
- **OCR table reconstruction is heuristic**, not a trained table-detection
  model — see `docs/tables.md` for exactly where it degrades gracefully
  (rotated/hand-written/heavily-skewed tables) rather than overclaiming
  precision.
- **The retrieval ablation study has real, working code but has not
  actually been *run* in this environment** (no live API keys were
  available) — `evaluation/README.md`, "Status" and "Known gaps", has the
  full picture; there are no measured numbers to report yet.
- **`LocalVectorIndex`/`LocalPineconeClient` (`app/vectorstore_local.py`,
  the default `VECTORSTORE_BACKEND=local` free/$0 mode) do a full
  read-modify-write of one JSON file per index on every mutation, and
  brute-force cosine similarity over every vector in a namespace on
  every query** — correct and simple, matching this project's corpus
  scale (see that module's docstring for why this mirrors
  `app/lexical_index.py`'s own "reload the whole thing" philosophy), but
  it is NOT what you'd want for a namespace with many tens of thousands
  of vectors. `VECTORSTORE_BACKEND=pinecone` remains available (opt-in,
  requires `PINECONE_API_KEY`) for that scale.
- **`S3StorageBackend` (`app/storage/s3.py`) has not been runtime
  -verified against a live S3 or S3-compatible (MinIO) endpoint in every
  environment this project has been developed in** — see
  `tests/test_storage.py`'s module docstring for exactly what IS
  covered (a hand-rolled fake boto3 client, when `boto3` itself wasn't
  installable) versus what still needs a real MinIO/AWS run before a
  first production S3 deployment.

The following were previously listed here as known gaps and have since
been fixed/implemented (not merely re-labeled) — kept below as a record
of what changed, so a reader of an older version of this document (or an
LLM re-reading a stale cached copy of it) doesn't propagate the
outdated claim:

- ~~"No Alembic migrations yet"~~ — `alembic/` now has a full migration
  history (see `alembic/versions/`); `init_db()`'s `create_all` remains
  as a convenience for local dev/tests only.
- ~~"Ingestion runs synchronously... a background job queue is future
  work"~~ — `POST /documents/upload` now persists the file and queues an
  `IngestionJob` row, returning `202 Accepted` immediately;
  `app.services.ingestion_jobs.worker`/`run_job` process it in the
  background. See `app/api/documents.py` and
  `app/services/ingestion_jobs.py`'s module docstrings.
- ~~"RAGAS / TruLens / Opik are not implemented yet"~~ — all three are
  implemented and optionally wired in: `app/evaluation/ragas_metrics.py`,
  `app/evaluation/trulens_feedback.py`,
  `app/observability/opik_integration.py`, orchestrated together by
  `evaluation/run_phase4_eval.py`. Each degrades independently and
  clearly (a typed `*UnavailableError`, not a bare import crash) when
  its optional dependency/credential isn't present — see those modules'
  docstrings for exactly what's been runtime-verified versus not.

Two upload/ingestion correctness issues found during a hardening pass
have been fixed (not merely documented as known gaps): uploaded files
used to be written into the live `data/uploads/` directory before the
full chunk/embed/index pipeline ran, and a failed ingestion run could
leave partial `Document`/`DocumentVersion` rows committed to the
database — see `app/api/documents.py` and
`app/services/ingestion_service.py`'s module/function docstrings for how
the atomic staging + transaction-split fixes work, and
`tests/test_upload_atomicity.py` / `tests/test_ingestion_service.py`'s
`test_ingest_failed_run_does_not_persist_partial_documents` for the
regression tests.