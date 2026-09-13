# Migration Plan — Veridoc Document Intelligence Platform

This document is the running record of what's shipped, what was ported
from the pre-Veridoc baseline, and what's still ahead — produced up front
per the product spec's own instructions (don't rewrite everything at
once; plan first, build in tested increments) and kept in sync with the
actual code as later increments land, rather than describing an earlier
state of the project as current.

## 1. What the pre-Veridoc baseline was

The project started from a single FastAPI service (~2,900 lines)
implementing grounded Q&A over a small fictional legal corpus:

- **Retrieval**: Gemini embeddings (`gemini-embedding-001`, L2-normalized,
  truncated to a configurable dimension) → Pinecone serverless index →
  lightweight BM25 reranking (no model download) → Groq-hosted LLM
  (`llama-3.1-8b-instant`) grading → grounded answer generation → citation
  validation, orchestrated as a LangGraph state machine with a bounded
  retry loop (`retrieve → rerank → grade_chunks → generate_answer →
  validate_citations`, with `rewrite_query` looping back into `retrieve` on
  insufficient evidence, up to `MAX_RETRIEVAL_LOOPS`).
- **Ingestion**: `.md`/`.txt`/`.pdf`/`.docx` loaders flattening every format
  to plain text; markdown-header-aware chunking with deterministic chunk
  IDs (`<source_root>::<filename>::<section-slug>::<index>`); idempotent
  upsert (unchanged chunks skipped via content-hash comparison) and
  stale-vector reconciliation across two directories (`data/corpus/`,
  immutable; `data/uploads/`, mutable, written by `POST /upload`).
- **API**: `GET /health`, `GET /ready`, `POST /ask`, `POST /upload`, plus a
  small vanilla-JS static frontend.
- **Tests**: ~17 files covering chunking, grading, citations, retrieval,
  reranking, routing, ingestion, and the API.

### What was reusable as-is (ported with import-path changes only)

`app/reranker.py` (BM25), `app/retrieval.py` (embed → search → threshold →
pool/dedupe), `app/clients.py` (Gemini/Groq/Pinecone client factories,
index-dimension validation, index-ready polling), the LangGraph shape
itself (`app/graph.py` → `app/rag/graph.py`, renamed `QAService` →
`DocumentQAService`). None of this logic is legal-domain-specific — it's
generic retrieval/reranking/orchestration engineering.

### What needed rewriting for genericization only

`app/llm.py` → `app/rag/llm.py`: the grading/rewrite/answer prompts
contained legal-domain examples ("non-compete clause", "hearing notice")
baked into the few-shot examples. The underlying design (label-indirection
via `CANDIDATE_N`/`EVIDENCE_N` instead of asking the model to reproduce a
`chunk_id`; the provider-error-vs-malformed-output distinction in
`_chat_json`) is domain-independent and was preserved unchanged — only the
prompt examples were rewritten around generic document scenarios (SaaS
pricing, HR policy, warehouse logs).

### What was 100% new build (Phase 1's actual scope)

The spec's core ask — structured, layout-aware, multi-format ingestion with
real table extraction and OCR — did not exist in the baseline at all.
`app/loaders.py` flattened every format straight to a plain string; a table
in a DOCX became `" | "`-joined prose indistinguishable from a sentence,
and PDF/DOCX/image formats had no OCR path, no table structure, and no page
numbers. Building this properly required a new document model layer,
described below.

## 2. What's shipped

**Rebrand + generic document model** (initial build): new name (Veridoc),
generic sample corpus (SaaS product docs, HR policy, meeting notes, a
financial report excerpt, a warehouse log) replacing the fictional legal
corpus, no legal-domain framing left in code or prompts. A structured
document model (`app/documents/models.py`) — every parser produces the
same `ExtractedDocument` (pages, `Block`s in reading order, each with page
number, bounding box, extraction source, confidence) — flattening to
plain text happens at the *chunking* boundary, not the *parsing*
boundary, so page numbers, headings, and tables survive as structured
data all the way to the vector store. OCR (`app/documents/ocr/`,
Tesseract), native + heuristic table extraction
(`app/documents/tables/`), structure-aware chunking (tables always get
their own chunk), a PostgreSQL-ready SQLAlchemy metadata layer
(`Document`/`DocumentVersion`/`IngestionRun`), and the FastAPI domain
restructure (`api/`, `core/`, `db/`, `documents/`, `rag/`, `services/`,
`schemas/`). Format coverage: `.md`, `.txt`, `.pdf` (native + scanned),
`.docx`, `.png`/`.jpg`/`.jpeg`.

**Next.js frontend** (see [`../frontend`](../frontend/README.md)): chat
UI (`/ask`) with expandable "evidence stamp" citations carrying
page/table/OCR-confidence provenance, a document library (`/library`)
showing per-document page/table/chunk/OCR stats from `GET /documents`,
and an upload flow (`/upload`) with per-file status and an ingestion
summary. See its README for its own known limitations (no auth, no chat
persistence, no document deletion UI — none of these exist on the
backend yet either).

**Hybrid retrieval + agentic hardening**: a corpus-wide BM25 lexical
index (`app/lexical_index.py`) fused with dense Pinecone retrieval via
Reciprocal Rank Fusion (`app/rag/fusion.py`), cross-encoder reranking
with an honest, tracked fallback to BM25 when the model is unavailable
(`app/rag/cross_encoder.py` — every response reports
`configured_reranker`/`actual_reranker`/`fallback_used`), deterministic
query classification (`app/rag/query_classifier.py`) as a routing hint
only (never the final answerability authority), and claim-level
grounding (`app/rag/grounding.py`) as additive metadata layered on top
of — never overriding — application-side citation validation. A 47
-question benchmark (`evaluation/datasets/v1.jsonl`) and an ablation
runner (`evaluation/run_ablation.py`) covering configurations A
(dense-only) through E (+ claim grounding); see
[`evaluation/README.md`](../backend/evaluation/README.md) for what's
actually been run versus what's real-but-unexecuted code.

**Phase 3A hardening pass** (this delivery): two correctness fixes to
upload/ingestion, plus a documentation sync.

1. **Lexical index artifact/lifecycle.** The repo used to ship a stale,
   dummy `data/lexical_index.json`. It's now a purely generated runtime
   artifact — removed from the repo and `.gitignore`d — and
   `ingest_corpus` writes it *last*, only once the Pinecone upsert and
   stale-vector cleanup for the same run have actually succeeded, so it
   can never describe chunks that were never really indexed.
2. **Atomic upload/index consistency.** `POST /documents/upload` used to
   write files straight into the live `data/uploads/` directory before
   the chunk/embed/index pipeline ran; if that pipeline failed partway,
   the file was left behind looking exactly like a successfully-ingested
   one. Uploads now stage in a temp directory, validate there, and are
   only moved into `data/uploads/` immediately before indexing — reverted
   if indexing fails. Separately, `ingest_corpus`'s failure path used to
   commit whatever partial `Document`/`DocumentVersion` rows an
   in-progress run had already staged (a `finally` block force-committed
   the session so the `IngestionRun` audit row would persist even on
   failure — which also committed everything else pending in that same
   session). The ingestion-work transaction and the audit-row transaction
   are now separate: a failure rolls back the former and still records
   the latter. See `app/api/documents.py` and
   `app/services/ingestion_service.py`'s docstrings, and
   `tests/test_upload_atomicity.py` /
   `test_ingest_failed_run_does_not_persist_partial_documents` for the
   regression tests.
3. **Documentation sync.** This file, `backend/README.md`,
   `docs/architecture.md`, `evaluation/README.md`, and
   `frontend/README.md` no longer describe an earlier phase as current —
   in particular, `evaluation/README.md` used to say retrieval-ranking
   metrics (Recall@K/MRR/nDCG) weren't wired into the ablation runner;
   that was already fixed in the code and the doc just hadn't caught up.

## 3. Known limitations

Documented here rather than hidden, per the spec's "no fake metrics/
features" instruction:

- **No font-based heading detection for native PDF text.** Headings are
  only detected from Markdown `#`/`##` syntax and DOCX "Heading N" styles.
  A PDF with no `pdfplumber`-visible structure falls into chunking's
  single "content" section fallback — correct, but coarser than it could
  be with font-size-based heading inference (see "Roadmap" below).
- **OCR table reconstruction is heuristic**, not a trained table-detection
  model. It works well on clearly-gridded, left-aligned tabular text (see
  `tests/test_table_reconstruction.py`) and degrades gracefully (lower
  confidence, or falling back to plain paragraph text) on rotated,
  hand-written, or heavily skewed tables. `docs/tables.md` documents this
  precisely so no citation ever implies more precision than the pipeline
  actually has.
- **No Alembic migrations yet.** `init_db()` calls `create_all`, which is
  additive-only. The first real schema change needs a proper migration
  tool before it ships.
- **Single-process, no distributed task queue.** Ingestion runs
  synchronously in the request thread for `POST /documents/upload`; a
  large batch of scanned PDFs will block that request for the OCR
  duration.
- **No RAGAS/TruLens/Opik integration yet.** The evaluation layer is
  deterministic-metrics-only — see `evaluation/README.md`, "Known gaps".
- **Benchmark is 47 questions**, not the 200-300+ target.
- **No auth/multi-tenancy, rate limiting, or object storage yet** —
  local filesystem + SQLite/Postgres only.

## 4. Roadmap

Follow-on work, roughly in order (see the product spec for the full
breakdown):

- **Evaluation & observability**: RAGAS, TruLens, Opik integration
  alongside the existing deterministic metrics; expand the benchmark
  toward 200-300+ questions covering citation traps, near-match
  retrieval, conflicting evidence, and prompt injection; TTFT/streaming
  measurement; an actual dashboard.
- **Advanced document intelligence**: OCR preprocessing (deskew,
  denoising, adaptive thresholding), layout-aware heading detection
  (font size/weight/position, not just markup), richer table structure
  (merged cells, multi-row/column headers, multi-page tables), figure/
  chart understanding.
- **Product features**: a document viewer with citation click-through
  and bounding-box highlighting, streaming chat, multi-document
  reasoning, comparison, structured extraction, summarization,
  conversation memory, report generation.
- **Production infrastructure**: PostgreSQL + Alembic in place of
  `create_all`, object storage in place of local filesystem, a
  background ingestion worker in place of synchronous upload handling,
  auth, multi-tenancy, rate limiting, file security hardening, prompt
  injection defenses, Docker/CI/CD, deployment, monitoring.

## 5. Verification performed

- Full corpus (all 6 formats, including a synthetic scanned image with a
  table) chunked end-to-end; table detection verified on native PDF,
  native DOCX, Markdown pipe syntax, and OCR image paths.
- Ingestion service tested end-to-end against a real SQLite DB with
  Pinecone/Gemini calls mocked: first-run creation, unchanged-chunk skip
  on a second run, document version bump on content change, a failed run
  correctly recorded in `IngestionRun` *without* committing partial
  document/version state, and the lexical snapshot only being written on
  a fully-successful run.
- Atomic upload workflow verified end-to-end through the API layer: a
  failed ingestion leaves no file behind in `data/uploads/` and doesn't
  disturb files from an earlier, successful upload batch.
- **162 automated tests passing** (`pytest tests/ -q`, verified in a
  clean virtualenv against `requirements.txt` in this session — see
  `backend/README.md`, "Tests"), covering document parsing, OCR table
  reconstruction, chunking, the DB layer, the ingestion service, LLM
  grading/answer/citation/grounding logic, vectorstore metadata
  round-tripping, and the API layer. The optional `sentence-transformers`
  cross-encoder dependency was not installed in this environment (disk
  constraints); the suite doesn't need it — the cross-encoder path is
  exercised through its documented BM25 fallback.
