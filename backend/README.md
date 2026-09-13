# Veridoc — AI Document Intelligence Platform (Backend)

Document-grounded Q&A over a mixed-format corpus (PDF, DOCX, TXT, MD, and
scanned/photographed documents via OCR), with real table extraction and
citations that carry page/table provenance. Answers are restricted to
retrieved evidence; unsupported questions are explicitly refused rather
than guessed at.

This is the backend API + ingestion pipeline. The product's UI is the
Next.js app in [`../frontend`](../frontend/README.md) (chat, document
library, upload) — see its README to run it. `MIGRATION_PLAN.md` has the
full history of what's shipped, what's deliberately deferred, and known
limitations, and is kept in sync with the actual code rather than
describing an earlier state of the project.

## What's here

- **Multi-format ingestion** with a shared structured document model
  (`app/documents/`): native PDF/DOCX text and tables extracted directly;
  scanned PDFs and PNG/JPG images OCR'd with Tesseract, including a
  heuristic table-reconstruction pass. See [`docs/ocr.md`](./docs/ocr.md)
  and [`docs/tables.md`](./docs/tables.md).
- **Structure-aware chunking** (`app/chunking.py`): real headings define
  sections; a table is always its own chunk, never flattened into prose.
- **Grounded Q&A** via a LangGraph pipeline (retrieve → rerank → grade →
  generate → validate citations), with a bounded retry loop on
  insufficient evidence. See [`docs/architecture.md`](./docs/architecture.md).
- **Idempotent, reconciled ingestion**: deterministic chunk IDs, unchanged
  -content skip, stale-vector cleanup, and document-version tracking in a
  SQL database (SQLite for dev, PostgreSQL-ready for production).

## Requirements

- Python 3.11+
- **Free/local by default**: `VECTORSTORE_BACKEND=local` (a file-persisted,
  Pinecone-API-compatible index — see `app/vectorstore_local.py`),
  `STORAGE_BACKEND=local`, and `DATABASE_URL` defaulting to local SQLite
  are all the out-of-the-box defaults — no Pinecone/S3/PostgreSQL account
  needed to run this at all. See `.env.example`'s "RUN VERIDOC FOR $0"
  section.
- The only credentials actually required are `GEMINI_API_KEY`
  (embeddings) and `GROQ_API_KEY` (LLM answers) — both have free tiers —
  and only for the endpoints/scripts that call an AI provider (ingest,
  `/ask`, evaluation runners). The app starts and most endpoints work
  without them.
- Optional, opt-in cloud infrastructure (never required): a real hosted
  Pinecone index (`VECTORSTORE_BACKEND=pinecone`), S3-compatible object
  storage (`STORAGE_BACKEND=s3`), PostgreSQL (`DATABASE_URL=postgresql+...`),
  Redis-backed rate limiting (`RATE_LIMIT_REDIS_URL`).
- System packages for OCR: `tesseract-ocr`, `poppler-utils`

```bash
apt-get install -y tesseract-ocr poppler-utils   # Debian/Ubuntu
# macOS: brew install tesseract poppler
```

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# fill in GEMINI_API_KEY and GROQ_API_KEY in .env — everything else has
# a working free/local default (see "Requirements" above)
```

## Ingest the sample corpus

```bash
python -m app.services.ingestion_service          # idempotent
python -m app.services.ingestion_service --reset   # wipe + re-ingest everything
```

This chunks and embeds every file in `data/corpus/` (the shipped sample
corpus — see `data/corpus/README.txt`) plus `data/uploads/` (empty until
you upload something), and records document/version metadata in the
database (`DATABASE_URL`, defaults to `./data/veridoc.db`).

## Run the API

```bash
uvicorn app.main:app --reload
```

- `GET /health` — liveness, no external calls
- `GET /ready` — readiness (tries to build the RAG service)
- `POST /ask` — `{"question": "..."}` → grounded answer + citations
- `POST /documents/upload` — multipart upload → ingested into
  `data/uploads/`
- `GET /documents` — list ingested documents with page/table/OCR stats
- `GET /docs` — interactive Swagger UI

Example:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How long is the free trial?"}' | python3 -m json.tool
```

## Tests

```bash
pytest tests/ -q
```

162 tests, no API keys required — every provider call (Gemini, Groq,
Pinecone) is mocked or exercised through a pure/offline code path. OCR
tests are skipped automatically if `tesseract` isn't installed. The
cross-encoder reranker is exercised through its documented fallback path
(BM25) rather than a real model download, so the test suite never needs
network access or the optional `sentence-transformers` dependency either.

## Project layout

```
app/
  core/               settings (single source of truth for env vars)
  db/                 SQLAlchemy models + session (document/ingestion metadata)
  documents/          structured document model, parsers, OCR, table reconstruction
  chunking.py         structure-aware chunking (headings, dedicated table chunks)
  clients.py          Gemini/Groq/Pinecone client factories
  vectorstore.py      Pinecone read/write
  retrieval.py        dense + hybrid (dense+BM25) retrieval
  reranker.py         lexical (BM25) reranking — the always-available fallback
  lexical_index.py    corpus-wide BM25 index (hybrid retrieval)
  rag/
    llm.py            grade/rewrite/answer prompts + parsing
    graph.py           the LangGraph pipeline (DocumentQAService)
    query_classifier.py  query type classification
    fusion.py          Reciprocal Rank Fusion
    cross_encoder.py   cross-encoder reranking, falls back to reranker.py
    bm25.py            shared BM25 scorer used by reranker.py + lexical_index.py
    grounding.py        claim-level grounding
  evaluation/         deterministic eval metrics — see evaluation/README.md
  services/           ingestion orchestration (CLI + upload endpoint share this)
  schemas/            Pydantic request/response models
  api/                FastAPI routers
  main.py             app assembly
data/
  corpus/        shipped sample corpus (all 6 supported formats)
  uploads/        runtime knowledge base (written by POST /documents/upload)
  lexical_index.json  BM25 snapshot — generated at ingestion time, not
                       committed to the repo (see .gitignore); run
                       `python -m app.services.ingestion_service` to create it
evaluation/        benchmark dataset + ablation runner — see evaluation/README.md
docs/            architecture.md, ocr.md, tables.md
tests/
MIGRATION_PLAN.md
```