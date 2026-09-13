# Phase 7 — Manual Evaluation Runbook

This is the exact, ordered set of commands to run the real Veridoc
benchmark on your own machine, with real credentials/network access.
**No live benchmark has been executed in the environment that prepared
this repository** — every command below has been checked against the
actual current code (module names, CLI flags, default paths), but the
numeric results are yours to generate, not already produced. See the
Phase 7 completion report for exactly what was and wasn't verifiable
without live credentials.

Run everything from `backend/` unless stated otherwise.

## 1. Create/activate the backend environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure `.env`

```bash
cp .env.example .env
```

Fill in `GEMINI_API_KEY` and `GROQ_API_KEY` (both have free tiers — see
`.env.example`'s "RUN VERIDOC FOR $0" banner). Everything else has a
working free/local default:
`VECTORSTORE_BACKEND=local`, `STORAGE_BACKEND=local`, `DATABASE_URL`
defaulting to local SQLite. Only fill in `PINECONE_API_KEY`/`S3_*`/
`DATABASE_URL`/`RATE_LIMIT_REDIS_URL` if you're deliberately opting into
one of those instead of the free/local defaults.

## 4. Start PostgreSQL (only if you opted into it in step 3)

Skip this step entirely if you're using the default local SQLite
database. If you set `DATABASE_URL=postgresql+...`:

```bash
# from the repo root, not backend/
docker compose --profile integration-test up -d postgres
```

(Or use your own existing PostgreSQL instance — any reachable
`DATABASE_URL` works.)

## 5. Apply Alembic migrations

```bash
alembic upgrade head
```

Verifies against: `alembic/versions/89dd0fc94357_baseline_schema.py`,
`f3a1c9d2e7b4_add_ingestion_jobs_table.py`,
`b811bd877381_conversation_memory_and_ingestion_job_backup_columns.py`.

## 6. Prepare the corpus

The sample corpus is already checked in at `data/corpus/` — nothing to
do here unless you're adding your own files (see `data/corpus/README.txt`).

## 7. Ingest the corpus

```bash
python -m app.services.ingestion_service          # idempotent — safe to re-run
python -m app.services.ingestion_service --reset   # wipe + re-ingest everything from scratch
```

This is also the point where `evaluation/datasets/phase4_v1.jsonl` rows
marked `"ground_truth_status": "pending_ingestion"` can finally be
backfilled — see step 8's note.

## 8. Verify the vector index

```bash
python3 -c "
from app.core.config import get_settings
from app.clients import get_pinecone, get_index
s = get_settings()
pc = get_pinecone(s)
idx = get_index(pc, s)
print(idx.describe_index_stats())
"
```

For the default local backend this reads `data/vector_index/` (or
whatever `LOCAL_VECTOR_INDEX_PATH` points to); for Pinecone it queries
the real hosted index. A non-zero `vector_count` confirms ingestion
actually wrote vectors.

**Backfilling `pending_ingestion` rows** (`evaluation/datasets/phase4_v1.jsonl`,
`evaluation/datasets/SCHEMA.md`): after a real ingestion run, look up the
real `object_id`/`bbox` for each `pending_ingestion` row's `expected_chunks`
by querying the index directly (same pattern as above, filtering
`metadata["chunk_id"]`), fill in `expected_objects`/`expected_citations`,
and change that row's `ground_truth_status` to `"verified_phase4"`. Do
NOT flip the status without actually doing this — see the schema doc's
explicit warning against treating `pending_ingestion` as verified.

## 9. Verify the lexical/BM25 index

```bash
python3 -c "
import json
from pathlib import Path
from app.core.config import get_settings
s = get_settings()
data = json.loads(Path(s.lexical_index_path).read_text())
print(f'{len(data)} chunks in the BM25 snapshot')
"
```

Generated automatically by step 7 (`app/lexical_index.py`'s
`save_snapshot`, called from `app/services/ingestion_service.py`) — no
separate build command.

## 10. Run the dataset validator

```bash
python -m evaluation.dataset_validator evaluation/datasets/phase4_v1.jsonl
```

Must report `"161 rows, all valid."` before proceeding.

`evaluation/datasets/multi_doc_v1.jsonl` (Config G) intentionally uses a
different, simpler schema (`id`/`topic`/`documents`/`expected_relations`
— see `run_ablation.py::load_multi_doc_dataset`'s docstring), not the
`phase4_v1.jsonl` schema `dataset_validator.py` checks — **do not** run
the validator against it; a "missing required fields" result from doing
so is a false positive from using the wrong tool on the wrong schema,
not a real problem with that file. `load_multi_doc_dataset` does its own
minimal JSON-shape parsing, which is sufficient for that file's simple
structure.

## 11. Run the A–H ablations

```bash
python -m evaluation.run_ablation --dataset evaluation/datasets/phase4_v1.jsonl
```

Runs every config (A, B, C, D, E, F, H) by default. To run a subset:

```bash
python -m evaluation.run_ablation --configs A B D E H
```

Writes `evaluation/reports/ablation_<letter>.json` per config plus a
combined `evaluation/reports/ablation_summary.json`.

## 12. Run the G multi-document evaluation

```bash
python -m evaluation.run_ablation --multi-doc --multi-doc-dataset evaluation/datasets/multi_doc_v1.jsonl
```

Writes `evaluation/reports/ablation_G.json`.

## 13. Run the Phase 4 (multimodal provenance) evaluation

```bash
python -m evaluation.run_phase4_eval --dataset evaluation/datasets/phase4_v1.jsonl
# or, to also run it across every ablation config:
python -m evaluation.run_phase4_eval --dataset evaluation/datasets/phase4_v1.jsonl --ablation
```

Writes `evaluation/reports/phase4_eval_<config>.json`,
`evaluation/reports/phase4_eval_summary.json`, and
`evaluation/reports/phase4_eval_report.md`.

## 14. Run RAGAS (two phases, across the venv boundary)

**RAGAS is a separate, second phase — it never runs the pipeline itself.**
`evaluation/run_ragas.py` deliberately does not import `app.rag.graph` (or
anything that touches `langgraph`), because `langgraph` cannot be
installed alongside RAGAS's own `langchain-core` requirement in the same
venv (a real, verified conflict — see `requirements-ragas.txt`'s header).
So scoring with RAGAS is always two steps, in two different Python
environments:

```
MAIN VENV                              .venv-ragas
──────────                             ───────────
generate_pipeline_outputs
    (runs the real pipeline)
       │
       ▼
pipeline_outputs_<LETTER>.json  ──────►  run_ragas --pipeline-outputs
                                              (scores the saved outputs)
                                                 │
                                                 ▼
                                          ragas_<LETTER>.json
```

Step 1 (`generate_pipeline_outputs`) needs `GEMINI_API_KEY`/`GROQ_API_KEY`/
a vector backend exactly like every other runner in this document. Step
2 (`run_ragas`) needs its own venv (`.venv-ragas`, set up below) and its
own `GROQ_API_KEY` (the RAGAS LLM judge) + `GEMINI_API_KEY` (the RAGAS
embeddings model).

### 14.1 Set up `.venv-ragas` (once)

```bash
cd backend   # if not already there
python3 -m venv .venv-ragas
.venv-ragas/bin/pip install -r requirements-ragas.txt
.venv-ragas/bin/pip install pydantic-settings "google-genai>=1.0,<2.0" groq pinecone numpy python-dotenv
```

(Windows: `.venv-ragas\Scripts\python` and `.venv-ragas\Scripts\pip`
in place of `.venv-ragas/bin/python` / `.venv-ragas/bin/pip` in every
command below.)

### 14.2 Run both phases for each of the 5 required configurations

The required final evaluation table (step 20 /
`evaluation/dashboard.py`'s `_render_final_spec_table`) needs RAGAS
scores for exactly these 5 configs
(`evaluation/ablation_config.py::FINAL_TABLE_ROW_CONFIGS`):

```
A = Vector
B = Vector + BM25
D = Vector + BM25 + Cross Encoder
H = Vector + BM25 + Cross Encoder + Grounding
E = Vector + BM25 + Cross Encoder + Grounding + Citation Validation
```

For EACH of A, B, D, H, E — main venv first, then `.venv-ragas`:

```bash
# --- Config A ---
python -m evaluation.generate_pipeline_outputs --config A
.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_A.json

# --- Config B ---
python -m evaluation.generate_pipeline_outputs --config B
.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_B.json

# --- Config D ---
python -m evaluation.generate_pipeline_outputs --config D
.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_D.json

# --- Config H ---
python -m evaluation.generate_pipeline_outputs --config H
.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_H.json

# --- Config E ---
python -m evaluation.generate_pipeline_outputs --config E
.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_E.json
```

`generate_pipeline_outputs --config <LETTER>` writes
`evaluation/reports/pipeline_outputs_<LETTER>.json` automatically (the
filename is derived from `--config`, not something you name yourself —
see that script's `main()`). `run_ragas --pipeline-outputs <path>`
likewise writes `evaluation/reports/ragas_<LETTER>.json` automatically,
reading the config letter back out of the pipeline-outputs file's own
`"config"` field — you never need to pass the output filename for
either step.

### 14.3 Cheap smoke test first

Before scoring a full 161-question run five times, add `--limit 10` to
BOTH phases to check the whole pipeline end-to-end cheaply (10 questions,
one config):

```bash
python -m evaluation.generate_pipeline_outputs --config A --limit 10
.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_A.json --limit 10
```

`run_ragas --limit` is independent of what `generate_pipeline_outputs
--limit` was — if the pipeline-outputs file has 10 rows and `run_ragas`
is also given `--limit 10`, it scores all 10; a smaller `run_ragas
--limit` than the file contains just scores a subset of an
already-generated file, useful for re-testing the RAGAS side alone
without re-running the pipeline.

### 14.4 Expected generated files, after running all 5 configs

```
evaluation/reports/
├── pipeline_outputs_A.json
├── pipeline_outputs_B.json
├── pipeline_outputs_D.json
├── pipeline_outputs_H.json
├── pipeline_outputs_E.json
├── ragas_A.json
├── ragas_B.json
├── ragas_D.json
├── ragas_H.json
└── ragas_E.json
```

Each `ragas_<LETTER>.json` holds per-question `scores` (`faithfulness`,
`answer_relevancy`, `context_precision`, `context_recall`,
`answer_correctness` when a reference answer was available — see
`evaluation/generate_pipeline_outputs.py`'s `reference_answer` field) and
a `summary` with the `mean_*` of each. `evaluation/dashboard.py` reads
these 5 files directly by this exact naming convention — do not rename
them.

## 15. Run TruLens (if configured)

Set `TRULENS_ENABLED=true` in `.env` (default), then TruLens scoring
runs automatically as part of `run_phase4_eval.py --trulens`:

```bash
python -m evaluation.run_phase4_eval --dataset evaluation/datasets/phase4_v1.jsonl --trulens
```

Uses its own local SQLite database (`TRULENS_DATABASE_URL`, default
`sqlite:///./data/trulens.sqlite`) — no external service required.

## 16. Enable Opik (optional)

Set `OPIK_ENABLED=true` in `.env`. Without `OPIK_API_KEY` set, traces go
to Opik's local/self-hosted mode rather than their cloud service — see
`app/observability/opik_integration.py`. Opik tracing then applies
automatically to every `service.ask()` call the runners above make
(`opik_traced_pipeline`) — no separate run step.

## 17. Generate the dashboard

```bash
python -m evaluation.dashboard --reports-dir evaluation/reports --out evaluation/reports/dashboard.md
```

Run this AFTER steps 11-15 have produced their report files — the
dashboard only renders what it finds in `evaluation/reports/`; it never
fabricates a row for a report that doesn't exist (see
`evaluation/dashboard.py`'s module docstring).

## 18. Inspect the generated reports

```
evaluation/reports/
├── ablation_A.json ... ablation_H.json
├── ablation_G.json
├── ablation_summary.json
├── phase4_eval_A.json ... (if --ablation was used)
├── phase4_eval_summary.json
├── phase4_eval_report.md
├── pipeline_outputs_A.json, pipeline_outputs_B.json, pipeline_outputs_D.json,
│   pipeline_outputs_H.json, pipeline_outputs_E.json   (from step 14.2, phase 1)
├── ragas_A.json, ragas_B.json, ragas_D.json,
│   ragas_H.json, ragas_E.json                          (from step 14.2, phase 2)
└── dashboard.md                (from step 17)
```

## 19. Run the final full test suite

```bash
pytest -q
pytest -q -rs   # same, with skip reasons shown (RAGAS/Redis/MinIO
                 # integration tests skip cleanly without their
                 # optional dependency/service — this is expected)
```

## 20. Collect final results

The single most useful artifact for reporting results is
`evaluation/reports/dashboard.md` (step 17) — it's designed to hold
exactly the "Configuration × Metric" table format used for reporting
Phase 7 results, populated from your real run's `evaluation/reports/*.json`
files, never hand-edited or pre-filled.
