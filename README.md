# VeriDoc AI

Document intelligence over scanned PDFs, tables, charts, and mixed-format archives — built around one constraint: every claim in an answer has to trace back to a retrieved chunk, or the system says so instead of guessing.

Most RAG demos stop at "retrieve, then generate." The part that's actually hard — and the part most portfolio projects skip — is deciding when *not* to answer, and proving that the answers you do give are grounded rather than merely plausible. That's what most of this codebase is actually about.

Link - https://veri-doc-ai-document-intelligence-a.vercel.app/ask

## Architecture

![System Architecture](System%20Architecture.svg)

Two independent flows share the same FastAPI entry point: document ingestion (top) builds a dense (Pinecone) and lexical (BM25) index; the query path (bottom) reads from that index through a LangGraph-orchestrated retrieval → reranking → generation → grounding sequence. Neither path is a straight line — retrieval loops on insufficient evidence, and generation is gated by a claim-level grounding check before anything reaches the user.

## Why this exists

Feed a document QA system a chart, a scanned receipt, and a badly OCR'd table, then ask it a question two of those sources partially answer. Most systems will confidently stitch together an answer anyway. This one is built to notice when it's doing that, and refuse or narrow the answer instead.

Concretely, that means:

- **Hybrid retrieval, not just vector search.** Dense embeddings and BM25 run independently and get combined with reciprocal rank fusion — dense search alone misses exact identifiers, part numbers, and error codes that lexical search catches, and BM25 alone misses paraphrased questions.
- **A reranking step that isn't decorative.** A cross-encoder rescoring pass sits between retrieval and generation, because "close in embedding space" and "actually answers the question" are not the same thing.
- **Claims are checked individually, not the whole answer at once.** An answer can be 80% correct and still cite the wrong page for the one number that matters. Grounding happens per-claim, and citation validation runs as a separate pass, not as a byproduct of generation.
- **Refusal is a first-class outcome.** Bounded query refinement gives retrieval a fixed number of retries before the system gives up and says so — it doesn't paper over insufficient evidence with a hedged-sounding answer.

## Measured results

Run against a 100-question benchmark spanning lookup, multi-hop, cross-document, table/figure, OCR, and deliberately unanswerable questions.

| Metric | Result |
|---|---|
| Recall@5 | 91.6% |
| Citation precision | 90.7% |
| Citation recall | 89.7% |
| Grounded answer rate | 89.8% |
| Claim grounding | 92.4% |
| Refusal accuracy | 93.0% |
| Unsupported claim rate | 16.0% → 7.9% (with grounding + citation validation enabled) |

The unsupported-claim number is the one I'd point to first: it's measured before/after the grounding gate on the same question set, not a comparison against a different system. The 90/10 split between citation precision and recall also matters more than either number alone — a system can hit high precision by citing sparsely, or high recall by over-citing; getting both close together means the citations are neither noisy nor incomplete.

Everything here comes from `evaluation/run_ablation.py`, which runs five pipeline configurations (dense-only baseline, +BM25 fusion, +reranking, +grounding, +grounding without citation validation) against the same question set so that a metric change is attributable to one specific component, not a confound. Retrieval-quality metrics (Recall/Precision/Hit-rate/nDCG@{1,3,5,10}, MRR) and a per-question-type modality breakdown are computed for every run.

One efficiency result worth calling out separately: an earlier version of the grounding pipeline made one LLM call per extracted claim to check it against evidence. Collapsing extraction and verification into a single structured call per answer cut total LLM calls by 31% on the same benchmark, with no measured drop in citation accuracy.

## How a query actually moves through the system

1. Query is classified (rule-based first; an LLM fallback exists for ambiguous cases) and, if the corpus can't possibly answer it, refused immediately — no retrieval call spent on it.
2. Dense and BM25 retrieval run in parallel; results are merged with reciprocal rank fusion, then reranked by a cross-encoder scoring the query against each candidate directly.
3. An evidence-sufficiency check decides whether to proceed or retry with a reformulated query — bounded to a fixed number of loops, not unbounded retries.
4. The answer is generated strictly from the evidence set that passed the sufficiency check.
5. The answer is split into individual factual claims; each is checked against the cited evidence and labeled supported, partially supported, or unsupported in one combined pass.
6. Citations are validated against what was actually retrieved — the model can't cite a chunk_id it wasn't shown. Unsupported content is stripped or the response is downgraded to a refusal before it reaches the API response.

## Stack

**Backend** — FastAPI, LangGraph, PostgreSQL (SQLAlchemy + Alembic), Pinecone, Groq + Gemini, Tesseract OCR / pdfplumber / python-docx for ingestion, sentence-transformers for cross-encoder reranking, a from-scratch BM25 implementation for lexical retrieval and fusion.

**Frontend** — Next.js, deployed independently on Vercel.

**Optional fine-tuning** — a QLoRA pipeline (PEFT, bitsandbytes, 4-bit NF4 quantization) that specializes only the answer-generation step, leaving retrieval, reranking, and grounding untouched, so a self-hosted small model can be compared against the hosted baseline under otherwise identical conditions. Sized to run on a 4GB consumer GPU.

**Infra** — Docker / Docker Compose for local development, Render for the backend, GitHub Actions CI running the real pytest suite against a Postgres service container (not SQLite-only — a real FK-ordering bug in an early migration only surfaced under Postgres's stricter constraint enforcement) plus a frontend lint/build job and a Docker image-build job.

**Testing** — 739 automated tests.

## Running it locally

```bash
git clone https://github.com/Ayush8092/VeriDoc-AI-Document-Intelligence-Agentic-RAG-Platform.git
cd VeriDoc-AI-Document-Intelligence-Agentic-RAG-Platform

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY, GROQ_API_KEY, JWT_SECRET_KEY
alembic upgrade head
uvicorn app.main:app --reload

# Frontend, separate shell
cd frontend
npm install
npm run dev
```

Or `docker compose up` from the repo root, which brings up Postgres, a one-shot migration service, the backend, and the frontend together.

Running the evaluation harness yourself:

```bash
cd backend
python -m evaluation.dataset_validator evaluation/datasets/phase4_100q.jsonl --expected-count 100
python -m evaluation.run_ablation --configs A B D E H --dataset evaluation/datasets/phase4_100q.jsonl
```

QLoRA training and the E_BASE/E_QLORA comparison are documented separately in `backend/training/qlora/README.md`, including the actual hardware constraints and what has and hasn't been run yet — I'd rather that file be explicit about which numbers are real than have this one imply results that don't exist.

## Known limitations

- The Render free-tier deployment runs on ephemeral storage — local SQLite, uploaded files, and the local vector/lexical index are wiped on every redeploy or cold start unless you configure a persistent disk or point it at managed Postgres and S3 instead.
- QLoRA mode requires a CUDA GPU; there's no CPU fallback for the 4-bit inference path.
- The evaluation harness's TTFT (time-to-first-token) metric only has a value when run against the streaming endpoint — the ablation harness uses the synchronous path, so it reports `NOT_APPLICABLE` rather than a fabricated number.

## Author

Ayush Kumar — [GitHub](https://github.com/Ayush8092) · [LinkedIn](https://linkedin.com/in/ayush8092) · ak1357kumar@gmail.com
