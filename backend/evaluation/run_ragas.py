"""RAGAS evaluation runner (Phase 2 of a two-phase flow) — MUST be run
from the isolated `.venv-ragas` environment (see `requirements-ragas.txt`
for exactly why RAGAS can't share the main backend venv).

**Two-phase flow (Phase 7 completion pass — see
`evaluation/generate_pipeline_outputs.py`'s docstring for the full
story of the real bug this fixes):** this script does NOT run the
DocumentQAService pipeline itself and does NOT import `app.rag.graph`
or anything else that touches `langgraph` — that import would fail
inside `.venv-ragas` (`langgraph` is deliberately not installed there,
since it conflicts with the `langchain-core` version RAGAS needs; this
was confirmed by actually trying it and hitting `ModuleNotFoundError:
No module named 'langgraph'`). Instead:

    1. In the MAIN venv, run:
       `python -m evaluation.generate_pipeline_outputs [--config LETTER]`
       — this executes the REAL pipeline and writes a JSON file of
       {question, answer, retrieved_contexts, ...}.
    2. In `.venv-ragas`, run THIS script against that file:
       `.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs <path>`
       — this loads the saved outputs and scores them with RAGAS
       (Faithfulness, Answer Relevancy, Context Precision, Context
       Recall). No pipeline execution happens here; only scoring.

This mirrors `evaluation/run_ablation.py`'s "run the real pipeline,
never fabricate a number" posture and reports/ output convention, split
across the venv boundary that a single script couldn't cross.

Usage (from `backend/`):
    # Phase 1 — main venv:
    python -m evaluation.generate_pipeline_outputs --config D
    # Phase 2 — .venv-ragas:
    python3 -m venv .venv-ragas
    .venv-ragas/bin/pip install -r requirements-ragas.txt
    .venv-ragas/bin/pip install pydantic-settings "google-genai>=1.0,<2.0" groq pinecone numpy python-dotenv
    .venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_D.json
    .venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs evaluation/reports/pipeline_outputs_D.json --limit 10   # smoke test, cheaper

Requires (Phase 1, main venv): GEMINI_API_KEY, GROQ_API_KEY,
PINECONE_API_KEY (or VECTORSTORE_BACKEND=local) set (backend/.env), and
the corpus already ingested. Requires (Phase 2, .venv-ragas):
GROQ_API_KEY (the RAGAS LLM judge — see `app.evaluation.ragas_metrics`)
and GEMINI_API_KEY (the RAGAS embeddings model, via the same
`app.clients.get_embeddings` the main pipeline uses).

**Verification status (Phase 7 completion pass):** the import/
construction chain this script depends on (`ragas`, `langchain_groq`,
`app.evaluation.ragas_metrics._wrapped_llm`/`_wrapped_embeddings`) was
verified for real against a clean install of `requirements-ragas.txt` —
see that file's own header for exactly what was checked and how two
real bugs (including THIS script's now-fixed langgraph import) were
found and fixed along the way. What remains unverified (and could not
be verified without network access to the Groq/Gemini APIs) is an
actual scored run — this script's first real execution against live
credentials is also its first end-to-end verification. Every number
this script produces IS a genuine measurement at run time; nothing here
is a stand-in or placeholder.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from app.evaluation.ragas_metrics import RagasUnavailableError, score_single
from app.core.config import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"


def load_pipeline_outputs(path: Path, limit: int | None = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        data = {**data, "rows": data["rows"][:limit]}
    return data


def run(pipeline_outputs_path: Path, limit: int | None = None) -> dict:
    settings = get_settings()
    data = load_pipeline_outputs(pipeline_outputs_path, limit=limit)
    rows_in = data["rows"]
    print(
        f"Loaded {len(rows_in)} pipeline outputs from {pipeline_outputs_path} "
        f"(config={data.get('config') or 'default'}, dataset={data.get('dataset')})"
    )

    per_metric: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        # Phase 7 completion pass — real bug found and fixed: score_single()
        # ALWAYS includes "answer_correctness" in its returned dict (see
        # that function's own docstring: "the key is always present"),
        # non-None whenever a row carries reference_answer (154/161 rows
        # in phase4_v1.jsonl do — see generate_pipeline_outputs.py). This
        # key was missing here, so `per_metric[key].append(value)` below
        # would raise KeyError the first time any row's answer_correctness
        # actually scored a real value — i.e. on almost any real run,
        # not some rare edge case.
        "answer_correctness": [],
    }
    rows: list[dict] = []
    n_answered = 0
    n_scored = 0
    n_errors = 0

    for row in rows_in:
        record: dict = {
            "id": row["id"],
            "question": row["question"],
            "query_type": row.get("query_type"),
            "answerable": row.get("answerable"),
            "found": row["found"],
            "answer": row["answer"],
            "scores": None,
            "error": None,
        }

        if row["found"]:
            n_answered += 1
            try:
                scores = score_single(
                    question=row["question"],
                    answer=row["answer"],
                    retrieved_contexts=row.get("retrieved_contexts") or [],
                    model=settings.answer_model,
                    reference_answer=row.get("reference_answer"),
                )
                record["scores"] = scores
                n_scored += 1
                for key, value in scores.items():
                    if value is not None:
                        per_metric[key].append(value)
            except RagasUnavailableError as exc:
                # Fail loudly and stop the whole run — every subsequent
                # question would hit the exact same unavailability, so
                # continuing would just waste API calls on the ones
                # already run for nothing.
                raise SystemExit(f"RAGAS unavailable: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - one question's scoring failing shouldn't abort the run
                n_errors += 1
                record["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(record)
        print(f"  {row['id']}: found={row['found']} scores={record['scores']}")

    summary = {
        "pipeline_outputs": str(pipeline_outputs_path),
        "config": data.get("config"),
        "dataset": data.get("dataset"),
        "n_questions": len(rows_in),
        "n_answered": n_answered,
        "n_scored": n_scored,
        "n_scoring_errors": n_errors,
    }
    for key, values in per_metric.items():
        summary[f"mean_{key}"] = round(statistics.mean(values), 4) if values else None

    return {"summary": summary, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-outputs",
        type=Path,
        required=True,
        help="Path to a JSON file produced by `python -m evaluation.generate_pipeline_outputs` (run in the MAIN venv first).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N questions (smoke test).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run(args.pipeline_outputs, limit=args.limit)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config_suffix = f"_{report['summary']['config']}" if report["summary"].get("config") else ""
    out_path = args.out or (REPORTS_DIR / f"ragas{config_suffix}.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RAGAS summary ===")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()