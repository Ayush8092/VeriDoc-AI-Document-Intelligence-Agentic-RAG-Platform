"""Generate real pipeline outputs for offline scoring — Phase 1 of a
two-phase flow with `evaluation/run_ragas.py` (Phase 2).

**Why this script exists as a separate step, not folded into
`run_ragas.py` directly (a real bug found and fixed in the Phase 7
completion pass):** `run_ragas.py` MUST run inside the isolated
`.venv-ragas` (see `requirements-ragas.txt`'s header for the full,
verified reason — `ragas` needs an old `langchain-core`, incompatible
with this project's `langgraph`). An earlier version of `run_ragas.py`
imported `app.rag.graph.DocumentQAService` directly to run the live
pipeline itself — but `DocumentQAService` imports `langgraph`
(`from langgraph.graph import END, StateGraph`), which is NOT installed
in `.venv-ragas` and CANNOT be, without reintroducing the exact
dependency conflict the isolated venv exists to avoid. Confirmed for
real: `from app.rag.graph import DocumentQAService` inside a clean
`.venv-ragas` install raises `ModuleNotFoundError: No module named
'langgraph'` immediately. This meant `run_ragas.py`, as written, could
never actually run in the one environment its own documentation said
to run it in — a real, previously-unverified architecture bug, not a
credentials/network limitation.

**The fix**: split "run the real pipeline" from "score with RAGAS" into
two scripts that never need to run in the same venv:

    1. THIS script (`generate_pipeline_outputs.py`) — runs in the MAIN
       venv (`requirements.txt`, which has `langgraph`). Executes
       `DocumentQAService.ask()` for real, for every benchmark question,
       optionally under one ablation config's overrides (`--config`,
       reusing `evaluation.run_ablation.CONFIGS` — the exact same
       overrides the deterministic ablation study uses, so a RAGAS score
       for config "D" is scored against the SAME pipeline configuration
       config "D" means everywhere else in this project). Writes a JSON
       file of {question, answer, retrieved context snippets, ...} —
       plain data, no RAGAS/langgraph-specific objects.
    2. `run_ragas.py` — runs in `.venv-ragas`. Reads that JSON file and
       scores it with RAGAS. Never imports `app.rag.graph` or anything
       else that touches `langgraph`.

Usage (from `backend/`, in the MAIN venv):
    python -m evaluation.generate_pipeline_outputs
    python -m evaluation.generate_pipeline_outputs --config D
    python -m evaluation.generate_pipeline_outputs --config D --limit 10
    python -m evaluation.generate_pipeline_outputs --dataset evaluation/datasets/v1.jsonl

Requires the same live credentials + ingested corpus as
`run_ablation.py` (see that script's own docstring) — this script makes
the exact same real LLM/embedding/vector-store calls `run_ablation.py`
does, it just also saves what it got back for RAGAS to score later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import Settings, get_settings
from app.rag.graph import DocumentQAService
from evaluation.run_ablation import CONFIGS, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "phase4_v1.jsonl"
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"


def _chunk_texts(citations: list[dict]) -> list[str]:
    """RAGAS needs retrieved-context TEXT. `app.rag.llm.validate_citations`
    (what populates `service.ask()`'s `citations`) only carries a 280-char
    `snippet` per chunk, not the full chunk text — a real, documented
    limitation carried over unchanged from the pre-split `run_ragas.py`
    (fixing THAT would mean widening `validate_citations`'s output shape,
    a separate, larger change outside this fix's scope — see that
    function's docstring). `context_precision`/`context_recall` scored
    from this output are therefore scored against a snippet of the
    retrieved context, not the full chunk.
    """
    texts = []
    for c in citations or []:
        text = c.get("snippet") if isinstance(c, dict) else None
        if text:
            texts.append(text)
    return texts


def generate(dataset_path: Path, config_letter: str | None, limit: int | None) -> dict:
    settings = get_settings()
    if config_letter:
        if config_letter not in CONFIGS:
            raise SystemExit(f"Unknown config {config_letter!r}. Valid: {sorted(CONFIGS)}")
        settings = settings.model_copy(update=CONFIGS[config_letter].overrides)

    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]
    print(f"Loaded {len(dataset)} benchmark questions from {dataset_path}")
    if config_letter:
        print(f"Applying config {config_letter} overrides: {CONFIGS[config_letter].overrides}")

    service = DocumentQAService(settings=settings)

    rows: list[dict] = []
    for row in dataset:
        outcome = service.ask(row["question"])
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "query_type": row.get("query_type"),
                "answerable": row.get("answerable"),
                "found": outcome["found"],
                "answer": outcome["answer"],
                "retrieved_contexts": _chunk_texts(outcome["citations"]),
                # Phase 7 dashboard completion pass: threaded through so
                # run_ragas.py can compute `answer_correctness` (needs a
                # ground-truth reference — see app/evaluation/
                # ragas_metrics.py::score_single's docstring) for the
                # rows that actually have one. `phase4_v1.jsonl` carries
                # `expected_answer` for a subset of its rows; this was
                # previously computed nowhere — dataset ground truth
                # existed but was never passed to RAGAS at all, so
                # `answer_correctness` could never be anything but None
                # even when a real reference answer was available.
                "reference_answer": row.get("expected_answer"),
            }
        )
        print(f"  {row['id']}: found={outcome['found']}")

    return {
        "config": config_letter,
        "config_overrides": CONFIGS[config_letter].overrides if config_letter else {},
        "dataset": str(dataset_path),
        "n_questions": len(dataset),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Ablation config letter to apply (reuses evaluation.run_ablation.CONFIGS). One of {sorted(CONFIGS)}. Omit for default settings.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (smoke test).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = generate(args.dataset, args.config, args.limit)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.config}" if args.config else ""
    out_path = args.out or (REPORTS_DIR / f"pipeline_outputs{suffix}.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out_path}")
    print(f"Next: .venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs {out_path}")


if __name__ == "__main__":
    main()