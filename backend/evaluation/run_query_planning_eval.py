"""Query-planning accuracy evaluation runner (Phase 7 completion pass,
item 2.2).

**Why this script exists on its own, separate from `run_ablation.py`/
`run_phase4_eval.py`**: `app.rag.query_classifier.classify_query_rule_based`
is a pure, deterministic, regex-based function — it needs no LLM call,
no embedding call, no vector store, no ingested corpus, and no API key
of any kind. It is the ONE piece of this project's evaluation story that
is directly executable in ANY environment, including a fully offline
sandbox with zero external credentials. Every other evaluation runner in
this project (`run_ablation.py`, `run_phase4_eval.py`, `run_ragas.py`)
is blocked on live `GEMINI_API_KEY`/`GROQ_API_KEY`/`PINECONE_API_KEY`
credentials and network access to those providers — this one is not,
and this script's whole purpose is to actually exploit that rather than
leave it sitting unused (see `complete_phase_7_prompt.md`, item 2.2).

**What this measures, precisely**: for every row in
`evaluation/datasets/phase4_v1.jsonl` that has a `query_type` field,
this calls `classify_query_rule_based(question)` (the rule-based path
ONLY — `classify_query`'s optional LLM-refinement path needs a live
`chat` client and model, which this script deliberately does not use,
so the reported number is scored against exactly what a
zero-credential environment can produce) and compares the result
against that row's `query_type` ground truth via
`app.evaluation.reasoning_metrics.query_planning_accuracy`.

**The label-drift caveat, stated up front, not discovered by surprise**:
`evaluation/datasets/SCHEMA.md` documents that `query_type` also carries
two labels — `FIGURE_QUERY` and `PROMPT_INJECTION` — that predate (and
have no corresponding member in) `app.rag.query_classifier.QueryType`.
`query_planning_accuracy`'s own docstring already establishes the
policy for this (score them anyway; they can never count as correct,
and that's real information about label-set drift, not a bug to hide) —
this script follows that exact policy rather than inventing a
different one, and ADDITIONALLY reports a second accuracy figure
computed only over rows whose ground truth IS a real `QueryType`
member, so a reader can see both "accuracy against the dataset as
labeled" and "accuracy against the classifier's actual output space"
side by side, without either number being hidden or the other silently
preferred.

Usage (from `backend/`, in the MAIN venv — no isolated venv, no API
keys, no network needed):
    python -m evaluation.run_query_planning_eval
    python -m evaluation.run_query_planning_eval --dataset evaluation/datasets/v1.jsonl
    python -m evaluation.run_query_planning_eval --out evaluation/reports/query_planning.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.reasoning_metrics import query_planning_accuracy
from app.rag.query_classifier import QueryType, classify_query_rule_based
from evaluation.run_ablation import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "phase4_v1.jsonl"
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"

_VALID_QUERY_TYPES = {member.value for member in QueryType}


def run(dataset_path: Path) -> dict:
    dataset = load_dataset(dataset_path)
    rows_with_ground_truth = [row for row in dataset if row.get("query_type")]
    skipped_no_ground_truth = len(dataset) - len(rows_with_ground_truth)

    per_question = []
    predicted_all: list[str] = []
    truth_all: list[str] = []
    predicted_in_taxonomy: list[str] = []
    truth_in_taxonomy: list[str] = []

    for row in rows_with_ground_truth:
        truth = row["query_type"]
        predicted = classify_query_rule_based(row["question"]).value
        per_question.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "predicted": predicted,
                "ground_truth": truth,
                "correct": predicted == truth,
                "ground_truth_in_classifier_taxonomy": truth in _VALID_QUERY_TYPES,
            }
        )
        predicted_all.append(predicted)
        truth_all.append(truth)
        if truth in _VALID_QUERY_TYPES:
            predicted_in_taxonomy.append(predicted)
            truth_in_taxonomy.append(truth)

    overall = query_planning_accuracy(predicted_all, truth_all)
    in_taxonomy = query_planning_accuracy(predicted_in_taxonomy, truth_in_taxonomy)

    out_of_taxonomy_labels = sorted({t for t in truth_all if t not in _VALID_QUERY_TYPES})

    return {
        "summary": {
            "dataset": str(dataset_path),
            "n_dataset_rows": len(dataset),
            "n_skipped_no_query_type_field": skipped_no_ground_truth,
            "n_scored_all_labels": overall["n"],
            "accuracy_all_labels": overall["accuracy"],
            "n_scored_in_classifier_taxonomy_only": in_taxonomy["n"],
            "accuracy_in_classifier_taxonomy_only": in_taxonomy["accuracy"],
            "out_of_taxonomy_ground_truth_labels": out_of_taxonomy_labels,
            "note": (
                "accuracy_all_labels scores every row with a query_type field, including "
                f"{out_of_taxonomy_labels} labels that predate and have no member in "
                "app.rag.query_classifier.QueryType (see evaluation/datasets/SCHEMA.md) — those "
                "rows can never be predicted correctly by design, per "
                "app.evaluation.reasoning_metrics.query_planning_accuracy's documented policy. "
                "accuracy_in_classifier_taxonomy_only excludes them, measuring the classifier "
                "purely against labels it can actually produce."
            ),
        },
        "confusion_all_labels": overall["confusion"],
        "confusion_in_classifier_taxonomy_only": in_taxonomy["confusion"],
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run(args.dataset)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (REPORTS_DIR / "query_planning_eval.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Query planning accuracy ===")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
