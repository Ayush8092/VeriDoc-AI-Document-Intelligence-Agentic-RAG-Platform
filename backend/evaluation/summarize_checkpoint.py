"""Computes real evaluation metrics from a PARTIAL/interrupted ablation
checkpoint — no API calls, nothing estimated or fabricated for the
questions that haven't completed yet.

Built for exactly this situation: `evaluation/run_ablation.py` got
interrupted (rate limit, multi-session resume) partway through a config,
leaving a checkpoint file with N completed per-question results out of
some larger target. This script reads ONLY those N completed results and
computes every metric `app.evaluation.metrics`/
`app.evaluation.retrieval_metrics` can compute from them — reusing those
exact modules, not reimplementing the math — then writes three files
that all say, explicitly and repeatedly, "based on N completed
questions," never silently presenting a partial run as if it were the
full benchmark.

Checkpoint file shape: a JSON list of per-question result dicts (the
same record shape `evaluation/run_ablation.py::run_config` builds — see
that file), OR a JSON object with a `"results"` key holding that list
(handles both since checkpoint-writing conventions vary run to run).
Each record is used as-is; any field a given record doesn't have (e.g.
`grounded_claim_rate` when claim grounding was disabled for that run) is
treated as absent by the metrics functions themselves — not
backfilled here.

Usage:
    python -m evaluation.summarize_checkpoint \\
        evaluation/reports/ablation_A.checkpoint.json \\
        --config A --target 300

    # or just point it at the file with everything else defaulted:
    python -m evaluation.summarize_checkpoint evaluation/reports/ablation_A.checkpoint.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.metrics import summarize
from app.evaluation.retrieval_metrics import evaluate_ranking

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"


def load_results(checkpoint_path: Path) -> list[dict]:
    """Load the completed-question result records from a checkpoint
    file, regardless of whether it's a bare list or wrapped in an
    object. Raises a clear error rather than silently returning []
    if the file doesn't look like either shape — a metrics report
    computed from an empty list by accident (e.g. a typo'd key name)
    is worse than a script that refuses to run."""
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        for key in ("results", "completed", "data", "items"):
            if isinstance(data.get(key), list):
                results = data[key]
                break
        else:
            raise ValueError(
                f"{checkpoint_path}: top-level object has no list-valued "
                f"'results'/'completed'/'data'/'items' key — found keys {sorted(data.keys())}. "
                "Pass --results-key to name the correct one, or check the checkpoint format."
            )
    else:
        raise ValueError(f"{checkpoint_path}: expected a JSON list or object, got {type(data).__name__}")

    if not results:
        raise ValueError(f"{checkpoint_path}: contains zero completed results — nothing to summarize.")
    return results


def latencies_from_results(results: list[dict]) -> list[float]:
    """Pulls per-question latency in seconds from whichever field name
    is actually present (different checkpoint-writing passes have used
    `latency_seconds` and `elapsed_seconds`) — returns [] rather than
    raising if neither is present, since latency percentiles are an
    optional part of the report, not a required one.
    """
    out = []
    for r in results:
        for key in ("latency_seconds", "elapsed_seconds", "duration_seconds"):
            if r.get(key) is not None:
                out.append(float(r[key]))
                break
    return out


def per_question_retrieval_metrics(results: list[dict]) -> list[dict]:
    """Recall@K/MRR/nDCG for each individual question that has both
    `retrieved_chunk_ids` and `required_chunk_ids` — used for the CSV's
    per-question breakdown. `app.evaluation.metrics.retrieval_metrics_mean`
    (called inside `summarize()`) already gives the aggregate; this is
    the same computation exposed per-row instead of averaged, for anyone
    who wants to spot-check which specific questions retrieval struggled
    on within this partial run.
    """
    rows = []
    for r in results:
        required = r.get("required_chunk_ids")
        retrieved = r.get("retrieved_chunk_ids")
        if not required or retrieved is None:
            continue
        row = evaluate_ranking(retrieved, required)
        row["id"] = r.get("id")
        rows.append(row)
    return rows


def build_report(results: list[dict], config: str, target_total: int | None, checkpoint_path: Path) -> dict:
    latencies = latencies_from_results(results)
    summary = summarize(results, latencies_seconds=latencies or None)

    n_completed = len(results)
    completion_note = (
        f"{n_completed} of {target_total} planned questions completed"
        if target_total
        else f"{n_completed} questions completed (target total not specified)"
    )

    return {
        "PARTIAL_RUN_NOTICE": (
            f"This report is based on {n_completed} COMPLETED questions only. "
            "It is not the full benchmark. No result for an incomplete question was "
            "estimated, interpolated, or fabricated — every number below is computed "
            "directly and only from the questions that actually finished."
        ),
        "config": config,
        "checkpoint_source": str(checkpoint_path),
        "n_completed": n_completed,
        "target_total": target_total,
        "completion_note": completion_note,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }


def write_json(report: dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(report: dict, results: list[dict], out_path: Path) -> None:
    """Two sections in one CSV, clearly delimited: the aggregate
    metric -> value table first (what you'd paste into a resume/slide),
    then a per-question retrieval-metrics breakdown below a blank row +
    header, for anyone who wants to audit individual questions. Both
    sections repeat the partial-run notice in a comment-style leading
    row, since a CSV opened outside this script's context can't show
    the JSON's notice field.
    """
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# PARTIAL RUN — {report['n_completed']} of "
                          f"{report['target_total'] or 'unspecified target'} questions completed. "
                          f"Nothing below is estimated or fabricated for incomplete questions."])
        writer.writerow([])
        writer.writerow(["metric", "value"])
        _write_flat_metrics(writer, report["summary"])

        per_q = per_question_retrieval_metrics(results)
        if per_q:
            writer.writerow([])
            writer.writerow([f"# per-question retrieval metrics ({len(per_q)} questions with ground truth)"])
            keys = [k for k in per_q[0].keys() if k != "id"]
            writer.writerow(["id"] + keys)
            for row in per_q:
                writer.writerow([row.get("id")] + [row.get(k) for k in keys])


def _write_flat_metrics(writer, summary: dict, prefix: str = "") -> None:
    """Flattens the (possibly nested, e.g. summary['retrieval']['recall@5'])
    summary dict into flat "metric,value" rows — None is written as the
    literal string "N/A", never blank (which could be misread as 0) and
    never a fabricated number.
    """
    for key, value in summary.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            _write_flat_metrics(writer, value, prefix=f"{label}.")
        else:
            writer.writerow([label, "N/A" if value is None else value])


def write_markdown(report: dict, out_path: Path) -> None:
    summary = report["summary"]
    lines = [
        f"# Ablation {report['config']} — Partial Results ({report['n_completed']} Questions)",
        "",
        f"> **{report['PARTIAL_RUN_NOTICE']}**",
        "",
        f"- Checkpoint source: `{report['checkpoint_source']}`",
        f"- Completed: **{report['n_completed']}**"
        + (f" of {report['target_total']} planned" if report["target_total"] else ""),
        f"- Generated: {report['generated_at']}",
        "",
        "## Application / grounding metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    app_metric_keys = [
        "n_questions", "grounded_answer_rate", "refusal_accuracy", "correct_answer_rate",
        "citation_precision", "citation_recall", "claim_grounding_rate", "unsupported_claim_rate",
        "hallucination_rate", "answerable_sensitivity", "unanswerable_specificity",
    ]
    for key in app_metric_keys:
        if key in summary:
            lines.append(f"| {key} | {_fmt(summary[key])} |")

    if summary.get("retrieval"):
        lines += ["", "## Retrieval metrics", "", "| Metric | Value |", "| --- | --- |"]
        for key, value in summary["retrieval"].items():
            lines.append(f"| {key} | {_fmt(value)} |")

    if summary.get("latency"):
        lines += ["", "## Latency (from the completed questions only)", "", "| Metric | Value |", "| --- | --- |"]
        for key, value in summary["latency"].items():
            lines.append(f"| {key} | {_fmt(value)} |")

    lines += [
        "",
        "## What's NOT in this report",
        "",
        f"- Any question beyond the {report['n_completed']} completed here — not run yet, not estimated.",
        "- RAGAS/TruLens scores, unless the checkpoint's records already carried them (this script only "
        "computes what `app.evaluation.metrics`/`retrieval_metrics` can derive from the fields present).",
        "- Anything requiring a live API call — this script made none.",
        "",
        "Re-run this script against an updated checkpoint (more completed questions) at any "
        "time to regenerate all three files with more data — nothing here needs to be hand-edited.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Path to the *.checkpoint.json file.")
    parser.add_argument("--config", default=None, help="Config letter (A, B, ...). Inferred from filename if omitted.")
    parser.add_argument("--target", type=int, default=None, help="Planned total question count, for the completion note.")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--out-prefix", default=None, help="Filename prefix; default: ablation_<config>_<N>q")
    args = parser.parse_args()

    results = load_results(args.checkpoint)
    n = len(results)

    config = args.config
    if config is None:
        stem = args.checkpoint.stem  # e.g. "ablation_A.checkpoint" -> "ablation_A"
        stem = stem.replace(".checkpoint", "")
        config = stem.split("_")[-1] if "_" in stem else stem

    report = build_report(results, config=config, target_total=args.target, checkpoint_path=args.checkpoint)

    prefix = args.out_prefix or f"ablation_{config}_{n}q"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{prefix}_metrics.json"
    csv_path = args.out_dir / f"{prefix}_metrics.csv"
    md_path = args.out_dir / f"{prefix}_summary.md"

    write_json(report, json_path)
    write_csv(report, results, csv_path)
    write_markdown(report, md_path)

    print(f"Completed questions: {n}" + (f" / {args.target} planned" if args.target else ""))
    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {md_path}")
    print("\nHeadline numbers (partial, based on", n, "questions):")
    for key in ("grounded_answer_rate", "refusal_accuracy", "citation_precision", "citation_recall"):
        if key in report["summary"]:
            print(f"  {key}: {_fmt(report['summary'][key])}")


if __name__ == "__main__":
    main()