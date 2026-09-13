"""Internal evaluation dashboard (Phase 6, spec item 13; Phase 7 spec
item 11).

NOT the end-user document dashboard (that's a frontend concept — spec
explicitly distinguishes the two: "This is an evaluation dashboard, not
the end-user document dashboard"). This is a developer-facing summary of
how the SYSTEM is performing, built by reading whatever
`evaluation/reports/*.json` files already exist on disk — it never runs
an evaluation itself (see `run_ablation.py`/`run_phase4_eval.py`/
`run_ragas.py` for that); it only aggregates and presents results that
already exist.

Deliberately a static Markdown report generator, not a live server: this
project already has a real backend server (`app/main.py`) whose job is
serving live traffic, and an evaluation dashboard's whole purpose is
being read by a developer/reviewer AFTER a benchmark run, not polled by
end users — a static file (checked into `evaluation/reports/dashboard.md`,
or opened directly) is the lower-complexity, equally-honest choice: no
new route, no new auth surface, no new thing that can silently drift
from what the underlying JSON reports actually say.

Phase 7 completion pass fixes (this file previously had real, confirmed
bugs — found by directly reading `evaluation/run_ablation.py`'s actual
report-writing code before touching this file, not assumed):
  - Config **H** was missing entirely from the ablation table loop —
    `for letter in ("A", "B", "C", "D", "E", "F")` never included it, so
    the one config whose whole purpose is a headline comparison (E vs H,
    "does citation validation actually help") never appeared.
  - The nDCG row read `(s.get("retrieval") or {}).get("ndcg")`, but
    `app.evaluation.retrieval_metrics.evaluate_ranking`'s actual output
    key is `f"ndcg@{max(k_values)}"` (e.g. `"ndcg@10"`) — this lookup
    could never have matched anything, so nDCG always rendered as "—"
    regardless of whether it was actually computed.
  - `reranker_integrity` (configured vs. actual reranker, fallback rate
    — the exact thing that lets a reader tell "Config D really used the
    cross-encoder" from "it silently fell back to BM25") was computed by
    every report but never surfaced here.
  - `run_metadata` (dataset path/version, question count, timestamp,
    embedding/answer model, vector backend — added to every report in
    the prior completion pass specifically for reproducibility) was
    likewise computed but never displayed.
  - RAGAS/TruLens results (written into `phase4_eval.json`'s summary by
    `run_phase4_eval.py`, when those optional dependencies are
    available) had no section here at all.
  - No token/cost section existed. Per audit: `DocumentQAService.ask()`
    (`app/rag/graph.py`) does not currently return any token/usage
    data — `app/observability/trace.py`'s `RequestTrace.record_llm_usage`
    exists and is used by the live API request path
    (`app/api/ask.py`), but the evaluation runner calls `service.ask()`
    directly, bypassing that instrumentation entirely. This is a
    genuine, confirmed gap, not something this pass fabricates a number
    for — this dashboard now reports token/cost as explicitly
    **UNAVAILABLE** with that exact reason, per the Phase 7 spec's own
    "never silently replace an unavailable metric with zero" rule.

Usage:
    python -m evaluation.dashboard
    python -m evaluation.dashboard --reports-dir evaluation/reports --out evaluation/reports/dashboard.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.ablation_config import CONFIGS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "dashboard.md"

# Every config the ablation runner can actually produce a report for —
# derived from evaluation/ablation_config.py's CONFIGS (the single
# source of truth for config identity — see that module's docstring),
# not a second hand-maintained letter list that could silently drift
# from it. `evaluation.ablation_config` is pure-data (a dataclass + two
# dict literals, zero heavy dependencies), so importing it here at
# module level does not reintroduce the pydantic/langgraph coupling this
# module's own docstring explains it was built to avoid. G (multi
# -document reasoning) is handled separately below since it's a
# different pipeline/dataset shape, not one more column in this table.
ABLATION_CONFIG_LETTERS = tuple(CONFIGS.keys())


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(value, digits: int = 4) -> str:
    """`None` renders as an em-dash — "not computed for this run", never
    a fabricated 0. Distinguished from the explicit "UNAVAILABLE" status
    string used for a metric family that isn't measured AT ALL (see
    module docstring) — `_fmt(None)` means "this run happened but this
    particular value wasn't produced", not "this can never be measured".
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_usage(usage: dict | None) -> str:
    """Renders `summary["usage"]` (see `evaluation/run_ablation.py`'s
    real token/cost capture via `start_trace()`/`record_llm_usage()`) as
    one compact cell: total tokens + estimated cost, or the honest
    UNAVAILABLE reason when nothing was captured for that run. Never a
    hard-coded string regardless of what the report actually contains —
    that was a real, confirmed bug this pass fixed (dashboard.py
    previously always rendered "UNAVAILABLE" here even after
    `run_ablation.py` started genuinely measuring this).
    """
    if not usage:
        return "—"
    if usage.get("status") != "MEASURED":
        return f"UNAVAILABLE ({usage.get('reason', 'not captured')})"
    total_tokens = usage.get("total_tokens")
    cost = usage.get("total_cost_usd")
    n_calls = usage.get("n_llm_calls")
    return f"{total_tokens:,} tok / ${cost:.4f} est. ({n_calls} calls)"


def _render_ablation_table(reports: dict[str, dict]) -> str:
    """`reports` maps config letter -> parsed `ablation_<letter>.json`."""
    if not reports:
        return "_No ablation reports found. Run `python -m evaluation.run_ablation` first._\n"

    metric_rows = [
        ("Refusal accuracy (blended)", lambda s: s.get("refusal_accuracy")),
        ("Answerable sensitivity", lambda s: s.get("answerable_sensitivity")),
        ("Unanswerable specificity", lambda s: s.get("unanswerable_specificity")),
        ("Grounded answer rate", lambda s: s.get("grounded_answer_rate")),
        ("Citation precision", lambda s: s.get("citation_precision")),
        ("Citation recall (mean)", lambda s: s.get("citation_recall")),
        ("Recall@1", lambda s: (s.get("retrieval") or {}).get("recall@1")),
        ("Recall@3", lambda s: (s.get("retrieval") or {}).get("recall@3")),
        ("Recall@5", lambda s: (s.get("retrieval") or {}).get("recall@5")),
        ("Recall@10", lambda s: (s.get("retrieval") or {}).get("recall@10")),
        ("Precision@1", lambda s: (s.get("retrieval") or {}).get("precision@1")),
        ("Precision@5", lambda s: (s.get("retrieval") or {}).get("precision@5")),
        ("Hit rate@1", lambda s: (s.get("retrieval") or {}).get("hit_rate@1")),
        ("Hit rate@5", lambda s: (s.get("retrieval") or {}).get("hit_rate@5")),
        ("MRR", lambda s: (s.get("retrieval") or {}).get("mrr")),
        ("nDCG@10", lambda s: (s.get("retrieval") or {}).get("ndcg@10")),
        ("Reranker (configured)", lambda s: (s.get("reranker_integrity") or {}).get("configured_reranker")),
        ("Reranker fallback rate", lambda s: (s.get("reranker_integrity") or {}).get("fallback_rate")),
        ("Mean latency (ms)", lambda s: (s.get("latency") or {}).get("mean_ms")),
        ("p50 latency (ms)", lambda s: (s.get("latency") or {}).get("p50_ms")),
        ("p95 latency (ms)", lambda s: (s.get("latency") or {}).get("p95_ms")),
        ("p99 latency (ms)", lambda s: (s.get("latency") or {}).get("p99_ms")),
        ("Token/cost", lambda s: _fmt_usage(s.get("usage"))),
    ]

    configs = [c for c in ABLATION_CONFIG_LETTERS if c in reports]
    lines = ["| Metric | " + " | ".join(f"Config {c}" for c in configs) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in configs) + " |")
    for label, getter in metric_rows:
        row = [label]
        for c in configs:
            summary = reports[c].get("summary", {})
            value = getter(summary)
            row.append(value if isinstance(value, str) else _fmt(value))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    any_usage_measured = any((reports[c].get("summary", {}).get("usage") or {}).get("status") == "MEASURED" for c in configs)
    any_usage_unavailable = any(
        (reports[c].get("summary", {}).get("usage") or {}).get("status") == "UNAVAILABLE" for c in configs
    )
    if any_usage_measured:
        lines.append(
            "> **Token/cost above is a real measurement** — `evaluation/run_ablation.py` wraps "
            "each `service.ask()` call with `app.observability.trace.start_trace()` and reads back "
            "`req_trace.to_dict()[\"llm_calls\"]`, the same per-call token/cost instrumentation the "
            "live `/ask` API path uses (`app/api/ask.py`). `total_cost_usd` is a configured PRICING "
            "ESTIMATE applied to real measured token counts (see `app/observability/cost.py`), not a "
            "provider-reported bill — treat it as order-of-magnitude, not exact. TTFT is a genuinely "
            "different, structural limitation: `mark_first_token()` only has meaning for the streaming "
            "endpoint (`app/api/ask_stream.py`), so it is never captured here regardless of usage "
            "status — see the TTFT columns above and `run_ablation.py`'s own comment on this.\n"
        )
    if any_usage_unavailable:
        lines.append(
            "> **Token/cost shows UNAVAILABLE for at least one config above** — no LLM response in "
            "that run exposed a `.usage` attribute (or every question was refused before any LLM call "
            "was made), so nothing was captured to report. Not a fabricated zero.\n"
        )

    run_metadata_lines = ["", "**Run metadata** (per config, from each report's own `run_metadata`):", ""]
    any_metadata = False
    for c in configs:
        meta = reports[c].get("run_metadata")
        if not meta:
            continue
        any_metadata = True
        run_metadata_lines.append(
            f"- **{c}**: dataset=`{meta.get('dataset')}`, n={meta.get('n_questions')}, "
            f"run_at={meta.get('run_at')}, embedding_model=`{meta.get('embedding_model')}`, "
            f"answer_model=`{meta.get('answer_model')}`, vectorstore_backend=`{meta.get('vectorstore_backend')}`"
        )
    if any_metadata:
        return "\n".join(lines) + "\n".join(run_metadata_lines) + "\n"
    return "\n".join(lines)


def _render_multi_doc_section(report: dict | None) -> str:
    if report is None:
        return "_No multi-document (Config G) report found. Run `python -m evaluation.run_ablation --multi-doc` first._\n"
    s = report.get("summary", {})
    lines = [
        f"- Rows evaluated: {s.get('n_rows')} ({s.get('n_errored')} errored)\n",
        f"- Mean citation coverage: {_fmt(s.get('mean_citation_coverage'))}\n",
        f"- Mean relation accuracy: {_fmt(s.get('mean_relation_accuracy'))}\n",
    ]
    meta = report.get("run_metadata")
    if meta:
        lines.append(
            f"- Run metadata: dataset=`{meta.get('dataset')}`, n={meta.get('n_questions')}, "
            f"run_at={meta.get('run_at')}\n"
        )
    return "".join(lines)


def _render_ragas_trulens_section(phase4_report: dict | None) -> str:
    """RAGAS/TruLens results, when present, come from `phase4_eval.json`'s
    summary (`run_phase4_eval.py` writes `summary["ragas"]`/`summary["trulens"]`
    only when those optional dependencies were actually importable and a
    live LLM judge call succeeded — see that script and
    `app.evaluation.ragas_metrics`/`app.evaluation.trulens_feedback` for
    the exact optional-dependency/typed-error handling). Absence here
    means "not run" (isolated RAGAS venv not set up, or TruLens
    dependency missing), reported as such rather than blank.
    """
    if phase4_report is None:
        return (
            "_No `phase4_eval.json` report found — RAGAS/TruLens are only scored as part of "
            "that run. Run `python -m evaluation.run_phase4_eval` first (requires the isolated "
            "`.venv-ragas` environment for RAGAS specifically — see `requirements-ragas.txt`)._\n"
        )
    s = phase4_report.get("summary", {})
    lines = []
    ragas = s.get("ragas")
    if ragas:
        lines.append("**RAGAS** (LLM-judge metrics, isolated `.venv-ragas` environment):\n")
        for key in ("mean_faithfulness", "mean_answer_relevancy", "mean_context_precision", "mean_context_recall", "mean_answer_correctness"):
            if key in ragas:
                lines.append(f"- {key.replace('mean_', '').replace('_', ' ').title()}: {_fmt(ragas.get(key))}\n")
    else:
        lines.append("_RAGAS: UNAVAILABLE — not run for this evaluation (see `evaluation/README.md`)._\n")

    trulens = s.get("trulens")
    if trulens:
        lines.append("\n**TruLens** (LLM-judge feedback functions):\n")
        for key in ("mean_groundedness", "mean_context_relevance", "mean_answer_relevance"):
            if key in trulens:
                lines.append(f"- {key.replace('mean_', '').replace('_', ' ').title()}: {_fmt(trulens.get(key))}\n")
    else:
        lines.append("\n_TruLens: UNAVAILABLE — not run for this evaluation (see `evaluation/README.md`)._\n")

    return "".join(lines)


def _render_modality_section(phase4_report: dict | None) -> str:
    if phase4_report is None:
        return "_No `phase4_eval.json` report found — no modality breakdown to show._\n"
    breakdown = phase4_report.get("summary", {}).get("modality_breakdown")
    if not breakdown:
        return "_No modality breakdown present in `phase4_eval.json`._\n"
    lines = ["| Modality | n | Refusal accuracy | Grounded answer rate |", "| --- | --- | --- | --- |"]
    for modality, bucket in sorted(breakdown.items()):
        lines.append(
            f"| {modality} | {bucket.get('n', 0)} | {_fmt(bucket.get('refusal_accuracy'))} | "
            f"{_fmt(bucket.get('grounded_answer_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def _render_final_spec_table(reports_dir: Path, ablation_reports: dict[str, dict]) -> str:
    """The exact table the Phase 7 spec asks for: 5 rows (Vector /
    Vector+BM25 / +CrossEncoder / +Grounding / +CitationValidation, via
    `evaluation.ablation_config.FINAL_TABLE_ROW_CONFIGS`), with every
    requested metric column — merging THREE report families per row's
    underlying config letter:
      - `ablation_<letter>.json` — Recall@K/Precision@K/Hit Rate@K/MRR/
        nDCG, Citation Precision, Grounded/Claim-Grounding/Hallucination
        rates, Refusal Accuracy + its Answerable-Sensitivity/
        Unanswerable-Specificity split, E2E latency percentiles.
      - `ragas_<letter>.json` — Context Precision, Context Recall,
        Faithfulness, Answer Relevancy, Answer Correctness (only
        non-"—" for rows where `evaluation/datasets/phase4_v1.jsonl`'s
        `expected_answer` ground truth exists AND was threaded through
        by `generate_pipeline_outputs.py` — see that file and
        `run_ragas.py`'s `reference_answer` wiring).
      - `ablation_G.json` — Multi-document reasoning accuracy. This is
        the ONE column that is NOT per-row-config: Config G's dataset
        (`multi_doc_v1.jsonl`) and pipeline
        (`app.rag.comparison.compare_documents`) don't vary along the
        same A/B/C/D/E retrieval-configuration axis the other columns
        do — there is no "multi-document reasoning accuracy under
        vector-only retrieval" concept to report. The SAME G-report
        number is therefore shown in every row, with an explicit
        footnote saying so — an honest single global measurement
        repeated for reading convenience, never five independently-
        varying numbers presented as if they were.
    A cell is "—" when its report file doesn't exist yet — NEVER a
    fabricated/zero value. TTFT columns are always "N/A (non-streaming
    runner)" — see `run_ablation.py`'s own documented, permanent
    architectural limitation (the ablation runner calls
    `DocumentQAService.ask()`, not the streaming endpoint TTFT is only
    meaningful for); not a missing-report gap, a structural one.
    """
    from evaluation.ablation_config import FINAL_TABLE_ROW_CONFIGS

    ragas_reports: dict[str, dict] = {}
    for letter in set(FINAL_TABLE_ROW_CONFIGS.values()):
        path = reports_dir / f"ragas_{letter}.json"
        if path.is_file():
            data = _load_json(path)
            if data is not None:
                ragas_reports[letter] = data

    multi_doc_path = reports_dir / "ablation_G.json"
    multi_doc_report = _load_json(multi_doc_path) if multi_doc_path.is_file() else None
    multi_doc_accuracy = (
        (multi_doc_report or {}).get("summary", {}).get("mean_relation_accuracy") if multi_doc_report else None
    )

    columns = [
        ("Context Precision", lambda a, r: r.get("mean_context_precision")),
        ("Context Recall", lambda a, r: r.get("mean_context_recall")),
        ("Faithfulness", lambda a, r: r.get("mean_faithfulness")),
        ("Answer Relevancy", lambda a, r: r.get("mean_answer_relevancy")),
        ("Answer Correctness", lambda a, r: r.get("mean_answer_correctness")),
        ("Citation Precision", lambda a, r: a.get("citation_precision")),
        ("Grounded Answer Rate", lambda a, r: a.get("grounded_answer_rate")),
        ("Claim Grounding Rate", lambda a, r: a.get("claim_grounding_rate")),
        ("Hallucination Rate", lambda a, r: a.get("hallucination_rate")),
        ("Refusal Accuracy", lambda a, r: a.get("refusal_accuracy")),
        ("Answerable Sensitivity", lambda a, r: a.get("answerable_sensitivity")),
        ("Unanswerable Specificity", lambda a, r: a.get("unanswerable_specificity")),
        ("Precision@10", lambda a, r: (a.get("retrieval") or {}).get("precision@10")),
        ("Hit Rate@10", lambda a, r: (a.get("retrieval") or {}).get("hit_rate@10")),
        ("Recall@10", lambda a, r: (a.get("retrieval") or {}).get("recall@10")),
        ("MRR", lambda a, r: (a.get("retrieval") or {}).get("mrr")),
        ("nDCG@10", lambda a, r: (a.get("retrieval") or {}).get("ndcg@10")),
        ("Multi-doc Reasoning Accuracy*", lambda a, r: multi_doc_accuracy),
        ("Median TTFT", lambda a, r: "N/A (non-streaming runner)"),
        ("P95 TTFT", lambda a, r: "N/A (non-streaming runner)"),
        ("Median E2E", lambda a, r: (a.get("latency") or {}).get("p50_ms")),
        ("P95 E2E", lambda a, r: (a.get("latency") or {}).get("p95_ms")),
    ]

    lines = ["| Configuration | " + " | ".join(name for name, _ in columns) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
    any_row_present = False
    for row_label, letter in FINAL_TABLE_ROW_CONFIGS.items():
        ablation_summary = (ablation_reports.get(letter) or {}).get("summary", {})
        ragas_summary = ragas_reports.get(letter, {}).get("summary", {})
        row = [row_label]
        for name, getter in columns:
            # Multi-doc reasoning accuracy and the TTFT columns don't
            # depend on THIS row's ablation/ragas report at all (see
            # this function's docstring) — they must not be blanked out
            # just because a per-config report hasn't been run yet.
            row_independent = name.startswith("Multi-doc") or "N/A" in name.upper()
            if not ablation_summary and not ragas_summary and not row_independent:
                row.append("—")
                continue
            value = getter(ablation_summary, ragas_summary)
            row.append(value if isinstance(value, str) else _fmt(value))
        if ablation_summary or ragas_summary or multi_doc_accuracy is not None:
            any_row_present = True
        lines.append("| " + " | ".join(row) + " |")

    if not any_row_present:
        return (
            "_No data for any row yet — run `python -m evaluation.run_ablation` and the "
            "`generate_pipeline_outputs.py` -> `run_ragas.py` pair for configs "
            f"{sorted(set(FINAL_TABLE_ROW_CONFIGS.values()))} first._\n"
        )

    missing_ragas = sorted(set(FINAL_TABLE_ROW_CONFIGS.values()) - set(ragas_reports))
    footer = "\n".join(lines) + "\n"
    if missing_ragas:
        footer += (
            f"\n> RAGAS columns (Context Precision/Recall, Faithfulness, Answer Relevancy) show "
            f"\"—\" for configs {missing_ragas} — no `ragas_<letter>.json` report found for them yet. "
            "Run `python -m evaluation.generate_pipeline_outputs --config <letter>` (main venv) then "
            "`.venv-ragas/bin/python -m evaluation.run_ragas --pipeline-outputs ...` (isolated venv) "
            "to produce them.\n"
        )
    return footer


def build_dashboard(reports_dir: Path) -> str:
    ablation_reports: dict[str, dict] = {}
    for letter in ABLATION_CONFIG_LETTERS:
        path = reports_dir / f"ablation_{letter}.json"
        if path.is_file():
            data = _load_json(path)
            if data is not None:
                ablation_reports[letter] = data

    multi_doc_path = reports_dir / "ablation_G.json"
    multi_doc_report = _load_json(multi_doc_path) if multi_doc_path.is_file() else None

    phase4_path = reports_dir / "phase4_eval.json"
    phase4_report = _load_json(phase4_path) if phase4_path.is_file() else None

    missing_configs = [c for c in ABLATION_CONFIG_LETTERS if c not in ablation_reports]

    lines = [
        "# Veridoc Evaluation Dashboard",
        "",
        "_Generated by `evaluation/dashboard.py` from whatever `evaluation/reports/*.json` "
        "files exist on disk at generation time — every number below is a real measurement "
        "from an actual run, or this dashboard says so explicitly instead of showing a blank/"
        "fabricated value._",
        "",
    ]
    if missing_configs:
        lines.append(
            f"> **Not yet run: {', '.join(missing_configs)}.** Missing reports are simply "
            "omitted from the table below, not shown as zero/blank columns.\n"
        )
    lines += [
        "## Required final evaluation table (Phase 7 spec's exact 5-row format)",
        "",
        _render_final_spec_table(reports_dir, ablation_reports),
        "## Retrieval ablation (Configs A–H)",
        "",
        _render_ablation_table(ablation_reports),
        "## Multi-document reasoning (Config G)",
        "",
        _render_multi_doc_section(multi_doc_report),
        "## Phase 4 multimodal evaluation",
        "",
    ]

    if phase4_report is None:
        lines.append("_No `phase4_eval.json` report found. Run `python -m evaluation.run_phase4_eval` first._\n")
    else:
        s = phase4_report.get("summary", {})
        lines.append(f"- Benchmark size: {s.get('n_questions', '—')}\n")
        prov = s.get("phase4_provenance") or {}
        for key, value in sorted(prov.items()):
            lines.append(f"- {key.replace('_', ' ').title()}: {_fmt(value)}\n")
        for key in ("table_extraction_accuracy", "ocr_cer", "ocr_wer"):
            if key in s:
                lines.append(f"- {key.replace('_', ' ').title()}: {_fmt(s.get(key))}\n")

    lines += [
        "",
        "## Modality breakdown",
        "",
        _render_modality_section(phase4_report),
        "",
        "## RAGAS / TruLens",
        "",
        _render_ragas_trulens_section(phase4_report),
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    content = build_dashboard(args.reports_dir)
    args.out.write_text(content, encoding="utf-8")
    print(f"Dashboard written to {args.out}")


if __name__ == "__main__":
    main()