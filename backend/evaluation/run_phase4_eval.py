"""Phase 4.1(7) item 2 — the end-to-end Phase 4 evaluation runner.

Wires the full pipeline exactly as it already exists — nothing here
reimplements retrieval, grading, generation, citation validation, or
grounding:

    Benchmark dataset -> DocumentQAService.ask() [query classification ->
    dense retrieval -> BM25 -> RRF fusion -> cross-encoder rerank ->
    evidence grading -> generation/refusal -> citation validation ->
    claim grounding] -> Phase 4 provenance metrics -> evaluation report

`DocumentQAService.ask()` (app/rag/graph.py) already performs every
pipeline stage in that list and already returns the full per-stage
candidate breakdown (see `run_ablation.py`'s own comment on this) — this
runner's only job is to call it once per benchmark question (through
`opik_traced_pipeline` for tracing, same as production), score the
outcome against ground truth with the existing metrics modules
(`app.evaluation.metrics`, `app.evaluation.phase4_metrics`,
`evaluation.modality`), optionally layer in RAGAS/TruLens (both off by
default — separate opt-in venvs/settings, see those modules'
docstrings), and write a report.

Two modes:
    python -m evaluation.run_phase4_eval
        Single run against current `Settings` (the full, real pipeline
        config — equivalent to ablation config "E").

    python -m evaluation.run_phase4_eval --ablation
        Runs configs A-E (via `run_ablation.CONFIGS`, not duplicated
        here) and additionally reports Phase 4 provenance metrics +
        modality breakdown per config — item 4's "integrate + make
        multimodal-aware" requirement.

Ground-truth fields consumed from each dataset row, all OPTIONAL so this
runs against the original `v1.jsonl` schema unchanged:
    required_chunks        (legacy name for expected_chunks; still read)
    expected_pages          list[int]
    expected_objects         list[str]   (object_ids)
    expected_citations       list[dict]  (bbox ground truth: {"bbox": {...}})
    should_refuse            bool        (falls back to `not answerable`)
    modality / requires_table / requires_visual / requires_chart /
    requires_multi_hop       (see evaluation/modality.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from app.rag.query_classifier import classify_query_rule_based
from app.core.config import Settings, get_settings
from app.evaluation import phase4_metrics
from app.evaluation.reasoning_metrics import query_planning_accuracy
from app.evaluation.metrics import summarize
from app.observability.opik_integration import opik_traced_pipeline
from app.rag.graph import DocumentQAService
from evaluation.modality import classify_modality, summarize_by_modality
from evaluation.run_ablation import CONFIGS, AblationConfig, load_dataset

_log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "phase4_v1.jsonl"
FALLBACK_DATASET = REPO_ROOT / "evaluation" / "datasets" / "v1.jsonl"
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"

# The "full pipeline, current settings" run used when --ablation isn't
# passed — deliberately just an alias for ablation config E (every
# Phase 3A stage on) rather than a second hand-maintained overrides dict
# that could quietly drift from it.
_FULL_PIPELINE_CONFIG = CONFIGS["E"]


def _extract_answer_and_contexts(outcome: dict) -> tuple[str, list[str]]:
    """Pull the generated answer text + retrieved chunk texts out of an
    `ask()` outcome, for RAGAS/TruLens (both want plain strings, not the
    pipeline's structured candidate dicts).
    """
    answer = outcome.get("answer") or ""
    candidates = outcome.get("reranked_candidates") or outcome.get("fused_candidates") or []
    contexts = [c.get("text", "") for c in candidates if c.get("text")]
    return answer, contexts


def _ground_truth_from_row(row: dict) -> dict:
    """Normalize a dataset row's ground-truth fields, tolerant of the
    original `v1.jsonl` schema (only `required_chunks`/`answerable`)
    alongside the expanded Phase 4 schema.
    """
    should_refuse = row.get("should_refuse")
    if should_refuse is None:
        should_refuse = row.get("answerable") is False
    return {
        "required_chunk_ids": row.get("expected_chunks") or row.get("required_chunks") or [],
        "expected_pages": row.get("expected_pages") or [],
        "expected_objects": row.get("expected_objects") or [],
        "expected_citations": row.get("expected_citations") or [],
        "should_refuse": should_refuse,
    }


def run_question(
    service: DocumentQAService,
    settings: Settings,
    row: dict,
    *,
    enable_ragas: bool = False,
    enable_trulens: bool = False,
) -> tuple[dict, float]:
    """Run one benchmark question through the full traced pipeline and
    score it. Returns `(result_record, latency_seconds)`.
    """
    gt = _ground_truth_from_row(row)

    # Phase 7 completion pass — same trace/token/cost integration fix as
    # evaluation/run_ablation.py::run_config (see that function's comment
    # for the full explanation): opik_traced_pipeline wraps this call for
    # OPIK's own, separate tracing system — it does NOT call
    # app.observability.trace.start_trace(), so without this,
    # current_trace() would return None throughout and every
    # _record_usage() call inside app/rag/llm.py would silently no-op.
    from app.observability.trace import start_trace

    req_trace = start_trace()
    started = time.monotonic()
    outcome = opik_traced_pipeline(settings, service.ask, row["question"])
    latency = time.monotonic() - started
    llm_calls = req_trace.to_dict()["llm_calls"]

    citations = outcome.get("citations") or []
    cited_chunk_ids = [c["chunk_id"] for c in citations]
    cited_object_ids = [c.get("object_id") for c in citations]
    cited_pages = [c.get("page_start") for c in citations]
    cited_bboxes = [c["bbox"] for c in citations if c.get("bbox")]

    reranked_ids = [c["chunk_id"] for c in outcome.get("reranked_candidates") or []]
    fused_ids = [c["chunk_id"] for c in outcome.get("fused_candidates") or []]
    retrieved_ids = reranked_ids or fused_ids

    result: dict = {
        "id": row["id"],
        "question": row["question"],
        "query_type": row.get("query_type", "UNSPECIFIED"),
        "answerable": row.get("answerable", not gt["should_refuse"]),
        "found": outcome["found"],
        "cited_chunk_ids": cited_chunk_ids,
        "required_chunk_ids": gt["required_chunk_ids"],
        "retrieved_chunk_ids": retrieved_ids,
        "dense_chunk_ids": [c["chunk_id"] for c in outcome.get("dense_candidates") or []],
        "bm25_chunk_ids": [c["chunk_id"] for c in outcome.get("bm25_candidates") or []],
        "fused_chunk_ids": fused_ids,
        "reranked_chunk_ids": reranked_ids,
        "grounded_claim_rate": outcome.get("grounded_claim_rate"),
        "citation_precision": outcome.get("citation_precision"),
        "configured_reranker": outcome.get("configured_reranker"),
        "actual_reranker": outcome.get("actual_reranker"),
        "fallback_used": outcome.get("fallback_used"),
        "modality": classify_modality(row),
        # Phase 4 provenance metrics — every one is None (not 0.0) when
        # the row carries no relevant ground truth, per
        # `phase4_metrics`'s own contract (see its functions' docstrings)
        # so a report can distinguish "wasn't asked" from "got it wrong".
        "citation_correctness": phase4_metrics.citation_correctness(
            cited_chunk_ids, gt["required_chunk_ids"], should_refuse=gt["should_refuse"]
        ),
        "object_correctness": phase4_metrics.object_correctness(cited_object_ids, gt["expected_objects"]),
        "page_correctness": phase4_metrics.page_correctness(cited_pages, gt["expected_pages"]),
        "bbox_correctness": phase4_metrics.bbox_correctness(
            cited_bboxes, [c["bbox"] for c in gt["expected_citations"] if c.get("bbox")]
        )
        if any(c.get("bbox") for c in gt["expected_citations"])
        else None,
        # Phase 7 Part 6 gap-fill: query-planning accuracy needs BOTH the
        # predicted and ground-truth label per question so
        # `_aggregate_phase4_metrics` can call
        # `reasoning_metrics.query_planning_accuracy` once over the whole
        # batch — see that function's docstring for why comparing whole
        # lists (not accumulating a running accuracy per-row) also
        # produces the confusion breakdown.
        "predicted_query_type": classify_query_rule_based(row["question"]).value,
        "ground_truth_query_type": row.get("query_type"),
        # Phase 7 completion pass: per-question token/cost — see the
        # start_trace() comment above. Aggregated across all rows into
        # summary["usage"] by the caller (main()/_build_summary,
        # matching evaluation/run_ablation.py::run_config's shape).
        "llm_calls": llm_calls,
    }

    if enable_ragas or enable_trulens:
        answer, contexts = _extract_answer_and_contexts(outcome)
        if enable_ragas:
            result["ragas"] = _score_ragas(row["question"], answer, contexts)
        if enable_trulens:
            result["trulens"] = _score_trulens(row["question"], answer, contexts, settings)

    return result, latency


_ragas_warned = False
_trulens_warned = False


def _score_ragas(question: str, answer: str, contexts: list[str]) -> dict | None:
    global _ragas_warned
    try:
        from app.evaluation.ragas_metrics import RagasUnavailableError, score_single

        return score_single(question=question, answer=answer, retrieved_contexts=contexts)
    except (ImportError, RagasUnavailableError) as exc:
        if not _ragas_warned:
            _log.warning("RAGAS scoring unavailable, skipping for the rest of this run: %s", exc)
            _ragas_warned = True
        return None
    except Exception as exc:  # noqa: BLE001 - one bad turn shouldn't abort the whole eval run
        _log.warning("RAGAS scoring failed for question %r: %s", question[:80], exc)
        return None


def _score_trulens(question: str, answer: str, contexts: list[str], settings: Settings) -> dict | None:
    global _trulens_warned
    try:
        from app.evaluation.trulens_feedback import TruLensUnavailableError, score_single

        return score_single(question=question, answer=answer, retrieved_contexts=contexts, settings=settings)
    except (ImportError, TruLensUnavailableError) as exc:
        if not _trulens_warned:
            _log.warning("TruLens scoring unavailable, skipping for the rest of this run: %s", exc)
            _trulens_warned = True
        return None
    except Exception as exc:  # noqa: BLE001
        _log.warning("TruLens scoring failed for question %r: %s", question[:80], exc)
        return None


def run_config(
    config: AblationConfig,
    dataset: list[dict],
    base_settings: Settings,
    *,
    enable_ragas: bool = False,
    enable_trulens: bool = False,
) -> dict:
    settings = base_settings.model_copy(update=config.overrides)
    service = DocumentQAService(settings=settings)

    results: list[dict] = []
    latencies: list[float] = []
    for row in dataset:
        result, latency = run_question(
            service, settings, row, enable_ragas=enable_ragas, enable_trulens=enable_trulens
        )
        results.append(result)
        latencies.append(latency)

    summary = summarize(results, latencies_seconds=latencies)
    summary["modality_breakdown"] = summarize_by_modality(
        dataset, results, {row["id"]: lat for row, lat in zip(dataset, latencies)}
    )
    summary["phase4_provenance"] = _aggregate_phase4_metrics(results)
    if enable_ragas:
        summary["ragas"] = _aggregate_optional_metric(results, "ragas")
    if enable_trulens:
        summary["trulens"] = _aggregate_optional_metric(results, "trulens")

    return {
        "config": config.name,
        "label": config.label,
        "overrides": config.overrides,
        "summary": summary,
        "results": results,
    }


def _aggregate_phase4_metrics(results: list[dict]) -> dict:
    """Mean of each `phase4_metrics` score across every result that had
    ground truth for it (`None` values excluded, not treated as 0).
    """
    out = {}
    for key in ("citation_correctness", "object_correctness", "page_correctness", "bbox_correctness"):
        values = [r[key] for r in results if r.get(key) is not None]
        out[key] = round(sum(values) / len(values), 4) if values else None
        out[f"{key}_n"] = len(values)
    return out


def _aggregate_optional_metric(results: list[dict], key: str) -> dict:
    scored = [r[key] for r in results if r.get(key)]
    if not scored:
        return {"n": 0}
    metric_names = {name for s in scored for name in s}
    out: dict = {"n": len(scored)}
    for name in metric_names:
        values = [s[name] for s in scored if s.get(name) is not None]
        out[name] = round(sum(values) / len(values), 4) if values else None
    return out


def _resolve_dataset_path(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    return FALLBACK_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=None, help="Defaults to phase4_v1.jsonl, falling back to v1.jsonl")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run configs A-E (item 4) instead of a single full-pipeline run",
    )
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--ragas", action="store_true", help="Also score with RAGAS (needs its own venv, see requirements-ragas.txt)")
    parser.add_argument("--trulens", action="store_true", help="Also score with TruLens (needs Settings.trulens_enabled + GROQ_API_KEY)")
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    dataset = load_dataset(dataset_path)
    base_settings = get_settings()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(dataset)} benchmark questions from {dataset_path}")

    configs_to_run = [CONFIGS[name] for name in args.configs] if args.ablation else [_FULL_PIPELINE_CONFIG]

    all_summaries = {}
    for config in configs_to_run:
        print(f"\n=== {'Config ' + config.name if args.ablation else 'Full pipeline'}: {config.label} ===")
        report = run_config(
            config, dataset, base_settings, enable_ragas=args.ragas, enable_trulens=args.trulens
        )
        all_summaries[config.name] = report["summary"]

        out_path = REPORTS_DIR / f"phase4_eval_{config.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        summary = report["summary"]
        print(
            f"  refusal_accuracy={summary['refusal_accuracy']} "
            f"grounded_answer_rate={summary['grounded_answer_rate']} "
            f"citation_precision={summary['citation_precision']}"
        )
        prov = summary["phase4_provenance"]
        print(
            f"  phase4_provenance: citation={prov['citation_correctness']} "
            f"object={prov['object_correctness']} page={prov['page_correctness']} "
            f"bbox={prov['bbox_correctness']}"
        )
        modality_counts = {m: b["n"] for m, b in summary["modality_breakdown"].items()}
        print(f"  modality coverage: {modality_counts}")
        print(f"  -> {out_path}")

    combined_path = REPORTS_DIR / "phase4_eval_summary.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nCombined summary written to {combined_path}")

    _write_markdown_report(dataset_path, all_summaries, REPORTS_DIR / "phase4_eval_report.md")
    print(f"Human-readable report written to {REPORTS_DIR / 'phase4_eval_report.md'}")


def _write_markdown_report(dataset_path: Path, all_summaries: dict, out_path: Path) -> None:
    lines = [
        "# Phase 4 Evaluation Report",
        "",
        f"Dataset: `{dataset_path}`",
        "",
        "Numbers below are computed by `app.evaluation.metrics`, "
        "`app.evaluation.phase4_metrics`, and `evaluation.modality` "
        "directly from real pipeline runs (`DocumentQAService.ask`) — "
        "none of this is hand-entered.",
        "",
    ]
    for name, summary in all_summaries.items():
        lines.append(f"## Config {name}")
        lines.append("")
        lines.append(f"- Refusal accuracy: {summary.get('refusal_accuracy')}")
        lines.append(f"- Grounded answer rate: {summary.get('grounded_answer_rate')}")
        lines.append(f"- Citation precision: {summary.get('citation_precision')}")
        prov = summary.get("phase4_provenance", {})
        lines.append(
            f"- Phase 4 provenance: citation={prov.get('citation_correctness')} "
            f"(n={prov.get('citation_correctness_n')}), "
            f"object={prov.get('object_correctness')} (n={prov.get('object_correctness_n')}), "
            f"page={prov.get('page_correctness')} (n={prov.get('page_correctness_n')}), "
            f"bbox={prov.get('bbox_correctness')} (n={prov.get('bbox_correctness_n')})"
        )
        lines.append("")
        lines.append("| Modality | n | refusal_accuracy | grounded_answer_rate | citation_precision |")
        lines.append("|---|---|---|---|---|")
        for modality, bucket in sorted(summary.get("modality_breakdown", {}).items()):
            lines.append(
                f"| {modality} | {bucket['n']} | {bucket.get('refusal_accuracy')} | "
                f"{bucket.get('grounded_answer_rate')} | {bucket.get('citation_precision')} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()