"""Modality classification + breakdown for Phase 4 multimodal evaluation.

Product spec (implement_4_7): "Make the ablation workflow multimodal
-aware. Results should be separable by: text, OCR, table, figure, chart,
visual, multi-hop, cross-modal, unanswerable." This module is the one
place that decides which modality bucket a benchmark row belongs to, and
the one place that groups+summarizes results by bucket — used by both
`evaluation/run_phase4_eval.py` (the new end-to-end runner) and
`evaluation/run_ablation.py` (extended, not replaced, to attach a
`modality_breakdown` to its existing summary — see that file's
`run_config`), so the two never compute this differently.

Backward compatible with the original `evaluation/datasets/v1.jsonl`
schema (`query_type`/`answerable` only): `classify_modality` falls back
to inferring a modality from those legacy fields when the newer,
explicit ones (`modality`, `requires_table`, `requires_visual`,
`requires_chart`, `requires_multi_hop`, `should_refuse`) aren't present
on a row, so every existing benchmark question still gets a bucket.
"""

from __future__ import annotations

from app.evaluation.metrics import summarize

# Canonical modality buckets (spec order).
MODALITIES = (
    "text",
    "ocr",
    "table",
    "figure",
    "chart",
    "visual",
    "multi_hop",
    "cross_modal",
    "unanswerable",
)

_LEGACY_QUERY_TYPE_MODALITY = {
    "OCR_QUERY": "ocr",
    "TABLE_QUERY": "table",
    "MULTI_HOP": "multi_hop",
    "CROSS_DOCUMENT": "cross_modal",
    "UNANSWERABLE": "unanswerable",
    "AMBIGUOUS": "unanswerable",
}


def classify_modality(row: dict) -> str:
    """One primary modality bucket for a benchmark row.

    Priority (most specific/explicit first):
    1. An explicit `modality` field on the row, if the expanded dataset
       schema set one directly — always trusted as-is.
    2. `should_refuse` / legacy `answerable is False` -> "unanswerable"
       (a refusal test is evaluated on refusal correctness, not on
       whatever modality the unanswerable question happens to mention).
    3. `requires_chart` / `requires_visual` / `requires_table` flags —
       chart takes priority over generic visual, which takes priority
       over table, matching how specific the ground truth actually is.
    4. `requires_multi_hop` -> "multi_hop".
    5. Legacy `query_type` mapping (`OCR_QUERY` -> "ocr", etc).
    6. Default: "text".

    A row can genuinely need MORE than one modality (e.g. "table AND
    chart") — `classify_modality` returns the single PRIMARY bucket used
    for grouping in a summary table; `cross_modal` is reserved for rows
    that explicitly combine text with a visual/table modality (spec:
    "text + table evidence", "text + visual evidence", "text + chart
    evidence") via `modality: \"cross_modal\"` or `requires_table AND
    requires_visual` both being true.
    """
    explicit = row.get("modality")
    if explicit:
        return explicit

    if row.get("should_refuse") is True or row.get("answerable") is False:
        return "unanswerable"

    requires_table = bool(row.get("requires_table"))
    requires_visual = bool(row.get("requires_visual"))
    requires_chart = bool(row.get("requires_chart"))
    requires_multi_hop = bool(row.get("requires_multi_hop"))

    combined_modalities = sum([requires_table, requires_visual or requires_chart])
    if combined_modalities >= 2:
        return "cross_modal"
    if requires_chart:
        return "chart"
    if requires_visual:
        return "visual"
    if requires_table:
        return "table"
    if requires_multi_hop:
        return "multi_hop"

    legacy = _LEGACY_QUERY_TYPE_MODALITY.get(row.get("query_type", ""))
    if legacy:
        return legacy

    return "text"


def group_by_modality(rows: list[dict], results_by_id: dict[str, dict]) -> dict[str, list[dict]]:
    """Group already-computed result records by their source row's modality.

    `rows` is the benchmark dataset (for `classify_modality`); `results`
    (keyed by `id`) is what `run_phase4_eval.py`/`run_ablation.py`
    already produced per question — this never re-runs the pipeline, only
    re-buckets what already ran.
    """
    buckets: dict[str, list[dict]] = {m: [] for m in MODALITIES}
    for row in rows:
        result = results_by_id.get(row["id"])
        if result is None:
            continue
        modality = classify_modality(row)
        buckets.setdefault(modality, []).append(result)
    return {k: v for k, v in buckets.items() if v}


def summarize_by_modality(
    rows: list[dict], results: list[dict], latencies_by_id: dict[str, float] | None = None
) -> dict[str, dict]:
    """`app.evaluation.metrics.summarize`, computed separately per modality bucket.

    Reuses `summarize` exactly as `run_ablation.py` already does for the
    overall run — this is purely the SAME aggregate function applied to
    subsets, never a second metrics implementation.
    """
    results_by_id = {r["id"]: r for r in results if "id" in r}
    buckets = group_by_modality(rows, results_by_id)
    latencies_by_id = latencies_by_id or {}

    breakdown: dict[str, dict] = {}
    for modality, bucket_results in buckets.items():
        latencies = [latencies_by_id[r["id"]] for r in bucket_results if r["id"] in latencies_by_id]
        breakdown[modality] = {
            "n": len(bucket_results),
            **summarize(bucket_results, latencies_seconds=latencies or None),
        }
    return breakdown
