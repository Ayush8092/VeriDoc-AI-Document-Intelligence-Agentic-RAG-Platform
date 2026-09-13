"""Phase 4 multimodal evaluation metrics.

Every function here is PURE — no network/model/DB call, no dependency on
a live pipeline run — so all of it is unit-testable with synthetic
inputs (see `tests/test_phase4_metrics.py`) without needing API keys or
an ingested corpus. This module only computes metrics FROM a prediction
+ ground-truth pair a caller (`evaluation/run_phase4_eval.py`, not yet
executed against live data — see that module's docstring) supplies; it
never generates or fabricates either side.

Grouped to match the Phase 4.1(4) spec's own metric list (Feature 9):
OCR (CER/WER), layout (bbox IoU, layout F1), table (row/column/cell/
header/merge/continuation accuracy), visual (detection/classification/
caption accuracy), chart (detection/type/field/value accuracy), and
citation/provenance correctness (object/page/bbox correctness) reused
alongside the existing `app/evaluation/metrics.py` /
`app/evaluation/retrieval_metrics.py` (grounding, retrieval ranking) —
this module does NOT duplicate those; it adds exactly the metric
families Phase 4 introduced that Phase 3A's evaluation layer had no
concept of (visual objects, charts, table structure, OCR ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# OCR: character/word error rate
# ============================================================


def _levenshtein(a: list, b: list) -> int:
    """Standard edit distance (insert/delete/substitute, unit cost) over
    two sequences (chars for CER, words for WER). O(len(a)*len(b)) time
    and O(min(len(a),len(b))) space — fine for the short OCR ground
    -truth snippets this is meant for; not intended for whole-document
    comparison.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def character_error_rate(hypothesis: str, reference: str) -> float | None:
    """CER = edit_distance(chars) / len(reference chars). `None` (not
    0.0 or 1.0) when `reference` is empty — there is no meaningful error
    rate against zero ground-truth characters, and reporting 0.0 would
    misleadingly read as "perfect".
    """
    if not reference:
        return None
    return round(_levenshtein(list(hypothesis), list(reference)) / len(reference), 4)


def word_error_rate(hypothesis: str, reference: str) -> float | None:
    """WER = edit_distance(words) / len(reference words). Same `None`
    -on-empty-reference convention as `character_error_rate`.
    """
    ref_words = reference.split()
    if not ref_words:
        return None
    return round(_levenshtein(hypothesis.split(), ref_words) / len(ref_words), 4)


# ============================================================
# Layout: bbox IoU, layout F1
# ============================================================


def bbox_iou(box_a: dict, box_b: dict) -> float:
    """Intersection-over-union of two bboxes, each `{x0, y0, x1, y1}`
    (matching `app.documents.models.BoundingBox`'s field names — pass
    `dataclasses.asdict(bbox)` or the API response's `bbox` dict
    directly). Returns 0.0 for a degenerate (zero-area) box on either
    side, or no overlap at all — never negative, never > 1.
    """
    ax0, ay0, ax1, ay1 = box_a["x0"], box_a["y0"], box_a["x1"], box_a["y1"]
    bx0, by0, bx1, by1 = box_b["x0"], box_b["y0"], box_b["x1"], box_b["y1"]

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return round(inter_area / union_area, 4)


@dataclass(frozen=True)
class LayoutF1Result:
    precision: float
    recall: float
    f1: float
    matched: int
    predicted_count: int
    ground_truth_count: int


def layout_f1(
    predicted_blocks: list[dict],
    ground_truth_blocks: list[dict],
    iou_threshold: float = 0.5,
    require_same_page: bool = True,
    require_same_type: bool = False,
) -> LayoutF1Result:
    """Greedy best-IoU-first matching between predicted and ground-truth
    blocks (each `{"bbox": {...}, "page_number": int, "block_type": str}`),
    same spirit as standard object-detection precision/recall: a
    predicted block counts as a true positive only if it's matched
    1-to-1 (no double-counting) to a ground-truth block whose IoU clears
    `iou_threshold` — and, if `require_same_page`/`require_same_type`,
    also agrees on page/type. Unmatched predictions are false positives;
    unmatched ground-truth blocks are false negatives.
    """
    gt_available = list(range(len(ground_truth_blocks)))
    candidate_pairs: list[tuple[float, int, int]] = []  # (iou, pred_idx, gt_idx)

    for pi, pred in enumerate(predicted_blocks):
        if not pred.get("bbox"):
            continue
        for gi in gt_available:
            gt = ground_truth_blocks[gi]
            if not gt.get("bbox"):
                continue
            if require_same_page and pred.get("page_number") != gt.get("page_number"):
                continue
            if require_same_type and pred.get("block_type") != gt.get("block_type"):
                continue
            iou = bbox_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidate_pairs.append((iou, pi, gi))

    candidate_pairs.sort(key=lambda t: t[0], reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for iou, pi, gi in candidate_pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    matched = len(matched_pred)
    precision = matched / len(predicted_blocks) if predicted_blocks else (1.0 if not ground_truth_blocks else 0.0)
    recall = matched / len(ground_truth_blocks) if ground_truth_blocks else (1.0 if not predicted_blocks else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return LayoutF1Result(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched=matched,
        predicted_count=len(predicted_blocks),
        ground_truth_count=len(ground_truth_blocks),
    )


def page_assignment_accuracy(predicted_pages: list[int | None], ground_truth_pages: list[int | None]) -> float | None:
    """Fraction of positionally-paired (predicted, ground_truth) blocks
    whose page number matches exactly. `None` if the lists are empty or
    mismatched in length (nothing meaningful to compare pairwise).
    """
    if not predicted_pages or len(predicted_pages) != len(ground_truth_pages):
        return None
    correct = sum(1 for p, g in zip(predicted_pages, ground_truth_pages) if p == g)
    return round(correct / len(predicted_pages), 4)


def ordering_accuracy(predicted_order: list[str], ground_truth_order: list[str]) -> float | None:
    """Fraction of adjacent PAIRS in `ground_truth_order` (by some stable
    ID — e.g. object_id or chunk_id) whose relative order is preserved in
    `predicted_order` — a standard, position-shift-tolerant way to score
    reading-order correctness (an exact-sequence-match would zero out a
    single swap's entire score, which overstates how wrong the ordering
    actually is). `None` if either list has fewer than 2 elements.
    """
    if len(ground_truth_order) < 2:
        return None
    position = {item_id: i for i, item_id in enumerate(predicted_order)}
    correct_pairs = 0
    total_pairs = 0
    for i in range(len(ground_truth_order) - 1):
        a, b = ground_truth_order[i], ground_truth_order[i + 1]
        if a not in position or b not in position:
            continue
        total_pairs += 1
        if position[a] < position[b]:
            correct_pairs += 1
    if total_pairs == 0:
        return None
    return round(correct_pairs / total_pairs, 4)


# ============================================================
# Table structure metrics
# ============================================================


def table_detection_accuracy(predicted_count: int, ground_truth_count: int) -> float:
    """1.0 if the predicted table count exactly matches ground truth,
    else a partial credit score reflecting how far off it is (never
    negative). Deliberately simple: table-level presence/absence, not
    IoU-based localization (use `layout_f1` with `block_type="table"`
    for that).
    """
    if ground_truth_count == 0:
        return 1.0 if predicted_count == 0 else 0.0
    return round(max(0.0, 1.0 - abs(predicted_count - ground_truth_count) / ground_truth_count), 4)


def _grid_cell_accuracy(predicted_rows: list[list[str]], ground_truth_rows: list[list[str]]) -> float | None:
    """Cell-by-cell exact-text-match accuracy over the OVERLAPPING grid
    region (min rows x min cols of the two grids) — a predicted grid
    that's a different shape than ground truth is scored on what does
    line up, since `table_row_accuracy`/`table_column_accuracy` already
    separately penalize a shape mismatch.
    """
    if not ground_truth_rows or not ground_truth_rows[0]:
        return None
    n_rows = min(len(predicted_rows), len(ground_truth_rows))
    if n_rows == 0:
        return 0.0
    n_cols = min(
        min((len(r) for r in predicted_rows[:n_rows]), default=0),
        min((len(r) for r in ground_truth_rows[:n_rows]), default=0),
    )
    if n_cols == 0:
        return 0.0
    total = n_rows * n_cols
    correct = sum(
        1
        for r in range(n_rows)
        for c in range(n_cols)
        if predicted_rows[r][c].strip() == ground_truth_rows[r][c].strip()
    )
    return round(correct / total, 4)


@dataclass(frozen=True)
class TableStructureResult:
    row_accuracy: float | None
    column_accuracy: float | None
    cell_accuracy: float | None
    header_accuracy: float | None
    merge_accuracy: float | None
    continuation_accuracy: float | None


def table_structure_accuracy(predicted: dict, ground_truth: dict) -> TableStructureResult:
    """Compare one predicted table (a dict shaped like
    `dataclasses.asdict(TableData)` — see `app.documents.models.TableData`)
    against its ground-truth counterpart.

    - `row_accuracy`/`column_accuracy`: 1.0 if `n_rows`/`n_cols` match
      exactly, else 0.0 (table shape is a discrete correctness question,
      not a partial-credit one — a table with the wrong shape has
      already misread the source).
    - `cell_accuracy`: see `_grid_cell_accuracy`.
    - `header_accuracy`: 1.0 if `header_row_count` matches exactly.
    - `merge_accuracy`: fraction of ground-truth cells whose
      `(row_span, col_span)` the predicted cell at the same (row, column)
      position also has — `None` if ground truth has no cells at all
      (nothing to check merges against).
    - `continuation_accuracy`: 1.0 if `continuation_group_id is not None`
      agrees between predicted and ground truth (i.e. "is this table part
      of a multi-page continuation, yes/no" was called correctly) — a
      coarse but honest signal; it does NOT check that the group's OTHER
      members also matched (that's `layout_f1`/`ordering_accuracy`'s job
      at the block level).
    """
    pred_rows = predicted.get("rows") or []
    gt_rows = ground_truth.get("rows") or []

    row_accuracy = 1.0 if len(pred_rows) == len(gt_rows) else 0.0
    pred_cols = len(pred_rows[0]) if pred_rows else 0
    gt_cols = len(gt_rows[0]) if gt_rows else 0
    column_accuracy = 1.0 if pred_cols == gt_cols else 0.0

    cell_accuracy = _grid_cell_accuracy(pred_rows, gt_rows)

    header_accuracy = None
    if "header_row_count" in ground_truth:
        header_accuracy = 1.0 if predicted.get("header_row_count") == ground_truth.get("header_row_count") else 0.0

    merge_accuracy = None
    gt_cells = ground_truth.get("cells") or []
    if gt_cells:
        pred_cell_by_pos = {(c["row"], c["column"]): c for c in (predicted.get("cells") or [])}
        matches = 0
        for gt_cell in gt_cells:
            pred_cell = pred_cell_by_pos.get((gt_cell["row"], gt_cell["column"]))
            if pred_cell is not None and (pred_cell.get("row_span", 1), pred_cell.get("col_span", 1)) == (
                gt_cell.get("row_span", 1),
                gt_cell.get("col_span", 1),
            ):
                matches += 1
        merge_accuracy = round(matches / len(gt_cells), 4)

    continuation_accuracy = None
    if "continuation_group_id" in ground_truth:
        pred_continued = bool(predicted.get("continuation_group_id"))
        gt_continued = bool(ground_truth.get("continuation_group_id"))
        continuation_accuracy = 1.0 if pred_continued == gt_continued else 0.0

    return TableStructureResult(
        row_accuracy=row_accuracy,
        column_accuracy=column_accuracy,
        cell_accuracy=cell_accuracy,
        header_accuracy=header_accuracy,
        merge_accuracy=merge_accuracy,
        continuation_accuracy=continuation_accuracy,
    )


# ============================================================
# Visual object metrics
# ============================================================


@dataclass(frozen=True)
class VisualDetectionResult:
    precision: float
    recall: float
    matched: int
    predicted_count: int
    ground_truth_count: int


def visual_detection_precision_recall(
    predicted_visuals: list[dict], ground_truth_visuals: list[dict], iou_threshold: float = 0.5
) -> VisualDetectionResult:
    """Same greedy IoU-matching approach as `layout_f1`, scoped to visual
    (figure/chart/visual) blocks specifically and returned as a
    dedicated result type so a caller doesn't have to filter
    `layout_f1`'s inputs down to visual block types itself.
    """
    result = layout_f1(predicted_visuals, ground_truth_visuals, iou_threshold=iou_threshold, require_same_type=False)
    return VisualDetectionResult(
        precision=result.precision,
        recall=result.recall,
        matched=result.matched,
        predicted_count=result.predicted_count,
        ground_truth_count=result.ground_truth_count,
    )


def visual_classification_accuracy(predicted_types: list[str], ground_truth_types: list[str]) -> float | None:
    """Fraction of positionally-paired (already-matched, e.g. via
    `visual_detection_precision_recall`) visuals whose `object_type`
    matches exactly. `None` if the lists are empty or mismatched length.
    """
    if not predicted_types or len(predicted_types) != len(ground_truth_types):
        return None
    correct = sum(1 for p, g in zip(predicted_types, ground_truth_types) if p == g)
    return round(correct / len(predicted_types), 4)


def caption_association_accuracy(predicted_captions: list[str | None], ground_truth_captions: list[str | None]) -> float | None:
    """Fraction of positionally-paired visuals where the predicted
    caption text matches ground truth (both `None` counts as a correct
    "no caption" agreement; one `None` and one non-`None` is incorrect).
    `None` (the metric itself) if the lists are empty/mismatched length.
    """
    if not predicted_captions or len(predicted_captions) != len(ground_truth_captions):
        return None
    correct = sum(
        1
        for p, g in zip(predicted_captions, ground_truth_captions)
        if (p or "").strip() == (g or "").strip()
    )
    return round(correct / len(predicted_captions), 4)


# ============================================================
# Chart understanding metrics
# ============================================================


def chart_detection_accuracy(predicted_is_chart: list[bool], ground_truth_is_chart: list[bool]) -> float | None:
    if not predicted_is_chart or len(predicted_is_chart) != len(ground_truth_is_chart):
        return None
    correct = sum(1 for p, g in zip(predicted_is_chart, ground_truth_is_chart) if p == g)
    return round(correct / len(predicted_is_chart), 4)


def chart_type_accuracy(predicted_types: list[str | None], ground_truth_types: list[str | None]) -> float | None:
    """Only over entries where ground truth actually specifies a chart
    type (an unclear/ambiguous chart's ground truth may legitimately be
    `None` — that pair is excluded rather than penalized either way).
    """
    pairs = [(p, g) for p, g in zip(predicted_types, ground_truth_types) if g is not None]
    if not pairs:
        return None
    correct = sum(1 for p, g in pairs if p == g)
    return round(correct / len(pairs), 4)


@dataclass(frozen=True)
class ChartFieldAccuracyResult:
    title_accuracy: float | None
    axis_accuracy: float | None
    legend_accuracy: float | None
    series_accuracy: float | None
    value_accuracy: float | None


def chart_field_accuracy(predicted: dict, ground_truth: dict) -> ChartFieldAccuracyResult:
    """Compare one chart's extracted `chart_data` dict (see
    `app.documents.figures.chart_understanding._validate_chart_json`'s
    return shape) against ground truth, field by field. Every sub
    -metric is `None` (not 0.0) when ground truth doesn't specify that
    field at all — "not evaluated" must stay distinguishable from
    "evaluated and wrong", especially for `value_accuracy`, which
    the spec explicitly restricts to "where ground truth exists".
    """

    def _text_match(p, g) -> float | None:
        if g is None:
            return None
        return 1.0 if (p or "").strip().lower() == (g or "").strip().lower() else 0.0

    title_accuracy = _text_match(predicted.get("title"), ground_truth.get("title"))

    axis_accuracy = None
    if ground_truth.get("x_axis_label") is not None or ground_truth.get("y_axis_label") is not None:
        x_match = _text_match(predicted.get("x_axis_label"), ground_truth.get("x_axis_label"))
        y_match = _text_match(predicted.get("y_axis_label"), ground_truth.get("y_axis_label"))
        scores = [s for s in (x_match, y_match) if s is not None]
        axis_accuracy = round(sum(scores) / len(scores), 4) if scores else None

    legend_accuracy = None
    gt_legend = ground_truth.get("legend")
    if gt_legend:
        pred_legend = set(x.strip().lower() for x in (predicted.get("legend") or []))
        gt_legend_set = set(x.strip().lower() for x in gt_legend)
        legend_accuracy = round(len(pred_legend & gt_legend_set) / len(gt_legend_set), 4)

    series_accuracy = None
    gt_series = ground_truth.get("series")
    if gt_series:
        gt_names = set((s.get("name") or "").strip().lower() for s in gt_series if s.get("name"))
        pred_names = set((s.get("name") or "").strip().lower() for s in (predicted.get("series") or []) if s.get("name"))
        series_accuracy = round(len(pred_names & gt_names) / len(gt_names), 4) if gt_names else None

    value_accuracy = None
    if gt_series:
        gt_values_by_name = {(s.get("name") or ""): s.get("values", []) for s in gt_series}
        pred_values_by_name = {(s.get("name") or ""): s.get("values", []) for s in (predicted.get("series") or [])}
        total, correct = 0, 0
        for name, gt_values in gt_values_by_name.items():
            pred_values = pred_values_by_name.get(name, [])
            for i, gt_v in enumerate(gt_values):
                total += 1
                if i < len(pred_values) and str(pred_values[i]).strip() == str(gt_v).strip():
                    correct += 1
        value_accuracy = round(correct / total, 4) if total else None

    return ChartFieldAccuracyResult(
        title_accuracy=title_accuracy,
        axis_accuracy=axis_accuracy,
        legend_accuracy=legend_accuracy,
        series_accuracy=series_accuracy,
        value_accuracy=value_accuracy,
    )


# ============================================================
# Citation / provenance correctness
# ============================================================


def object_correctness(predicted_object_ids: list[str | None], ground_truth_object_ids: list[str]) -> float | None:
    """Fraction of `ground_truth_object_ids` that appear ANYWHERE in
    `predicted_object_ids` — i.e. "did the citation set correctly
    identify the exact table/figure/chart object(s) it should have",
    independent of order. `None` if ground truth is empty (nothing
    object-level was required for this question).
    """
    if not ground_truth_object_ids:
        return None
    predicted_set = {oid for oid in predicted_object_ids if oid}
    correct = sum(1 for oid in ground_truth_object_ids if oid in predicted_set)
    return round(correct / len(ground_truth_object_ids), 4)


def page_correctness(predicted_pages: list[int | None], ground_truth_pages: list[int]) -> float | None:
    if not ground_truth_pages:
        return None
    predicted_set = {p for p in predicted_pages if p is not None}
    correct = sum(1 for p in ground_truth_pages if p in predicted_set)
    return round(correct / len(ground_truth_pages), 4)


def bbox_correctness(predicted_bboxes: list[dict], ground_truth_bboxes: list[dict], iou_threshold: float = 0.5) -> float | None:
    """Fraction of `ground_truth_bboxes` that have AT LEAST ONE predicted
    bbox clearing `iou_threshold` against them (each ground-truth box
    matched independently — not a 1-to-1 assignment like `layout_f1`,
    since this asks "was the right region cited at all", not "how many
    extra/missing regions were there").
    """
    if not ground_truth_bboxes:
        return None
    correct = 0
    for gt_box in ground_truth_bboxes:
        if any(bbox_iou(pred_box, gt_box) >= iou_threshold for pred_box in predicted_bboxes):
            correct += 1
    return round(correct / len(ground_truth_bboxes), 4)


def citation_correctness(
    predicted_chunk_ids: list[str], ground_truth_chunk_ids: list[str], should_refuse: bool = False
) -> float | None:
    """Whole-citation-set correctness for one question: 1.0 if
    `should_refuse` and no citations were given (a correct refusal has
    an empty, therefore "correct", citation set); otherwise the F1
    overlap between predicted and ground-truth chunk_id sets. `None`
    only when there's nothing to evaluate (`should_refuse=False` and
    `ground_truth_chunk_ids` is empty — a malformed benchmark record).
    """
    if should_refuse:
        return 1.0 if not predicted_chunk_ids else 0.0
    if not ground_truth_chunk_ids:
        return None
    pred_set, gt_set = set(predicted_chunk_ids), set(ground_truth_chunk_ids)
    if not pred_set:
        return 0.0
    precision = len(pred_set & gt_set) / len(pred_set)
    recall = len(pred_set & gt_set) / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)