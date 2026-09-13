"""Best-effort table reconstruction: OCR word boxes (heuristic) and
native pdfplumber tables (cell-accurate, Phase 4).

This module covers two different reliability tiers, and both are honest
about which tier they're in (see docs/tables.md "Confidence, not
certainty"):

1. `detect_tables` — OCR word-box heuristic (scanned pages / images).
   Groups OCR words into lines, detects candidate table regions, and
   emits a flat `TableData` grid. Cell-level structure (merges, header
   hierarchy) is NOT attempted here — there's no reliable geometric
   signal for it in OCR word boxes alone, and fabricating merge
   boundaries would be worse than not claiming them (see `TableData`'s
   docstring in app/documents/models.py).

2. `cells_from_pdfplumber_table` — native PDF tables (Phase 4, problems
   #21-26). pdfplumber's `Table.rows` gives REAL per-cell bounding boxes
   (not a heuristic) for every unmerged/anchor cell; a merged cell's
   anchor position reports a bbox wider/taller than a normal cell, with
   the positions it covers left as `None`. Turning that bbox size back
   into an integer rowspan/colspan is the one heuristic step (rounding
   to the nearest multiple of the smallest cell width/height seen in the
   table) — see that function's docstring for exactly what is and isn't
   guaranteed. Every emitted `Cell` still carries a real,
   pdfplumber-derived bbox and confidence 1.0 (native extraction).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from app.documents.models import BoundingBox, Cell, CoordinateSpace, TableData, _stable_object_id
from app.documents.ocr.engine import OcrLine, OcrWord

# A horizontal gap between two words wider than this multiple of the
# preceding word's height is treated as a column boundary rather than a
# normal inter-word space. Tuned for typical 150-300 DPI renders.
_GAP_HEIGHT_RATIO = 1.8
_MIN_CELL_GROUPS_FOR_TABLE_ROW = 2
_MIN_TABLE_ROWS = 2
_COLUMN_CLUSTER_TOLERANCE_PX = 40


@dataclass(frozen=True)
class CellGroup:
    text: str
    x0: float
    x1: float
    y0: float = 0.0
    y1: float = 0.0
    confidence: float = 0.0  # mean OCR word confidence (0-100) across this group's words


def _split_line_into_cell_groups(line: OcrLine) -> list[CellGroup]:
    words = sorted(line.words, key=lambda w: w.left)
    if not words:
        return []
    groups: list[list[OcrWord]] = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        gap = cur.left - (prev.left + prev.width)
        threshold = max(prev.height, cur.height) * _GAP_HEIGHT_RATIO
        if gap > threshold:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return [
        CellGroup(
            text=" ".join(w.text for w in g),
            x0=min(w.left for w in g),
            x1=max(w.left + w.width for w in g),
            y0=min(w.top for w in g),
            y1=max(w.top + w.height for w in g),
            confidence=sum(w.confidence for w in g) / len(g),
        )
        for g in groups
    ]


def _cluster_columns(all_groups: list[CellGroup]) -> list[tuple[float, float]]:
    """Cluster cell-group x-ranges into column bins (sorted by x0)."""
    starts = sorted({g.x0 for g in all_groups})
    bins: list[list[float]] = []
    for x in starts:
        if bins and x - bins[-1][-1] <= _COLUMN_CLUSTER_TOLERANCE_PX:
            bins[-1].append(x)
        else:
            bins.append([x])
    return [(min(b), max(b)) for b in bins]


def _assign_to_column(group: CellGroup, columns: list[tuple[float, float]]) -> int:
    best_idx, best_dist = 0, float("inf")
    for i, (lo, hi) in enumerate(columns):
        dist = abs(group.x0 - lo)
        if dist < best_dist:
            best_idx, best_dist = i, dist
    return best_idx


def _column_slot_boundaries(columns: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The x-range each column "owns" for overlap testing (Phase 4 OCR
    cell geometry): the midpoint between one column's cluster and the
    next's, rather than each column's own (much narrower) cluster
    extent — a cell-group's x0/x1 legitimately extends past its own
    column's clustered word positions (e.g. a right-aligned number), so
    testing overlap against the NEIGHBORING boundary (not the cluster
    itself) is what actually distinguishes "slightly wide" from
    "genuinely spans into the next column".
    """
    n = len(columns)
    bounds = []
    for i, (lo, hi) in enumerate(columns):
        left = -math.inf if i == 0 else (columns[i - 1][1] + lo) / 2.0
        right = math.inf if i == n - 1 else (hi + columns[i + 1][0]) / 2.0
        bounds.append((left, right))
    return bounds


def _estimate_col_span(group: CellGroup, columns: list[tuple[float, float]], boundaries: list[tuple[float, float]]) -> tuple[int, int]:
    """Return `(start_column, col_span)` for one OCR cell-group.

    Deliberately conservative (Phase 4 spec section 5, "do not fabricate
    geometry"): a group must overlap the FULL slot of every additional
    column it's claimed to span (not just brush its edge) before
    `col_span` exceeds 1. A group that merely runs a little wide into a
    neighboring column's boundary (common with proportional fonts /
    right-aligned numbers) stays `col_span=1` — this only fires for a
    cell-group that clearly, substantially overlaps more than one
    column's owned x-range, e.g. a table title/merged header cell
    printed across several columns.
    """
    start_col = _assign_to_column(group, columns)
    span = 1
    for c in range(start_col + 1, len(columns)):
        left, right = boundaries[c]
        if left is math.inf:
            break
        overlap = min(group.x1, right) - max(group.x0, left)
        slot_width = right - left if right not in (math.inf,) and left not in (-math.inf,) else 0.0
        if slot_width > 0 and overlap >= slot_width * 0.6:
            span += 1
        else:
            break
    return start_col, span


def detect_tables(lines: list[OcrLine], object_id_prefix: str = "") -> list[tuple[TableData, float, tuple[int, int]]]:
    """Return `(table, confidence, (start_line_idx, end_line_idx))` for each
    detected table region among `lines` (sorted top-to-bottom already).

    Populates `TableData.cells` (Phase 4 spec section 5, "OCR table cell
    geometry") with real per-cell bounding boxes derived from the OCR
    word boxes that made up each cell-group, and a best-effort
    `col_span` for cell-groups that clearly overlap more than one
    detected column (see `_estimate_col_span`). `row_span` is always 1:
    unlike column boundaries (which cell-group x-ranges give real
    geometric evidence for), OCR word boxes carry NO reliable signal for
    vertical (row) merges — a blank cell below a tall cell looks
    identical, from word boxes alone, to two unrelated blank cells. Per
    this module's "do not fabricate geometry" rule, row-span detection
    for OCR tables is left undone rather than guessed; `TableData.cells`
    still correctly reports `has_merged_cells=True` whenever a real
    `col_span` > 1 was found.

    `object_id_prefix`, when given, should identify the parent document
    (e.g. `f"{source_path}::p{page_number}"`) so cell object_ids stay
    stable across re-ingestion of an unchanged file; per-table-in-page
    disambiguation (`::table{N}`) is added internally. Cells produced
    here are in the SAME (pre-`CoordinateTransform`) pixel space as
    `OcrWord`/the table's own bbox — the caller (`pdf_parser.py`'s
    `_parse_scanned_page`, `image_parser.py`'s equivalent) is
    responsible for mapping every cell bbox back to the document's
    canonical coordinate space exactly as it already does for the
    table's own bbox, via `CoordinateTransform.bbox_to_original`.
    """
    per_line_groups = [_split_line_into_cell_groups(line) for line in lines]
    is_table_row = [len(g) >= _MIN_CELL_GROUPS_FOR_TABLE_ROW for g in per_line_groups]

    results: list[tuple[TableData, float, tuple[int, int]]] = []
    i = 0
    n = len(lines)
    table_index = 0
    while i < n:
        if not is_table_row[i]:
            i += 1
            continue
        j = i
        while j < n and is_table_row[j]:
            j += 1
        # region is [i, j)
        if j - i >= _MIN_TABLE_ROWS:
            region_groups = per_line_groups[i:j]
            flat = [g for row in region_groups for g in row]
            columns = _cluster_columns(flat)
            boundaries = _column_slot_boundaries(columns)
            n_cols = len(columns)

            prefix = f"{object_id_prefix}::table{table_index}" if object_id_prefix else f"ocr_table{table_index}"

            grid: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(j - i)]
            cells: list[Cell] = []
            covered: set[tuple[int, int]] = set()
            for row_idx, row_groups in enumerate(region_groups):
                for g in row_groups:
                    start_col, col_span = _estimate_col_span(g, columns, boundaries)
                    col_span = min(col_span, n_cols - start_col)
                    if (row_idx, start_col) in covered:
                        continue
                    for cc in range(start_col, start_col + col_span):
                        covered.add((row_idx, cc))
                    if start_col < n_cols:
                        # The anchor (start) column holds the text; any
                        # additional spanned columns are left blank in the
                        # raw `rows` grid — matching `rows`' existing
                        # convention for a native pdfplumber merge (see
                        # `Cell`'s docstring, "never duplicate merged-cell
                        # content"). `TableData.grid()` is what expands a
                        # merge's text into every position it visually
                        # covers, for display.
                        grid[row_idx][start_col] = g.text
                    cells.append(
                        Cell(
                            text=g.text,
                            row=row_idx,
                            column=start_col,
                            row_span=1,
                            col_span=col_span,
                            is_header=(row_idx == 0),
                            bbox=BoundingBox(x0=g.x0, y0=g.y0, x1=g.x1, y1=g.y1, coordinate_space=CoordinateSpace.UNSPECIFIED),
                            confidence=round((g.confidence / 100.0) if g.confidence > 1.0 else g.confidence, 3),
                            object_id=_stable_object_id(prefix, "cell", str(row_idx), str(start_col)),
                        )
                    )
            grid_rows = tuple(tuple(row) for row in grid)

            region_words = [w for k in range(i, j) for w in lines[k].words]
            mean_conf = sum(w.confidence for w in region_words) / len(region_words) if region_words else 0.0
            row_lengths = [len(row_groups) for row_groups in region_groups]
            consistency = 1.0 - (max(row_lengths) - min(row_lengths)) / max(max(row_lengths), 1)
            confidence = round((mean_conf / 100.0) * 0.7 + consistency * 0.3, 3)

            cells.sort(key=lambda c: (c.row, c.column))
            results.append((TableData(rows=grid_rows, cells=tuple(cells)), confidence, (i, j - 1)))
            table_index += 1
        i = j
    return results


def cells_from_pdfplumber_table(table, object_id_prefix: str) -> tuple[Cell, ...]:
    """Build real `Cell` objects (Phase 4) from a pdfplumber `Table`.

    `table.rows` gives one `Row` per logical grid row; each `Row.cells` is
    a list of `(x0, top, x1, bottom)` bboxes (or `None`) — one entry per
    column slot. This is real geometry, not a heuristic guess about
    where cells are: pdfplumber represents a merged cell by giving its
    ANCHOR position (top-left of the merge) a bbox that is wider/taller
    than a normal single cell, and leaving every OTHER grid position the
    merge covers as `None`.

    What IS a heuristic here — because pdfplumber doesn't expose an
    explicit "this bbox spans N columns" integer, only the bbox itself —
    is turning "this bbox is wider/taller than normal" into an actual
    colspan/rowspan count: this function takes, PER COLUMN, the width
    that recurs most often across that column's rows as the "normal"
    (unspanned) width for that column specifically — not one single
    global unit size, since a table's columns are very often naturally
    different widths from each other even with zero merges (e.g. a
    narrow "Q1" next to a wide "Region") — and rounds each cell's own
    width to the nearest multiple of its column's normal width (same
    idea per-row for height). This is reliable for the regular, gridded
    tables Veridoc targets (see docs/tables.md) and is exactly the
    geometric information available; it is not claimed to be exact for
    tables with irregular, non-grid-aligned column widths.

    `object_id_prefix` should uniquely identify the parent table (e.g.
    an already-computed table `object_id`) so each cell's own
    `object_id` is deterministic across re-ingestion of an unchanged
    file (see `_stable_object_id`).
    """
    try:
        rows = table.rows
        text_grid = table.extract()
    except Exception:
        return ()
    if not rows or not text_grid:
        return ()

    n_rows = len(rows)
    n_cols = max((len(r.cells) for r in rows), default=0)
    if n_cols == 0:
        return ()

    # Per-column reference width / per-row reference height, NOT one
    # global unit size: a table's columns/rows are very often naturally
    # different sizes from each other (e.g. a narrow "Q1" column next to
    # a wide "Region" column) even with zero merged cells, so a single
    # global "smallest cell" unit produces false-positive merges (an
    # unmerged wide column looking like it "spans 2 of the narrow unit").
    # Instead: for each column index, the WIDTH that recurs most often
    # across every row at that column is almost certainly that column's
    # true (unspanned) width — an anchor cell's wider bbox is the rare
    # outlier, not the mode. Ties broken toward the smaller value, since
    # a span is by definition >= the normal size, never smaller.
    from collections import Counter

    col_widths: list[list[float]] = [[] for _ in range(n_cols)]
    row_heights: list[list[float]] = [[] for _ in range(n_rows)]
    for r, row in enumerate(rows):
        for c in range(n_cols):
            bbox = row.cells[c] if c < len(row.cells) else None
            if bbox is not None:
                col_widths[c].append(round(bbox[2] - bbox[0], 2))
                row_heights[r].append(round(bbox[3] - bbox[1], 2))

    def _reference_size(values: list[float], fallback: float) -> float:
        if not values:
            return fallback
        counts = Counter(values)
        best_count = max(counts.values())
        return min(v for v, n in counts.items() if n == best_count)

    fallback_width = min((w for col in col_widths for w in col), default=1.0)
    fallback_height = min((h for row in row_heights for h in row), default=1.0)
    col_unit = [_reference_size(col_widths[c], fallback_width) for c in range(n_cols)]
    row_unit = [_reference_size(row_heights[r], fallback_height) for r in range(n_rows)]

    def _cell_text(r: int, c: int) -> str:
        try:
            return (text_grid[r][c] or "").strip()
        except IndexError:
            return ""

    def _span(size: float, unit: float) -> int:
        if unit <= 0:
            return 1
        return max(1, round(size / unit))

    covered: set[tuple[int, int]] = set()
    cells: list[Cell] = []

    for r, row in enumerate(rows):
        for c in range(n_cols):
            if (r, c) in covered:
                continue
            bbox = row.cells[c] if c < len(row.cells) else None
            if bbox is None:
                # Not an anchor and not (yet) marked covered by an
                # earlier anchor's span estimate -- pdfplumber found no
                # rect here at all. Emit a safe, bbox-less 1x1 cell
                # rather than silently dropping the grid position.
                cells.append(
                    Cell(
                        text=_cell_text(r, c),
                        row=r,
                        column=c,
                        is_header=(r == 0),
                        bbox=None,
                        confidence=1.0,
                        object_id=_stable_object_id(object_id_prefix, "cell", str(r), str(c)),
                    )
                )
                continue

            col_span = min(_span(bbox[2] - bbox[0], col_unit[c]), n_cols - c)
            row_span = min(_span(bbox[3] - bbox[1], row_unit[r]), n_rows - r)
            for rr in range(r, r + row_span):
                for cc in range(c, c + col_span):
                    covered.add((rr, cc))

            cells.append(
                Cell(
                    text=_cell_text(r, c),
                    row=r,
                    column=c,
                    row_span=row_span,
                    col_span=col_span,
                    is_header=(r == 0),
                    bbox=BoundingBox(*bbox, coordinate_space=CoordinateSpace.PDF_POINTS),
                    confidence=1.0,
                    object_id=_stable_object_id(object_id_prefix, "cell", str(r), str(c)),
                )
            )

    cells.sort(key=lambda c: (c.row, c.column))
    _mark_multi_row_header(cells, n_rows, n_cols)
    return tuple(cells)


def header_row_count_for(cells: tuple[Cell, ...]) -> int:
    """How many leading rows are marked `is_header` — for setting
    `TableData.header_row_count` (see `_mark_multi_row_header`).
    Returns 1 (the always-safe single-header-row default) if `cells` is
    empty or nothing is marked as a header.
    """
    header_rows = {c.row for c in cells if c.is_header}
    return max(header_rows) + 1 if header_rows else 1


def _mark_multi_row_header(cells: list[Cell], n_rows: int, n_cols: int) -> None:
    """Detect and mark a multi-row header block IN PLACE (Phase 4).

    Row 0 is always the header (existing behavior, unchanged for a
    normal single-row-header table). Row 1 is ALSO marked as header,
    extending the header block to 2 rows, when there's a real structural
    signal for it — a genuine grouped-header layout (e.g. "Region"
    spanning above "North | South") rather than merely "row 0 happens to
    look boldface" (which pdfplumber's geometry can't tell us anyway):

    - at least one row-0 cell has `col_span > 1` (a group heading sitting
      above narrower sub-columns — the actual geometric signature of a
      grouped header), AND
    - row 1 exists, has no row that is itself a continuation of a row-0
      rowspan (i.e., every grid position in row 1 has its OWN cell, not
      one it merely inherits from a row-0 cell spanning downward), AND
    - row 1 isn't just a data row that happens to sit under a spanned
      title cell — approximated by requiring row 1 to have MORE distinct
      cells than row 0 (a sub-header row necessarily subdivides at least
      one of row 0's grouped columns, so it has strictly more column
      entries than row 0's collapsed grouping).

    This intentionally does NOT try to detect a 3+ row header — that
    would need a much stronger signal than bbox geometry alone provides,
    and a wrong `header_row_count` silently corrupts every downstream
    consumer (chunking, to_markdown, citation cell lookup). Under
    -detecting (staying at 1 row) is the safe failure mode; this function
    only ever extends to 2, never guesses higher.
    """
    if n_rows < 2 or n_cols < 2:
        return

    row0_cells = [c for c in cells if c.row == 0]
    row1_cells = [c for c in cells if c.row == 1]
    if not row0_cells or not row1_cells:
        return

    row0_has_group_header = any(c.col_span > 1 for c in row0_cells)
    row1_is_finer_grained = len(row1_cells) > len(row0_cells)
    row1_fully_own_cells = all(c.row == 1 for c in row1_cells)  # no row-0 rowspan swallowed row 1

    if row0_has_group_header and row1_is_finer_grained and row1_fully_own_cells:
        for i, cell in enumerate(cells):
            if cell.row == 1:
                cells[i] = dataclasses.replace(cell, is_header=True)