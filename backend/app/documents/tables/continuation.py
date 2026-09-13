"""Multi-page table continuation detection (Phase 4 spec section 7).

Runs AFTER per-page/per-file table extraction (`pdf_parser.py` already
produced one independent `TableData` per page) — this module only ever
LINKS tables that already exist; it never merges their rows into one
combined grid, and it never invents content. Each page's `TableData`
keeps its own `rows`/`cells`/`bbox` exactly as extracted, preserving
per-page provenance (spec requirement) — continuation is expressed
purely as metadata (`continuation_group_id`, `page_start`/`page_end`,
`continued_from_previous_page`/`continued_to_next_page`) layered onto
the existing, unmodified tables.

Heuristic (deliberately conservative — a false-positive continuation
link is worse than a missed one, since it would make `to_markdown()`
readers assume rows belong together that don't): two tables on
consecutive pages of the SAME source document are linked only when ALL
of:

1. Same column count (`n_cols`).
2. The earlier table's LAST row and the later table's FIRST row are not
   near-duplicates of each other (rules out two independent tables that
   simply repeat the same header row by coincidence).
3. The later table's first row does NOT look like a fresh header —
   approximated by: it doesn't share >= half its cell texts with the
   earlier table's own header row(s). A genuine continuation's first row
   is body data, not another copy of the header; a table that repeats
   its header on every page (common PDF-generator behavior) is DELIBERATELY
   still linked as a continuation if this check passes on the row AFTER
   the repeated header — see `_strip_repeated_header`.
"""

from __future__ import annotations

import dataclasses

from app.documents.models import TableData, _stable_object_id

_MIN_SHARED_HEADER_FRACTION = 0.5


def _row_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    a_norm = [c.strip().lower() for c in a]
    b_norm = [c.strip().lower() for c in b]
    n = min(len(a_norm), len(b_norm))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a_norm[i] == b_norm[i] and a_norm[i])
    return matches / max(len(a_norm), len(b_norm))


def _strip_repeated_header(table: TableData, header_rows: tuple[tuple[str, ...], ...]) -> TableData:
    """If a later page's table starts by repeating the earlier table's
    header row(s) verbatim (a common PDF-generator pattern: "repeat
    header on every page"), drop those repeated rows from `rows` before
    treating the remainder as continuation body — otherwise the header
    would appear twice in the logical, continued table. Cell-level
    `cells` (if any) for the dropped rows are dropped too, and every
    remaining cell's `row` index is shifted down to stay consistent with
    the shortened `rows` grid.
    """
    n_header = len(header_rows)
    if n_header == 0 or len(table.rows) <= n_header:
        return table
    if table.rows[:n_header] != header_rows:
        return table

    new_rows = table.rows[n_header:]
    new_cells = tuple(
        dataclasses.replace(c, row=c.row - n_header) for c in table.cells if c.row >= n_header
    )
    return dataclasses.replace(table, rows=new_rows, cells=new_cells, header_row_count=0)


def detect_continuations(tables_by_page: list[tuple[int, TableData]], source_path: str) -> list[TableData]:
    """`tables_by_page`: `[(page_number, table), ...]`, already in
    document order, ideally containing every table extracted from ONE
    source file (continuation across different files is never valid —
    callers must not mix tables from different documents into one call).

    Returns a new list of `TableData`, same length and order as the
    input, each with continuation metadata filled in where a real link
    was found (repeated-header rows stripped per `_strip_repeated_header`
    where applicable) — every entry that ISN'T part of a continuation is
    returned completely unchanged (`table_id` still gets assigned, since
    every table needs one for citation purposes, but every
    continuation-specific field stays at its default "not continued").
    """
    n = len(tables_by_page)
    out: list[TableData | None] = [None] * n

    # Assign a stable per-table ID first (every table needs one,
    # continued or not) — derived from source path + page + position
    # among this page's tables, so it's stable across re-ingestion.
    per_page_index: dict[int, int] = {}
    table_ids: list[str] = []
    for page_number, _ in tables_by_page:
        idx = per_page_index.get(page_number, 0)
        table_ids.append(_stable_object_id(source_path, "table", str(page_number), str(idx)))
        per_page_index[page_number] = idx + 1

    for i, (page_number, table) in enumerate(tables_by_page):
        out[i] = dataclasses.replace(table, table_id=table_ids[i])

    for i in range(n - 1):
        page_a, table_a = tables_by_page[i]
        page_b, table_b = tables_by_page[i + 1]
        cur_a = out[i]
        assert cur_a is not None

        if page_b != page_a + 1:
            continue  # not adjacent pages — never a continuation
        if cur_a.n_cols == 0 or table_b.n_cols != cur_a.n_cols:
            continue  # different shape — not the same logical table
        if not table_b.rows:
            continue

        header_rows = cur_a.rows[: max(cur_a.header_row_count, 0)]
        candidate_b = _strip_repeated_header(table_b, header_rows) if header_rows else table_b
        if not candidate_b.rows:
            continue  # table_b was ENTIRELY the repeated header — nothing left to link as body

        # Reject if table_b's first row still looks like a fresh header
        # (shares most cells with table_a's header) even after stripping
        # an exact repeat — e.g. a similar-but-reworded header on an
        # unrelated table.
        if header_rows and _row_similarity(candidate_b.rows[0], header_rows[0]) >= _MIN_SHARED_HEADER_FRACTION:
            continue
        # Reject two independent tables whose data rows simply happen to
        # look alike (last row of A vs. first row of B being identical
        # is far more likely coincidence/repeated-total-row than a
        # genuine split-across-pages continuation of DIFFERENT data).
        if cur_a.rows and _row_similarity(cur_a.rows[-1], candidate_b.rows[0]) >= 0.95:
            continue

        group_id = cur_a.continuation_group_id or _stable_object_id(source_path, "table_continuation", str(page_a))
        page_start = cur_a.page_start or page_a
        out[i] = dataclasses.replace(
            cur_a,
            continuation_group_id=group_id,
            page_start=page_start,
            page_end=page_b,
            continued_to_next_page=True,
        )
        out[i + 1] = dataclasses.replace(
            candidate_b,
            table_id=table_ids[i + 1],
            continuation_group_id=group_id,
            page_start=page_start,
            page_end=page_b,
            continued_from_previous_page=True,
        )

    _normalize_group_page_ranges(out, table_ids)
    return [t for t in out if t is not None]


def _normalize_group_page_ranges(out: list[TableData | None], table_ids: list[str]) -> None:
    """A chain longer than 2 pages (A->B->C) is built one pairwise link at
    a time above, so when the B->C link is discovered, A's `page_end` (set
    while processing A->B) doesn't automatically extend to C — this pass
    fixes that by giving every member of a `continuation_group_id` the
    SAME, group-wide `page_start`/`page_end` (min/max across the whole
    group) as a final normalization step, rather than trying to
    thread the correct running range through the pairwise loop itself.
    """
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(out):
        if t is not None and t.continuation_group_id:
            groups.setdefault(t.continuation_group_id, []).append(i)

    for group_id, indices in groups.items():
        starts = [out[i].page_start for i in indices if out[i].page_start is not None]
        ends = [out[i].page_end for i in indices if out[i].page_end is not None]
        if not starts or not ends:
            continue
        page_start, page_end = min(starts), max(ends)
        for i in indices:
            out[i] = dataclasses.replace(out[i], page_start=page_start, page_end=page_end)
