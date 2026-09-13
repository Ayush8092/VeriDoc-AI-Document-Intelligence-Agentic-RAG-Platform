from pathlib import Path

from app.documents.models import BlockType
from app.documents.parsers.pdf_parser import parse_pdf


def _build_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Report Title", styles["Title"]),
        Spacer(1, 12),
        Paragraph("This is a native-text paragraph for extraction testing.", styles["Normal"]),
        Spacer(1, 12),
    ]
    data = [["Region", "Q1"], ["North", "120"], ["South", "90"]]
    t = Table(data)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, (0, 0, 0))]))
    elements.append(t)
    doc.build(elements)


def test_parse_native_pdf_extracts_text_and_table(tmp_path: Path):
    f = tmp_path / "report.pdf"
    _build_pdf(f)

    doc = parse_pdf(f)

    assert doc.pages[0].is_scanned is False
    paragraph_texts = " ".join(b.text for b in doc.blocks if b.block_type == BlockType.PARAGRAPH)
    assert "native-text paragraph" in paragraph_texts

    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].table.rows[0] == ("Region", "Q1")
    assert ("North", "120") in tables[0].table.rows
    assert tables[0].source.value == "native"
    assert tables[0].confidence == 1.0


def test_parse_pdf_missing_file_raises(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError):
        parse_pdf(tmp_path / "missing.pdf")


def _build_pdf_with_merged_header(path: Path) -> None:
    """A table whose top-left header cell spans 2 columns (colspan) via
    reportlab's SPAN command -- a real merged cell, not a simulated one,
    so `cells_from_pdfplumber_table`'s bbox-grouping merge detection
    (Phase 4, problems #21-26) is exercised against genuine pdfplumber
    geometry rather than a hand-built fixture."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

    doc = SimpleDocTemplate(str(path), pagesize=letter)
    # Row 0: "Revenue" spans both data columns (colspan=2); row 1: two
    # sub-headers under it; rows 2-3: data.
    data = [
        ["Region", "Revenue", ""],
        ["", "Q1", "Q2"],
        ["North", "120", "130"],
        ["South", "90", "95"],
    ]
    t = Table(data)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, (0, 0, 0)),
                ("SPAN", (1, 0), (2, 0)),  # "Revenue" spans columns 1-2 on row 0
                ("SPAN", (0, 0), (0, 1)),  # "Region" spans rows 0-1 on column 0
            ]
        )
    )
    doc.build([t])


def test_parse_native_pdf_detects_merged_cells_via_real_bbox_geometry(tmp_path: Path):
    f = tmp_path / "merged.pdf"
    _build_pdf_with_merged_header(f)

    doc = parse_pdf(f)
    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table

    assert table.has_merged_cells is True

    cells_by_pos = {(c.row, c.column): c for c in table.cells}
    # "Revenue" is the colspan=2 merge, anchored at (0, 1).
    revenue_cell = cells_by_pos[(0, 1)]
    assert revenue_cell.text == "Revenue"
    assert revenue_cell.col_span == 2
    assert revenue_cell.row_span == 1
    assert revenue_cell.bbox is not None

    # "Region" is the rowspan=2 merge, anchored at (0, 0).
    region_cell = cells_by_pos[(0, 0)]
    assert region_cell.text == "Region"
    assert region_cell.row_span == 2
    assert region_cell.col_span == 1

    # The merge must be emitted EXACTLY ONCE -- (0, 2) and (1, 0) are
    # spanned-into positions, never separate Cell entries duplicating the
    # same content (see TableData docstring, "never duplicate merged-cell
    # content").
    assert (0, 2) not in cells_by_pos
    assert (1, 0) not in cells_by_pos

    # But grid() DOES repeat the merged text into every visually-spanned
    # position, for display purposes.
    grid = table.grid()
    assert grid[0][1] == "Revenue"
    assert grid[0][2] == "Revenue"
    assert grid[0][0] == "Region"
    assert grid[1][0] == "Region"

    # Unmerged data cells are unaffected.
    assert cells_by_pos[(2, 0)].text == "North"
    assert cells_by_pos[(2, 0)].row_span == 1
    assert cells_by_pos[(2, 0)].col_span == 1

    # Every cell has a stable, non-empty object_id (Phase 4 provenance).
    assert all(c.object_id for c in table.cells)
    assert len({c.object_id for c in table.cells}) == len(table.cells)  # all unique


def test_parse_native_pdf_table_without_merges_reports_no_merged_cells(tmp_path: Path):
    f = tmp_path / "report.pdf"
    _build_pdf(f)
    doc = parse_pdf(f)
    table = [b for b in doc.blocks if b.block_type == BlockType.TABLE][0].table
    assert table.has_merged_cells is False
    assert all(c.row_span == 1 and c.col_span == 1 for c in table.cells)