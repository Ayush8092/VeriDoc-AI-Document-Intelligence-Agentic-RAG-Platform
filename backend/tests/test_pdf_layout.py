"""Tests for app.documents.parsers.pdf_layout — font-size/boldness-based
PDF heading detection (Phase 4 spec section 10) and the heuristic
LayoutDetector (section 9).

Builds PDFs with EXPLICIT font sizes/weights via reportlab's
ParagraphStyle (rather than relying on the default stylesheet's Title/
Heading1 names, whose exact point sizes aren't part of this test's
contract) so each test controls precisely the typographic signal being
tested.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from app.documents.parsers.pdf_layout import LayoutDetector, RegionType, extract_layout_paragraphs


def _build_pdf(path: Path, flowables) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate

    doc = SimpleDocTemplate(str(path), pagesize=letter)
    doc.build(flowables)


def _style(name, size, bold=False, alignment=0):
    from reportlab.lib.styles import ParagraphStyle

    return ParagraphStyle(
        name, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, leading=size * 1.2, alignment=alignment
    )


def _paragraphs(path: Path):
    with pdfplumber.open(str(path)) as pdf:
        return extract_layout_paragraphs(pdf.pages[0])


def test_large_title_detected_as_h1(tmp_path: Path):
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph("Annual Report 2025", _style("title", 22, bold=True)),
            Spacer(1, 12),
            Paragraph(
                "This is ordinary body text at a normal reading size, long enough to look like a real paragraph "
                "rather than a heading, describing the contents of the report in plain prose.",
                _style("body", 10),
            ),
        ],
    )

    cps = _paragraphs(f)
    headings = [cp for cp in cps if cp.kind == "heading"]
    assert any(cp.heading_level == 1 and "Annual Report 2025" in cp.text for cp in headings)


def test_section_heading_detected_as_h2(tmp_path: Path):
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph(
                "Ordinary body paragraph establishing the page's normal reading size for comparison purposes here.",
                _style("body", 10),
            ),
            Spacer(1, 10),
            Paragraph("Revenue Overview", _style("h2", 14, bold=True)),
            Spacer(1, 10),
            Paragraph(
                "More ordinary body text following the section heading, describing revenue in plain prose sentences.",
                _style("body", 10),
            ),
        ],
    )

    cps = _paragraphs(f)
    headings = {cp.text: cp for cp in cps if cp.kind == "heading"}
    assert "Revenue Overview" in headings
    assert headings["Revenue Overview"].heading_level in (2, 3)


def test_subsection_heading_detected_at_lower_level_than_section(tmp_path: Path):
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph("Main Section Title", _style("h1", 20, bold=True)),
            Spacer(1, 10),
            Paragraph("Subsection Detail", _style("h3", 12, bold=True)),
            Spacer(1, 10),
            Paragraph(
                "Ordinary body paragraph at the normal reading size for this document, several words long.",
                _style("body", 10),
            ),
        ],
    )

    cps = _paragraphs(f)
    headings = {cp.text: cp for cp in cps if cp.kind == "heading"}
    assert headings["Main Section Title"].heading_level == 1
    assert headings["Subsection Detail"].heading_level > headings["Main Section Title"].heading_level


def test_bold_long_paragraph_is_not_classified_as_heading(tmp_path: Path):
    """The explicit false-positive guard: a BOLD paragraph at normal body
    size, several sentences long, must not be misclassified as a
    heading just because it's bold."""
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph(
                "This entire paragraph is bold for emphasis, but it is a normal-size, multi-sentence "
                "paragraph of real prose that happens to use a bold font weight. It should still be "
                "classified as an ordinary paragraph, not a heading, because bold alone at body text "
                "size is not sufficient evidence of a structural heading.",
                _style("bold_body", 10, bold=True),
            ),
        ],
    )

    cps = _paragraphs(f)
    assert len(cps) == 1
    assert cps[0].kind == "paragraph"


def test_numbered_heading_detected(tmp_path: Path):
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph(
                "Ordinary body paragraph establishing the page's normal reading size for this document.",
                _style("body", 10),
            ),
            Spacer(1, 10),
            Paragraph("3.2 Termination Clauses", _style("numbered", 11)),
            Spacer(1, 10),
            Paragraph(
                "Ordinary body paragraph following the numbered heading, in plain prose sentences again.",
                _style("body", 10),
            ),
        ],
    )

    cps = _paragraphs(f)
    headings = {cp.text: cp for cp in cps if cp.kind == "heading"}
    assert "3.2 Termination Clauses" in headings


def test_all_caps_short_line_boosts_heading_confidence(tmp_path: Path):
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph(
                "Ordinary body paragraph establishing the page's normal reading size for this document here.",
                _style("body", 10),
            ),
            Spacer(1, 10),
            Paragraph("IMPORTANT NOTICE", _style("caps", 13, bold=True)),
            Spacer(1, 10),
            Paragraph(
                "Ordinary body paragraph following the all-caps heading, written in plain prose sentences.",
                _style("body", 10),
            ),
        ],
    )

    cps = _paragraphs(f)
    headings = {cp.text: cp for cp in cps if cp.kind == "heading"}
    assert "IMPORTANT NOTICE" in headings
    assert headings["IMPORTANT NOTICE"].confidence >= 0.5


def test_false_positive_all_caps_long_disclaimer_not_a_heading(tmp_path: Path):
    """An ALL-CAPS block at BODY size but many words long (a legal
    disclaimer, a common real-world pattern) must not be classified as a
    heading purely from the all-caps signal — length and size both
    matter, not just capitalization."""
    from reportlab.platypus import Paragraph

    f = tmp_path / "doc.pdf"
    long_disclaimer = (
        "THIS DOCUMENT IS PROVIDED FOR SAMPLE PURPOSES ONLY AND DOES NOT CONSTITUTE LEGAL "
        "ADVICE OF ANY KIND WHATSOEVER AND SHOULD NOT BE RELIED UPON FOR ANY BUSINESS DECISION."
    )
    _build_pdf(f, [Paragraph(long_disclaimer, _style("disclaimer", 10))])

    cps = _paragraphs(f)
    assert len(cps) == 1
    assert cps[0].kind == "paragraph"


def test_layout_detector_returns_typed_regions_with_confidence(tmp_path: Path):
    from reportlab.platypus import Paragraph, Spacer

    f = tmp_path / "doc.pdf"
    _build_pdf(
        f,
        [
            Paragraph("Report Title", _style("title", 20, bold=True)),
            Spacer(1, 10),
            Paragraph("Body text describing the report in ordinary prose sentences here.", _style("body", 10)),
        ],
    )

    with pdfplumber.open(str(f)) as pdf:
        regions = LayoutDetector().detect(pdf.pages[0])

    assert any(r.region_type in (RegionType.TITLE, RegionType.HEADING) for r in regions)
    assert all(0.0 <= r.confidence <= 1.0 for r in regions)
    assert all(r.bbox is not None for r in regions)


def test_layout_detector_detect_multi_page_is_an_explicit_extension_point(tmp_path: Path):
    """Cross-page HEADER/FOOTER/SIDEBAR/COLUMN detection is not
    implemented — it must fail loudly (NotImplementedError), never
    silently return an empty/fabricated result."""
    import pytest

    with pytest.raises(NotImplementedError):
        LayoutDetector().detect_multi_page([])
