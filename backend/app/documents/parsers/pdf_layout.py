"""PDF layout analysis: heading detection + a provider-agnostic
`LayoutDetector` abstraction (Phase 4, spec sections 9 and 10).

Native PDFs have no equivalent of DOCX's "Heading 1" paragraph style —
pdfplumber's plain `extract_text()` (what `pdf_parser.py` used before this
module existed) returns an undifferentiated block of text with every
paragraph the same `BlockType.PARAGRAPH`. This module extracts words WITH
their font metadata (`page.extract_words(extra_attrs=["size", "fontname"])`)
so paragraphs can be classified using real typographic signals — font
size relative to the page's own body-text size, boldness, ALL-CAPS,
numbered-heading patterns, and paragraph length — rather than treated as
uniform prose.

Deliberately NOT a trained layout model (per spec section 36, "do not
overengineer" / "do not require a huge GPU model", and section 9: "A
strong baseline using PDF geometry / font metadata / heuristics is
acceptable"). `LayoutDetector` is an abstract interface specifically so a
future stronger model (LayoutLM, a vision transformer, ...) could be
swapped in later without changing any caller.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from enum import Enum

_BULLET_RE = re.compile(r"^[\u2022\u25aa\u25cf\-\*•▪●]\s+")
_NUMBERED_LIST_RE = re.compile(r"^(\d{1,3}|[a-zA-Z])[\.\)]\s+")
_NUMBERED_HEADING_RE = re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\s+\S")
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

# Ratio of a paragraph's dominant font size to the page's body-text size,
# above which it's a heading candidate at all. Below this, typography
# alone gives no signal and only the numbered-heading pattern can still
# promote it.
_HEADING_MIN_RATIO = 1.08
_H1_RATIO = 1.45
_H2_RATIO = 1.2
# A heading candidate this long (words) is almost certainly a real
# sentence/paragraph that merely happens to be in a larger/bold font
# (e.g. a pull quote, a bold lead-in) — not a structural heading. This is
# the guard against "classify every bold sentence as a heading" (spec
# section 10).
_MAX_HEADING_WORDS = 14


class RegionType(str, Enum):
    """Layout region types `LayoutDetector.detect` can produce (spec
    section 9). Not every region type is populated by every page/format —
    HEADER/FOOTER/SIDEBAR/COLUMN detection specifically require repeated-
    position evidence across MULTIPLE pages (see `LayoutDetector.detect`'s
    docstring), so a single-page call can only ever return the others.
    """

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    COLUMN = "column"


@dataclass(frozen=True)
class LayoutRegion:
    """One detected region of a page, with the typographic/geometric
    evidence that produced it (`confidence`) — never a bare label with no
    justification. `bbox` is `None` when the source (e.g. plain .txt) has
    no geometry at all."""

    region_type: RegionType
    text: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    page_number: int | None = None
    reading_order: int = 0


@dataclass(frozen=True)
class _Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    fontname: str

    @property
    def is_bold(self) -> bool:
        return "bold" in self.fontname.lower()


@dataclass(frozen=True)
class _Line:
    words: tuple[_Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def dominant_size(self) -> float:
        return statistics.median(w.size for w in self.words)

    @property
    def bold_fraction(self) -> float:
        return sum(1 for w in self.words if w.is_bold) / len(self.words)


@dataclass(frozen=True)
class _Paragraph:
    lines: tuple[_Line, ...]

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(l.x0 for l in self.lines),
            min(l.top for l in self.lines),
            max(l.x1 for l in self.lines),
            max(l.bottom for l in self.lines),
        )

    @property
    def dominant_size(self) -> float:
        sizes = [w.size for line in self.lines for w in line.words]
        return statistics.median(sizes) if sizes else 0.0

    @property
    def bold_fraction(self) -> float:
        words = [w for line in self.lines for w in line.words]
        return sum(1 for w in words if w.is_bold) / len(words) if words else 0.0

    @property
    def word_count(self) -> int:
        return sum(len(line.words) for line in self.lines)


def _words_from_page(page) -> list[_Word]:
    raw = page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=False)
    return [
        _Word(
            text=w["text"],
            x0=w["x0"],
            x1=w["x1"],
            top=w["top"],
            bottom=w["bottom"],
            size=float(w.get("size") or 0.0),
            fontname=str(w.get("fontname") or ""),
        )
        for w in raw
    ]


def _group_lines(words: list[_Word], y_tolerance: float = 3.0) -> list[_Line]:
    """Group words into lines by vertical (`top`) proximity, then sort
    each line left-to-right. Words are first sorted by `top` so lines
    come out in reading order (top-to-bottom)."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w.top / y_tolerance), w.x0))
    lines: list[list[_Word]] = []
    for w in ordered:
        if lines and abs(w.top - lines[-1][-1].top) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [_Line(words=tuple(sorted(ws, key=lambda w: w.x0))) for ws in lines]


def _group_paragraphs(lines: list[_Line]) -> list[_Paragraph]:
    """Group consecutive lines into paragraphs by vertical gap: a gap
    noticeably larger than the typical line-to-line gap on the page (or
    a change in dominant font size) starts a new paragraph."""
    if not lines:
        return []
    
    gaps = [
        lines[i + 1].top - lines[i].bottom
        for i in range(len(lines) - 1)
]
    positive_gaps = [g for g in gaps if g > 0]

    if positive_gaps:
        # Large gaps can represent paragraph breaks, captions, or space
        # occupied by non-text objects such as figures/tables. They must not
        # influence the estimate of normal line spacing.
        sorted_gaps = sorted(positive_gaps)

        # Estimate normal line spacing from the lower half of the observed
        # gaps. This keeps unusually large layout gaps from inflating the
        # paragraph-separation threshold, especially on short pages where
        # there may be only a few text lines.
        typical_candidates = sorted_gaps[: max(1, (len(sorted_gaps) + 1) // 2)]
        typical_gap = statistics.median(typical_candidates)
    else:
        typical_gap = 2.0

    gap_threshold = max(typical_gap * 1.8, typical_gap + 4.0)

    paragraphs: list[list[_Line]] = [[lines[0]]]
    for i in range(1, len(lines)):
        gap = lines[i].top - lines[i - 1].bottom
        size_changed = abs(lines[i].dominant_size - lines[i - 1].dominant_size) > 0.5
        if gap > gap_threshold or size_changed:
            paragraphs.append([lines[i]])
        else:
            paragraphs[-1].append(lines[i])
    return [_Paragraph(lines=tuple(p)) for p in paragraphs]


def _body_size(paragraphs: list[_Paragraph]) -> float:
    """The page's own body-text font size: the mode (most common) size
    across every word, weighted by word count so a one-line title doesn't
    skew it. Falls back to the smallest paragraph's size if there's no
    clear mode (e.g. a page that's ALL headings)."""
    sizes = [round(w.size, 1) for p in paragraphs for line in p.lines for w in line.words]
    if not sizes:
        return 0.0
    counts: dict[float, int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _classify_paragraph(paragraph: _Paragraph, body_size: float) -> tuple[str, int | None, float]:
    """Returns (kind, heading_level, confidence) where kind is
    "heading" | "list_item" | "paragraph".

    Confidence is a simple, explainable evidence count normalized to
    [0, 1] — not a claim of calibrated probability (spec section 10:
    "Detect H1/H2/H3 with confidence").
    """
    text = paragraph.text.strip()
    if _BULLET_RE.match(text) or (_NUMBERED_LIST_RE.match(text) and paragraph.word_count > 4):
        return "list_item", None, 0.9

    if body_size <= 0:
        return "paragraph", None, 1.0

    ratio = paragraph.dominant_size / body_size
    is_bold = paragraph.bold_fraction >= 0.6
    is_short = paragraph.word_count <= _MAX_HEADING_WORDS
    is_all_caps = text.isupper() and len(text) >= 3
    ends_mid_sentence = _SENTENCE_END_RE.search(text) is None
    is_numbered_heading = bool(_NUMBERED_HEADING_RE.match(text)) and is_short

    evidence = 0
    if ratio >= _HEADING_MIN_RATIO:
        evidence += 1
    if is_bold:
        evidence += 1
    if is_all_caps:
        evidence += 1
    if is_numbered_heading:
        evidence += 1
    if is_short:
        evidence += 1
    if ends_mid_sentence:
        evidence += 1  # headings don't end with terminal punctuation

    # The actual gate: typography or an explicit numbered-heading pattern
    # MUST be present, and it must be short — a long, bold, all-caps
    # PARAGRAPH (e.g. a disclaimer block) is still not a heading just
    # because it clears the word-count bar on every OTHER signal (spec
    # section 10, "do not classify every bold sentence as a heading").
    has_typographic_or_pattern_signal = ratio >= _HEADING_MIN_RATIO or is_numbered_heading
    if not (has_typographic_or_pattern_signal and is_short):
        return "paragraph", None, 1.0 - min(evidence / 6.0, 0.4)

    if ratio >= _H1_RATIO:
        level = 1
    elif ratio >= _H2_RATIO:
        level = 2
    elif ratio >= _HEADING_MIN_RATIO or is_numbered_heading:
        level = 3
    else:
        level = 3

    confidence = round(min(0.5 + evidence / 8.0, 0.98), 2)
    return "heading", level, confidence


@dataclass(frozen=True)
class ClassifiedParagraph:
    text: str
    kind: str  # "heading" | "list_item" | "paragraph"
    heading_level: int | None
    confidence: float
    bbox: tuple[float, float, float, float]


def extract_layout_paragraphs(page) -> list[ClassifiedParagraph]:
    """The function `pdf_parser.py`'s native-page path actually calls:
    words -> lines -> paragraphs -> (heading | list_item | paragraph)
    classification, using ONLY this page's own font metadata (a document-
    wide body-size baseline would be more robust across multi-page PDFs
    with a consistent style, but requires a second pass across all pages
    before any block can be classified — left as a documented possible
    improvement, not implemented here to keep this a single-page-at-a-time
    function matching how `pdf_parser.py` already processes pages).
    """
    words = _words_from_page(page)
    lines = _group_lines(words)
    print("\n--- PDF LAYOUT DEBUG ---")
    paragraphs = _group_paragraphs(lines)
    body_size = _body_size(paragraphs)

    out = []
    for p in paragraphs:
        text = p.text.strip()
        if not text:
            continue
        kind, level, confidence = _classify_paragraph(p, body_size)
        out.append(ClassifiedParagraph(text=text, kind=kind, heading_level=level, confidence=confidence, bbox=p.bbox))
    return out


class LayoutDetector:
    """Provider-agnostic layout-region detection (spec section 9).

    `detect(page)` is deliberately narrow — one pdfplumber `Page` in, one
    list of regions out — so a future implementation (a real ML layout
    model, or a different backend for OCR'd pages) can be swapped in
    without any caller needing to change. This default implementation is
    the heuristic, font/geometry-based classifier above; it does not
    require a GPU or any model download.

    HEADER/FOOTER/SIDEBAR/COLUMN are NOT produced by this implementation:
    reliably telling a running header from a genuine top-of-page heading,
    or a genuine sidebar from a narrow single-column page, needs evidence
    from MULTIPLE pages (a line that repeats at the same position on
    every page is a header/footer; a page whose text splits into two
    non-overlapping x-ranges across most of its height is multi-column) —
    signal a single-page call structurally cannot have. This is stated
    here rather than silently returning nothing that looks like a
    considered decision: `detect_multi_page` below is the (currently
    unimplemented, extension-point) place that evidence would be
    aggregated.
    """

    def detect(self, page) -> list[LayoutRegion]:
        regions: list[LayoutRegion] = []
        order = 0

        found_tables = []
        try:
            found_tables = page.find_tables()
        except Exception:
            found_tables = []
        non_table_page = page
        for t in found_tables:
            try:
                non_table_page = non_table_page.outside_bbox(t.bbox)
            except Exception:
                pass

        for cp in extract_layout_paragraphs(non_table_page):
            if cp.kind == "heading":
                region_type = RegionType.TITLE if cp.heading_level == 1 and order == 0 else RegionType.HEADING
            elif cp.kind == "list_item":
                region_type = RegionType.LIST
            else:
                region_type = RegionType.PARAGRAPH
            regions.append(
                LayoutRegion(
                    region_type=region_type,
                    text=cp.text,
                    confidence=cp.confidence,
                    bbox=cp.bbox,
                    page_number=getattr(page, "page_number", None),
                    reading_order=order,
                )
            )
            order += 1

        for t in found_tables:
            regions.append(
                LayoutRegion(
                    region_type=RegionType.TABLE,
                    text="",
                    confidence=0.85,
                    bbox=tuple(t.bbox),
                    page_number=getattr(page, "page_number", None),
                    reading_order=order,
                )
            )
            order += 1

        regions.sort(key=lambda r: (r.bbox[1] if r.bbox else 0, r.bbox[0] if r.bbox else 0))
        return regions

    def detect_multi_page(self, pages) -> list[LayoutRegion]:  # pragma: no cover - documented extension point
        """Cross-page HEADER/FOOTER/SIDEBAR/COLUMN detection — NOT
        implemented (see class docstring). Raises so a caller can't
        silently get an empty, misleadingly "no headers found" result;
        this makes the gap explicit rather than fabricating a null
        answer that looks like a considered "there are none".
        """
        raise NotImplementedError(
            "Cross-page layout regions (HEADER/FOOTER/SIDEBAR/COLUMN) require "
            "repeated-position evidence across multiple pages and are not yet "
            "implemented — see LayoutDetector's docstring."
        )
