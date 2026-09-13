"""Structured document representation.

Every parser/OCR path (native PDF, DOCX, TXT/MD, scanned PDF, PNG/JPG)
converts its input into this same shape before anything downstream
(chunking, embedding, citation) ever touches it. This is what lets the
ingestion pipeline preserve reading order, page numbers, headings, lists
and — critically — tables as *structured* data instead of flattening
everything into one plain-text blob (see docs/ocr.md and docs/tables.md).

Design intent:

- A document is a sequence of `Block`s in reading order. Each block knows
  what page it came from and, where available, its bounding box.
- A table is never silently flattened into paragraph text. It stays a
  `Block` of type "table" carrying real rows/cells (`TableData`), plus a
  Markdown rendering (`text`) used for embedding and for showing the
  table to the LLM. Both representations are kept side by side.
- `source` + `confidence` record whether a block came from native
  extraction (confidence 1.0, exact) or OCR (confidence from the OCR
  engine, < 1.0), so low-confidence extractions can be surfaced to the
  user instead of being presented as ground truth (see docs/ocr.md,
  "Never claim perfect preservation").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


def _stable_object_id(*parts: str) -> str:
    """A deterministic, content-derived ID for a sub-document object (a
    table cell, a figure, ...) that stays the same across re-ingestion of
    an unchanged file — the same idempotency principle chunking.py's
    `_content_hash` uses for chunk IDs, applied one level finer-grained.
    16 hex chars (64 bits) — short enough to be a readable `object_id`,
    long enough that an accidental collision within one document is not a
    realistic concern.
    """
    joined = "::".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"     # Phase 4: a FIGURE further classified as a chart (bar/line/pie/...)
    VISUAL = "visual"    # Phase 4: a visual object that is neither a plain figure nor a chart


class VisualObjectType(str, Enum):
    """What kind of visual a `VisualObject` is (Phase 4 spec section 8).
    Conservative by design (spec section 9): `UNKNOWN_VISUAL` is a valid,
    expected outcome whenever classification confidence is low — never
    a confident-but-wrong guess. See `app.documents.figures.classifier`.
    """

    FIGURE = "figure"           # generic figure, not further classified
    PHOTO = "photo"
    DIAGRAM = "diagram"
    ILLUSTRATION = "illustration"
    SCREENSHOT = "screenshot"
    CHART = "chart"
    UNKNOWN_VISUAL = "unknown_visual"


class ChartType(str, Enum):
    """Sub-classification for a `VisualObject` whose `object_type` is
    `VisualObjectType.CHART` (Phase 4 spec section 11). `UNKNOWN_CHART`
    is the honest default — see `app.documents.figures.chart_detector`.
    """

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    UNKNOWN_CHART = "unknown_chart"


class ExtractionSource(str, Enum):
    NATIVE = "native"       # extracted directly from the file's own text layer
    OCR = "ocr"              # extracted by running OCR over a rendered image


class CoordinateSpace(str, Enum):
    """Which pixel/point grid a `BoundingBox` is measured in (Phase 4
    Issues 2/3). A bbox is meaningless for source-viewer highlighting,
    citation click-through, or a page overlay without knowing this — a
    box drawn using PDF-point coordinates on a pixel-space image (or vice
    versa) lands in the wrong place, and a box from a rotated/cropped/
    resized OCR-preprocessed image is wrong on ANY canonical image unless
    it's been mapped back first (see `app.documents.visual.coordinates`).
    """

    PDF_POINTS = "pdf_points"    # native PDF page coordinate system (pdfplumber): points, origin top-left
    IMAGE_PIXELS = "image_pixels"  # pixel coordinates of ONE specific, named image (see BoundingBox.image_ref)
    NORMALIZED = "normalized"     # [0,1] x [0,1] fraction of page/image width/height
    UNSPECIFIED = "unspecified"   # pre-Phase-4 bbox with no recorded space; do not trust for pixel-exact use


@dataclass(frozen=True)
class BoundingBox:
    """Pixel/point coordinates on a rendered page image, top-left origin.

    `coordinate_space` (Phase 4) says which grid `x0..y1` are measured
    against — see `CoordinateSpace`. `image_ref` disambiguates WHICH
    image when `coordinate_space == IMAGE_PIXELS` (a page can have
    several: the rendered-at-DPI page image, and the further-processed
    OCR image, are different pixel grids) — `None` when the space is
    `PDF_POINTS`/`NORMALIZED` (page-level, not tied to one raster) or for
    a pre-Phase-4 `UNSPECIFIED` box.

    By the time a `Block`/`Cell`/`FigureData` bbox reaches
    `ExtractedDocument` (i.e. leaves `app/documents/parsers/*.py`), it
    MUST be in the page's canonical space — `PDF_POINTS` for a PDF page,
    `IMAGE_PIXELS` (of the ORIGINAL upload) for a standalone image — never
    left in OCR-preprocessed-image pixel space. See
    `app.documents.visual.coordinates.CoordinateTransform`.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace = CoordinateSpace.UNSPECIFIED
    image_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "coordinate_space": self.coordinate_space.value,
            "image_ref": self.image_ref,
        }


@dataclass(frozen=True)
class Cell:
    """One cell of a table, with real geometry when available (Phase 4).

    Produced by `app.documents.tables.reconstruct.cells_from_pdfplumber_table`
    for native PDF tables (pdfplumber gives real per-cell bounding boxes
    — see that function's docstring for exactly what is and isn't a
    heuristic), and by `detect_tables` (same module) for OCR-reconstructed
    tables — OCR cells carry a real bbox (from the OCR word boxes that
    made up the cell) and real per-cell confidence (mean OCR word
    confidence), with a best-effort `col_span` when word boxes clearly
    overlap more than one detected column. `row_span` for an OCR cell is
    always 1: OCR word boxes carry no reliable signal for vertical
    (row) merges, so — per this module's "do not fabricate geometry"
    rule — that dimension is left undetected rather than guessed, even
    though column merges are.

    A merged cell is represented ONCE, at its anchor (top-left) position,
    with `row_span`/`col_span` > 1 — never as separate `Cell` entries
    duplicating the same content into every position it visually covers.
    Cell.text for a spanned position lives only on the anchor cell.
    """

    text: str
    row: int
    column: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    bbox: BoundingBox | None = None
    confidence: float = 1.0
    object_id: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "row": self.row,
            "column": self.column,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "is_header": self.is_header,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "object_id": self.object_id,
        }


@dataclass(frozen=True)
class TableData:
    """Structured table content: a grid of cell strings.

    `n_rows`/`n_cols` describe the logical grid size. Merged cells are
    represented by repeating the same text in every cell the merge spans
    where that can be detected (native PDF/DOCX tables via pdfplumber /
    python-docx merge info); for OCR-reconstructed tables, merges are best
    -effort and `confidence` should be treated as the reliability signal,
    not the grid shape itself (see docs/tables.md, "Irregular and merged
    cells").

    `cells` (Phase 4) is the cell-accurate structure ALONGSIDE `rows`, not
    a replacement for it: `rows` is always the raw extracted text grid
    (what `to_markdown()`/embedding use, and what every pre-Phase-4
    consumer of this class already expects — unchanged for
    backward-compatibility), while `cells` carries real per-cell bounding
    boxes, header/merge/confidence metadata, and a stable `object_id` for
    citation-level provenance ("this answer came from cell (0, 1) of this
    exact table"). `cells` is empty (`()`) whenever cell-level extraction
    wasn't attempted or didn't apply (OCR-reconstructed tables, or a
    table extracted before Phase 4 existed) — always check `bool(cells)`
    before relying on it, never assume it's populated.
    """

    rows: tuple[tuple[str, ...], ...]
    cells: tuple[Cell, ...] = ()
    # Phase 4 spec section 7, "multi-page table continuation". All default
    # to "not part of any continuation" so a plain, single-page table
    # (the overwhelming common case) is unaffected. Populated by
    # `app.documents.tables.continuation.detect_continuations`, which runs
    # AFTER per-page/per-file table extraction and only ever links tables
    # that already independently exist — it never invents table content.
    table_id: str = ""
    continuation_group_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    continued_from_previous_page: bool = False
    continued_to_next_page: bool = False
    header_row_count: int = 1

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def has_merged_cells(self) -> bool:
        """True if any `cells` entry has row_span/col_span > 1. False
        (never "unknown") when `cells` is empty — a table with no
        cell-level structure makes no merge claim either way."""
        return any(c.row_span > 1 or c.col_span > 1 for c in self.cells)

    def grid(self) -> list[list[str]]:
        """A display grid the same shape as `rows`, but with each merged
        cell's text repeated into EVERY position it visually spans — the
        difference from `rows` itself, which (for a native pdfplumber
        table) leaves a merge's non-anchor positions blank because that's
        what the underlying extraction returns for them. Falls back to
        `rows` verbatim when `cells` is empty (nothing to expand).
        """
        if not self.cells:
            return [list(r) for r in self.rows]

        n_rows, n_cols = self.n_rows, self.n_cols
        out = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for cell in self.cells:
            for r in range(cell.row, min(cell.row + cell.row_span, n_rows)):
                for c in range(cell.column, min(cell.column + cell.col_span, n_cols)):
                    out[r][c] = cell.text
        return out

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        header, *body = self.rows
        width = self.n_cols
        header = list(header) + [""] * (width - len(header))
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for row in body:
            row = list(row) + [""] * (width - len(row))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {"rows": [list(r) for r in self.rows], "n_rows": self.n_rows, "n_cols": self.n_cols}
        if self.cells:
            d["cells"] = [c.to_dict() for c in self.cells]
            d["has_merged_cells"] = self.has_merged_cells
        if self.table_id:
            d["table_id"] = self.table_id
        if self.continuation_group_id:
            d.update(
                {
                    "continuation_group_id": self.continuation_group_id,
                    "page_start": self.page_start,
                    "page_end": self.page_end,
                    "continued_from_previous_page": self.continued_from_previous_page,
                    "continued_to_next_page": self.continued_to_next_page,
                }
            )
        return d


@dataclass(frozen=True)
class VisualObject:
    """Unified representation of a non-text/non-table document object —
    a figure, photo, diagram, illustration, screenshot, or chart (Phase 4
    spec section 13, "Unified visual object model").

    This is the ONE model every visual-extraction path
    (`app.documents.figures`) produces, so a `Block` of type FIGURE/
    CHART/VISUAL is never a "disconnected figure subsystem" (spec
    section 13) — it's always this same shape, attached to `Block.visual`
    exactly the way `Block.table` already attaches a `TableData`.

    `metadata` carries type-specific detail that doesn't apply to every
    visual (chart axis/legend/series data — see
    `app.documents.figures.chart_understanding`), instead of a wide
    dataclass with mostly-`None` chart-only fields on every plain photo.
    Every metadata VALUE this module writes is either a real, extracted
    piece of information or explicitly `None` with a corresponding
    `*_confidence` of 0.0 — never a fabricated placeholder (spec section
    12, "do not hallucinate values").
    """

    object_id: str
    object_type: VisualObjectType
    page_number: int | None
    bbox: BoundingBox | None
    confidence: float = 0.0
    source: ExtractionSource = ExtractionSource.NATIVE
    extraction_method: str = ""
    parent_object_id: str | None = None
    caption: str | None = None
    caption_bbox: BoundingBox | None = None
    caption_object_id: str | None = None
    caption_confidence: float = 0.0
    chart_type: ChartType | None = None   # only meaningful when object_type == CHART
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type.value,
            "page_number": self.page_number,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "source": self.source.value,
            "extraction_method": self.extraction_method,
            "parent_object_id": self.parent_object_id,
            "caption": self.caption,
            "caption_bbox": self.caption_bbox.to_dict() if self.caption_bbox else None,
            "caption_object_id": self.caption_object_id,
            "caption_confidence": self.caption_confidence,
            "chart_type": self.chart_type.value if self.chart_type else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Block:
    """One structural unit of a document, in reading order.

    Provenance (Phase 4 spec section 3, "unified provenance model"):
    `object_id` is a deterministic, content-derived ID (see
    `_stable_object_id`) assigned by
    `app.documents.provenance.assign_provenance` right after parsing —
    every parser produces `object_id=""` initially; nothing downstream
    should rely on an un-assigned block ever reaching chunking (
    `extract_document`, the shared entry point, always calls
    `assign_provenance` before returning). `parent_object_id` is `None`
    for a top-level block; a table `Cell` or a `VisualObject`'s caption
    already carries its OWN `object_id` and points back at its parent
    (the table's/figure's `object_id`) via `parent_object_id` on the
    owning `Block` — see `assign_provenance`'s docstring for exactly how
    IDs stay stable `parser -> block -> chunk -> retrieval -> citation`.
    """

    block_type: BlockType
    text: str                       # plain/markdown text used for embedding + display
    page_number: int | None = None  # 1-indexed; None for formats with no page concept (md/txt)
    order: int = 0                  # position within the document, for stable ordering
    bbox: BoundingBox | None = None
    heading_level: int | None = None  # 1 for H1/title, 2 for H2, ... (headings only)
    table: TableData | None = None    # populated only when block_type == TABLE
    visual: VisualObject | None = None  # populated only when block_type in {FIGURE, CHART, VISUAL}
    source: ExtractionSource = ExtractionSource.NATIVE
    confidence: float = 1.0
    object_id: str = ""              # Phase 4: deterministic ID, see class docstring
    parent_object_id: str | None = None
    extraction_method: str = ""      # e.g. "pdfplumber_native", "tesseract_ocr", "pdf_embedded_image"

    def to_dict(self) -> dict:
        d = {
            "block_type": self.block_type.value,
            "text": self.text,
            "page_number": self.page_number,
            "order": self.order,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "heading_level": self.heading_level,
            "source": self.source.value,
            "confidence": self.confidence,
            "object_id": self.object_id,
            "parent_object_id": self.parent_object_id,
            "extraction_method": self.extraction_method,
        }
        if self.table is not None:
            d["table"] = self.table.to_dict()
        if self.visual is not None:
            d["visual"] = self.visual.to_dict()
        return d


@dataclass(frozen=True)
class PageInfo:
    """`width`/`height` are always in the page's OWN canonical coordinate
    space (Phase 4 Issue 3): `PDF_POINTS` for a PDF page (matches
    `pdfplumber`'s `page.width`/`page.height` exactly, regardless of
    whether that page ended up native or OCR'd), `IMAGE_PIXELS` (of the
    original upload) for a standalone image. `coordinate_space` records
    which. Every `Block.bbox` on this page is guaranteed to be in this
    same space — see `BoundingBox`'s docstring and
    `app.documents.visual.coordinates.CoordinateTransform`.
    """

    page_number: int
    width: float | None = None
    height: float | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.UNSPECIFIED
    is_scanned: bool = False
    ocr_confidence: float | None = None  # mean OCR word confidence, when OCR ran on this page
    ocr_preprocessing: dict | None = None  # metadata from app.documents.visual.ocr_preprocessing (Phase 4)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "coordinate_space": self.coordinate_space.value,
            "is_scanned": self.is_scanned,
            "ocr_confidence": self.ocr_confidence,
            "ocr_preprocessing": self.ocr_preprocessing,
        }


@dataclass
class ExtractedDocument:
    """The full structured result of parsing one source file."""

    source_path: str
    file_type: str                    # extension, e.g. "pdf", "docx", "png"
    pages: list[PageInfo] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages) or (1 if self.blocks else 0)

    @property
    def has_scanned_pages(self) -> bool:
        return any(p.is_scanned for p in self.pages)

    @property
    def tables(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type == BlockType.TABLE]

    @property
    def figures(self) -> list[Block]:
        """FIGURE-typed blocks only — a generic figure not further
        classified as a chart. See `.charts`/`.visuals` for the other
        two Phase 4 visual block types; `.visual_blocks` for all three
        together.
        """
        return [b for b in self.blocks if b.block_type == BlockType.FIGURE]

    @property
    def charts(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type == BlockType.CHART]

    @property
    def visuals(self) -> list[Block]:
        """VISUAL-typed blocks only (photo/diagram/illustration/screenshot/
        unknown_visual — i.e. neither a plain FIGURE nor a CHART)."""
        return [b for b in self.blocks if b.block_type == BlockType.VISUAL]

    @property
    def visual_blocks(self) -> list[Block]:
        """Every block carrying a `VisualObject` (FIGURE + CHART + VISUAL
        combined) — the set `app.chunking`'s multimodal chunking and
        `app.services.ingestion_service`'s figure_count/chart_count
        bookkeeping both iterate over.
        """
        return [b for b in self.blocks if b.visual is not None]

    def plain_text(self) -> str:
        """Flattened text, used only for legacy consumers / debugging."""
        parts = []
        for b in self.blocks:
            parts.append(b.table.to_markdown() if b.table else b.text)
        return "\n\n".join(p for p in parts if p)