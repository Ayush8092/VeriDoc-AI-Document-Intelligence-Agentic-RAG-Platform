"""Parse `.pdf` files: native text/table extraction, OCR fallback per page.

Pipeline per page (docs/ocr.md, docs/tables.md):

1. Try native extraction with pdfplumber: text outside detected table
   regions, plus structured tables via `page.find_tables()`.
2. If the page lacks enough usable native content — combined paragraph
   text AND any detected table's own cell text (Phase 4 Issue 4 fix: a
   lone, possibly-spurious detected table no longer blocks OCR by
   itself) — below `min_native_chars`, discard the native result for
   that page, render it to an image, and run OCR + heuristic table
   reconstruction instead.

A PDF can mix native and scanned pages (e.g. a signature page); each
page's blocks record their own `source` (native/ocr) and `confidence`
accordingly — see `ExtractedDocument.has_scanned_pages`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from app.core.config import Settings
from app.documents.figures.extract import associate_all_captions, extract_pdf_page_visuals
from app.documents.models import (
    Block,
    BlockType,
    BoundingBox,
    CoordinateSpace,
    ExtractedDocument,
    ExtractionSource,
    PageInfo,
    TableData,
    _stable_object_id,
)
from app.documents.ocr.engine import group_lines
from app.documents.ocr.scanned_pdf import render_pdf_pages
from app.documents.parsers.pdf_layout import extract_layout_paragraphs
from app.documents.tables.continuation import detect_continuations
from app.documents.tables.reconstruct import cells_from_pdfplumber_table, detect_tables, header_row_count_for


def _native_content_chars(native_text: str, native_blocks: list[Block]) -> int:
    """Total usable native content on a page: extracted paragraph text
    PLUS actual table cell text (Phase 4 Issue 4).

    Counting only `native_text` let a page with little/no real content
    but ONE detected table object skip OCR entirely (the old condition
    was `is_scanned_page(native_text, ...) and not any table`) — a
    pdfplumber-detected table (e.g. from a couple of stray ruled lines on
    an otherwise-scanned page) could have near-empty cells and still
    block the OCR fallback. Counting the table's own cell text means a
    trivial/empty table no longer masks a page that actually needs OCR,
    while a table with substantial real text still correctly counts as
    "this page has usable native content".
    """
    total = len(native_text)
    for block in native_blocks:
        if block.block_type != BlockType.TABLE or block.table is None:
            continue
        if block.table.cells:
            total += sum(len(c.text) for c in block.table.cells)
        else:
            total += sum(len(cell) for row in block.table.rows for cell in row)
    return total


def _parse_native_page(page, page_number: int, order: int, source_path: str) -> tuple[list[Block], int, str]:
    blocks: list[Block] = []
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

    text = (non_table_page.extract_text() or "").strip()

    # Phase 4: font-size/boldness-based heading + list-item detection
    # (app/documents/parsers/pdf_layout.py), replacing a plain "every
    # paragraph is BlockType.PARAGRAPH" split. Falls back to that same
    # plain-paragraph behavior if word-level font extraction raises for
    # any reason (e.g. a malformed/encrypted PDF's font tables) — a
    # heading-detection failure must never block text extraction
    # entirely.
    try:
        classified = extract_layout_paragraphs(non_table_page)
    except Exception:
        classified = None

    if classified:
        for cp in classified:
            block_type = (
                BlockType.HEADING
                if cp.kind == "heading"
                else BlockType.LIST_ITEM
                if cp.kind == "list_item"
                else BlockType.PARAGRAPH
            )
            x0, top, x1, bottom = cp.bbox
            blocks.append(
                Block(
                    block_type=block_type,
                    text=cp.text,
                    page_number=page_number,
                    order=order,
                    bbox=BoundingBox(x0=x0, y0=top, x1=x1, y1=bottom, coordinate_space=CoordinateSpace.PDF_POINTS),
                    heading_level=cp.heading_level,
                    source=ExtractionSource.NATIVE,
                    confidence=cp.confidence,
                )
            )
            order += 1
    else:
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            blocks.append(
                Block(block_type=BlockType.PARAGRAPH, text=para, page_number=page_number, order=order, source=ExtractionSource.NATIVE)
            )
            order += 1

    for table_index, t in enumerate(found_tables):
        try:
            grid = t.extract()
        except Exception:
            continue
        if not grid:
            continue
        rows = tuple(tuple((c or "").strip() for c in row) for row in grid)
        # Real per-cell bounding boxes + merge detection from pdfplumber's
        # own geometry (Phase 4) — see cells_from_pdfplumber_table's
        # docstring for exactly what is/isn't a heuristic here.
        object_id_prefix = f"{source_path}::p{page_number}::table{table_index}"
        cells = cells_from_pdfplumber_table(t, object_id_prefix)
        table_data = TableData(rows=rows, cells=cells, header_row_count=header_row_count_for(cells))
        x0, top, x1, bottom = t.bbox

        blocks.append(
            Block(
                block_type=BlockType.TABLE,
                text=table_data.to_markdown(),
                page_number=page_number,
                order=order,
                table=table_data,
                bbox=BoundingBox(x0=x0, y0=top, x1=x1, y1=bottom, coordinate_space=CoordinateSpace.PDF_POINTS),
                source=ExtractionSource.NATIVE,
            )
        )
        order += 1

    return blocks, order, text


def _parse_scanned_page(
    image,
    page_number: int,
    order: int,
    language: str,
    confidence_floor: float,
    source_path: str,
    preprocess: bool = True,
    ocr_dpi: int = 200,
) -> tuple[list[Block], int, float, dict | None]:
    # Phase 4 Issue 1 fix: this used to call `run_ocr` directly, bypassing
    # OCR preprocessing entirely for every scanned PDF page — the exact
    # same rendered page saved as a standalone PNG would have gotten
    # deskew/denoise/contrast/orientation correction it never got here.
    # `preprocess_and_ocr` is the one shared implementation; see its
    # docstring ("Shared by both ingestion paths") and
    # `app/documents/parsers/image_parser.py`'s identical call site.
    from app.documents.visual.coordinates import CoordinateTransform
    from app.documents.visual.ocr_preprocessing import preprocess_and_ocr

    blocks: list[Block] = []
    rendered_size = (image.width, image.height)
    words, ocr_image, preprocessing_metadata = preprocess_and_ocr(
        image, language=language, confidence_floor=confidence_floor, preprocess=preprocess
    )
    if not words:
        return blocks, order, 0.0, preprocessing_metadata

    # Phase 4 Issue 2/3 fix: `words`' bboxes are in `ocr_image`'s pixel
    # grid (post orientation/deskew/crop/resize), NOT the rendered page's
    # -- and definitely not PDF points, which is what every native-page
    # bbox on this same document uses (`_parse_native_page`, `page.width`/
    # `page.height`). `transform` maps every OCR bbox below back through
    # exactly what preprocessing did, then rendered-pixels -> PDF points
    # via the DPI this page was rendered at, so scanned and native pages'
    # blocks end up in ONE consistent coordinate space (see
    # `app.documents.visual.coordinates.CoordinateTransform`'s docstring).
    transform = CoordinateTransform.from_preprocessing(
        rendered_size=rendered_size,
        preprocessing_metadata=preprocessing_metadata,
        dpi=ocr_dpi,
        output_space=CoordinateSpace.PDF_POINTS,
    )

    lines = group_lines(words)
    table_regions = detect_tables(lines, object_id_prefix=f"{source_path}::p{page_number}")
    consumed = set()
    for table_data, confidence, (start, end) in table_regions:
        consumed.update(range(start, end + 1))
        region_words = [w for k in range(start, end + 1) for w in lines[k].words]
        x0 = min(w.left for w in region_words)
        y0 = min(w.top for w in region_words)
        x1 = max(w.left + w.width for w in region_words)
        y1 = max(w.top + w.height for w in region_words)
        bbox = transform.bbox_to_original(BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))
        # Phase 4: cell bboxes from detect_tables are still in the raw
        # ocr_image pixel grid (pre-transform) — map each one through
        # the same transform as the table's own bbox above, so cell
        # geometry lands in the document's canonical PDF_POINTS space.
        import dataclasses as _dc

        table_data = _dc.replace(
            table_data,
            cells=tuple(
                _dc.replace(c, bbox=transform.bbox_to_original(c.bbox)) if c.bbox else c
                for c in table_data.cells
            ),
        )
        blocks.append(
            Block(
                block_type=BlockType.TABLE,
                text=table_data.to_markdown(),
                page_number=page_number,
                order=order,
                table=table_data,
                bbox=bbox,
                source=ExtractionSource.OCR,
                confidence=confidence,
            )
        )
        order += 1

    paragraph_lines = []
    for idx, line in enumerate(lines):
        if idx in consumed or not line.text.strip():
            if paragraph_lines:
                para_text = " ".join(paragraph_lines)
                blocks.append(
                    Block(block_type=BlockType.PARAGRAPH, text=para_text, page_number=page_number, order=order, source=ExtractionSource.OCR)
                )
                order += 1
                paragraph_lines = []
            continue
        paragraph_lines.append(line.text)
    if paragraph_lines:
        blocks.append(
            Block(block_type=BlockType.PARAGRAPH, text=" ".join(paragraph_lines), page_number=page_number, order=order, source=ExtractionSource.OCR)
        )
        order += 1

    mean_conf = sum(w.confidence for w in words) / len(words)
    # Attach page-level mean OCR confidence to plain paragraph blocks (table
    # blocks already carry their own, region-specific confidence).
    blocks = [
        b if b.block_type == BlockType.TABLE else Block(
            block_type=b.block_type,
            text=b.text,
            page_number=b.page_number,
            order=b.order,
            bbox=b.bbox,
            heading_level=b.heading_level,
            table=b.table,
            source=b.source,
            confidence=round(mean_conf / 100.0, 3),
        )
        for b in blocks
    ]
    return blocks, order, mean_conf, preprocessing_metadata


def parse_pdf(
    path: Path,
    ocr_language: str = "eng",
    ocr_dpi: int = 200,
    min_native_chars: int = 20,
    ocr_confidence_floor: float = 0.0,
    preprocess: bool = True,
    settings: Settings | None = None,
) -> ExtractedDocument:
    """`settings` drives Phase 4 visual extraction
    (`figure_extraction_enabled`/`chart_understanding_enabled`/
    `vision_model` — see app.core.config.Settings and
    app.documents.figures.extract) — kept as ONE settings object rather
    than more scattered keyword arguments (spec section 1, "do not
    scatter dozens of figure-specific arguments across parser
    functions"). Every existing positional/keyword call site (this
    function's OWN text/OCR parameters, and every test that calls
    `parse_pdf(path)` with no `settings`) is unaffected: `settings`
    defaults to `None`, in which case a default `Settings()` -- figure
    extraction ON, chart understanding OFF (its own default) -- is used.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise ValueError("PDF parsing requires the 'pdfplumber' package") from exc

    if settings is None:
        settings = Settings(_env_file=None)

    doc = ExtractedDocument(source_path=str(path), file_type="pdf")
    order = 0

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise ValueError(f"{path.name}: could not be read as a PDF ({exc})") from exc

    scanned_page_numbers: list[int] = []
    with pdf:
        for i, page in enumerate(pdf.pages, start=1):
            native_blocks, order, native_text = _parse_native_page(page, i, order, str(path))
            # Phase 4 Issue 4 fix: a page is scanned when it lacks
            # USABLE native content overall -- not merely when its plain
            # paragraph text is short. The old condition
            # (`is_scanned_page(native_text, ...) and not any table`)
            # let ANY detected table object block the OCR fallback
            # outright, even a near-empty/spurious one (e.g. pdfplumber
            # mistaking a couple of ruled lines for a table on an
            # otherwise-scanned page) -- see `_native_content_chars`.
            total_native_chars = _native_content_chars(native_text, native_blocks)
            needs_ocr = total_native_chars < min_native_chars
            if needs_ocr:
                scanned_page_numbers.append(i)
                doc.pages.append(
                    PageInfo(
                        page_number=i,
                        width=page.width,
                        height=page.height,
                        coordinate_space=CoordinateSpace.PDF_POINTS,
                        is_scanned=True,
                    )
                )
                # native_blocks discarded for this page; OCR runs below in one batch render.
            else:
                doc.pages.append(
                    PageInfo(
                        page_number=i,
                        width=page.width,
                        height=page.height,
                        coordinate_space=CoordinateSpace.PDF_POINTS,
                        is_scanned=False,
                    )
                )
                doc.blocks.extend(native_blocks)

                # Phase 4 (spec Problem 1): embedded-image (figure/chart)
                # extraction, native pages only -- a SCANNED page's
                # "embedded image" is the whole rasterized page itself
                # (that's precisely what OCR above already processes as
                # the page's text/table content), so running figure
                # extraction there would misdetect the entire scan as one
                # giant spurious "figure" rather than finding genuine
                # embedded figures. `extract_pdf_page_visuals` itself
                # no-ops when `settings.figure_extraction_enabled` is
                # False. Never allowed to fail the whole page/document —
                # a bad embedded image is recorded as a warning, exactly
                # like a bad OCR render above, not an ingestion failure
                # (spec Problem 23).
                try:
                    visual_blocks = extract_pdf_page_visuals(page, i, str(path), order, settings)
                except Exception as exc:  # noqa: BLE001
                    doc.warnings.append(f"page {i}: visual extraction failed ({exc})")
                    visual_blocks = []
                if visual_blocks:
                    doc.blocks.extend(visual_blocks)
                    order += len(visual_blocks)

    if scanned_page_numbers:
        try:
            images = render_pdf_pages(path, dpi=ocr_dpi)
        except RuntimeError as exc:
            doc.warnings.append(str(exc))
            images = []
        for page_number in scanned_page_numbers:
            if page_number - 1 >= len(images):
                doc.warnings.append(f"page {page_number}: could not be rendered for OCR")
                continue
            image = images[page_number - 1]
            ocr_blocks, order, mean_conf, preprocessing_metadata = _parse_scanned_page(
                image, page_number, order, ocr_language, ocr_confidence_floor, str(path) ,preprocess=preprocess, ocr_dpi=ocr_dpi
            )
            if not ocr_blocks:
                doc.warnings.append(f"page {page_number}: OCR found no text")
            doc.blocks.extend(ocr_blocks)
            for idx, p in enumerate(doc.pages):
                if p.page_number == page_number:
                    doc.pages[idx] = PageInfo(
                        page_number=page_number,
                        width=p.width,
                        height=p.height,
                        coordinate_space=CoordinateSpace.PDF_POINTS,
                        is_scanned=True,
                        ocr_confidence=mean_conf,
                        ocr_preprocessing=preprocessing_metadata,
                    )

    doc.blocks.sort(key=lambda b: (b.page_number or 0, b.order))
   
    # TEMP DEBUG — remove after diagnosing caption association
    for block in doc.blocks:
        if block.visual is not None:
            print(
                "\nVISUAL:",
                block.visual.object_type,
                "page=", block.visual.page_number,
                "bbox=", block.visual.bbox,
                "caption=", repr(block.visual.caption),
            )
        else:
            print(
                "TEXT:",
                repr(block.text),
                "page=", block.page_number,
                "bbox=", block.bbox,
            )     

    # Phase 4 (spec Problem 1/6): caption association must run once, over
    # the FULL assembled block list (text + tables + visuals, all pages),
    # since a caption can be the paragraph immediately before/after a
    # figure anywhere in reading order -- see `associate_all_captions`'s
    # docstring for why this can't happen per-page inside the loop above.
    if any(b.visual is not None for b in doc.blocks):
        doc.blocks = associate_all_captions(doc.blocks)

    _apply_table_continuations(doc, str(path))

    if not doc.blocks:
        raise ValueError(f"{path.name}: extracted no text (empty PDF, or OCR failed on every scanned page)")

    return doc


def _apply_table_continuations(doc: ExtractedDocument, source_path: str) -> None:
    """Run multi-page table continuation detection (Phase 4) across every
    TABLE block in `doc`, in page order, and write the (possibly-updated
    — see `detect_continuations`/`_strip_repeated_header`) `TableData`
    back onto each block. A no-op if the document has 0 or 1 tables —
    continuation only ever applies between two or more.
    """
    table_indices = [i for i, b in enumerate(doc.blocks) if b.block_type == BlockType.TABLE and b.table is not None]
    if len(table_indices) < 2:
        return

    tables_by_page = [(doc.blocks[i].page_number or 0, doc.blocks[i].table) for i in table_indices]
    updated = detect_continuations(tables_by_page, source_path)

    for block_idx, new_table in zip(table_indices, updated):
        old_block = doc.blocks[block_idx]
        doc.blocks[block_idx] = dataclasses.replace(old_block, table=new_table, text=new_table.to_markdown())