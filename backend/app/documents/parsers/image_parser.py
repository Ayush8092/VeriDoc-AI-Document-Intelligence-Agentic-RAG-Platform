"""Parse standalone `.png`/`.jpg`/`.jpeg` files via OCR.

Handles both plain photos/screenshots of text and screenshots/photos of
tables — the same heuristic table reconstruction used for scanned PDF
pages (`app/documents/tables/reconstruct.py`) applies here, since a
"screenshot of a table" and a "scanned table page" are the same problem
once both are just an image.

Coordinate provenance (Phase 4 fix): OCR preprocessing
(`app.documents.visual.ocr_preprocessing`) can reorient, deskew, crop and
resize the image before Tesseract ever sees it, so the OCR word/line
bboxes `preprocess_and_ocr` returns are in THAT processed image's pixel
grid — not the original upload's. This module now maps every bbox back
through `CoordinateTransform` to the ORIGINAL image's pixel space before
it ever reaches a `Block`, and reports the original image's own
`width`/`height` on `PageInfo` — exactly the same pattern
`app/documents/parsers/pdf_parser.py`'s `_parse_scanned_page` already
uses for scanned PDF pages (see that function's docstring). Before this
fix, a caller drawing a highlight box or table-cell region on the
*original* uploaded image using these bboxes would draw it in the wrong
place whenever preprocessing actually did anything (deskew, crop,
orientation correction, or resize) — see
`tests/test_image_parser_coordinates.py` for regression coverage of each
transformation individually and in combination.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.documents.models import (
    Block,
    BlockType,
    BoundingBox,
    CoordinateSpace,
    ExtractedDocument,
    ExtractionSource,
    PageInfo,
    TableData,
)
from app.documents.ocr.engine import group_lines
from app.documents.tables.reconstruct import detect_tables


def parse_image(
    path: Path,
    ocr_language: str = "eng",
    ocr_confidence_floor: float = 0.0,
    preprocess: bool = True,
) -> ExtractedDocument:
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise ValueError(f"{path.name}: could not be read as an image ({exc})") from exc

    # The ORIGINAL upload's own dimensions — this is what `PageInfo` must
    # report and what every bbox below must ultimately be expressed
    # against, regardless of what preprocessing did internally.
    original_size = (image.width, image.height)

    file_type = path.suffix.lower().lstrip(".")
    doc = ExtractedDocument(
        source_path=str(path),
        file_type=file_type,
        pages=[
            PageInfo(
                page_number=1,
                width=original_size[0],
                height=original_size[1],
                coordinate_space=CoordinateSpace.IMAGE_PIXELS,
                is_scanned=True,
            )
        ],
    )

    # Phase 4 OCR preprocessing, via the ONE shared implementation also used
    # by the scanned-PDF path (app/documents/parsers/pdf_parser.py) — see
    # app.documents.visual.ocr_preprocessing.preprocess_and_ocr's docstring,
    # "Shared by both ingestion paths". `preprocess_and_ocr` itself never
    # raises for a preprocessing failure (it falls back to the original
    # image internally — see `run_preprocessing`), so no try/except is
    # needed at this call site.
    from app.documents.visual.coordinates import CoordinateTransform
    from app.documents.visual.ocr_preprocessing import preprocess_and_ocr

    words, ocr_image, preprocessing_metadata = preprocess_and_ocr(
        image, language=ocr_language, confidence_floor=ocr_confidence_floor, preprocess=preprocess
    )
    if not words:
        raise ValueError(f"{path.name}: OCR found no text in this image")

    # Maps every OCR bbox below from `ocr_image`'s (post orientation/
    # deskew/crop/resize) pixel grid back to `original_size` — the same
    # transform construction pdf_parser.py's `_parse_scanned_page` uses,
    # minus the DPI->PDF-points step (there's no PDF page here; the
    # canonical space for a standalone image IS its own original pixel
    # grid). `image_ref="original"` names which image these coordinates
    # are relative to, for a future multi-image-per-block scenario.
    transform = CoordinateTransform.from_preprocessing(
        rendered_size=original_size,
        preprocessing_metadata=preprocessing_metadata,
        dpi=None,
        output_space=CoordinateSpace.IMAGE_PIXELS,
        image_ref="original",
    )

    lines = group_lines(words)
    table_regions = detect_tables(lines, object_id_prefix=str(path))
    consumed = set()
    order = 0

    for table_data, confidence, (start, end) in table_regions:
        consumed.update(range(start, end + 1))
        region_words = [w for k in range(start, end + 1) for w in lines[k].words]
        raw_bbox = BoundingBox(
            x0=min(w.left for w in region_words),
            y0=min(w.top for w in region_words),
            x1=max(w.left + w.width for w in region_words),
            y1=max(w.top + w.height for w in region_words),
        )
        bbox = transform.bbox_to_original(raw_bbox)
        # Phase 4: map each OCR-reconstructed cell's bbox (still in
        # ocr_image's pre-transform pixel space — see
        # `reconstruct.detect_tables`'s docstring) through the SAME
        # transform used for the table's own bbox above, so cell
        # geometry ends up in the document's canonical coordinate space
        # too, not left in a throwaway intermediate pixel grid.
        import dataclasses as _dc

        table_data = _dc.replace(
            table_data,
            cells=tuple(
                _dc.replace(c, bbox=transform.bbox_to_original(c.bbox)) if c.bbox else c
                for c in table_data.cells
            ),
        )
        doc.blocks.append(
            Block(
                block_type=BlockType.TABLE,
                text=table_data.to_markdown(),
                page_number=1,
                order=order,
                table=table_data,
                bbox=bbox,
                source=ExtractionSource.OCR,
                confidence=confidence,
            )
        )
        order += 1

    mean_conf = round((sum(w.confidence for w in words) / len(words)) / 100.0, 3)
    paragraph_lines = []
    paragraph_words: list = []

    def _flush_paragraph():
        nonlocal order, paragraph_lines, paragraph_words
        if not paragraph_lines:
            return
        bbox = None
        if paragraph_words:
            raw_bbox = BoundingBox(
                x0=min(w.left for w in paragraph_words),
                y0=min(w.top for w in paragraph_words),
                x1=max(w.left + w.width for w in paragraph_words),
                y1=max(w.top + w.height for w in paragraph_words),
            )
            bbox = transform.bbox_to_original(raw_bbox)
        doc.blocks.append(
            Block(
                block_type=BlockType.PARAGRAPH,
                text=" ".join(paragraph_lines),
                page_number=1,
                order=order,
                bbox=bbox,
                source=ExtractionSource.OCR,
                confidence=mean_conf,
            )
        )
        order += 1
        paragraph_lines = []
        paragraph_words = []

    for idx, line in enumerate(lines):
        if idx in consumed or not line.text.strip():
            _flush_paragraph()
            continue
        paragraph_lines.append(line.text)
        paragraph_words.extend(line.words)
    _flush_paragraph()

    doc.pages[0] = PageInfo(
        page_number=1,
        width=original_size[0],
        height=original_size[1],
        coordinate_space=CoordinateSpace.IMAGE_PIXELS,
        is_scanned=True,
        ocr_confidence=mean_conf * 100,
        ocr_preprocessing=preprocessing_metadata,
    )

    if not doc.blocks:
        raise ValueError(f"{path.name}: OCR found no usable text or tables")

    return doc
