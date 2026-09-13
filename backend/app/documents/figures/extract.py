"""Orchestrates figure/chart extraction: extractor -> classification ->
chart refinement -> `VisualObject`/`Block` construction. Caption
association runs separately, AFTER all of a document's blocks (text +
tables + these visual blocks) exist — see `associate_all_captions` below
— since it needs to search the full block list, not just the visuals.

This is the ONE place that turns a raw `ExtractedImage`
(`extractor.py`) into the project's unified `VisualObject` model — so
every visual, from every source format, goes through the exact same
classification/chart-refinement path (spec section 13, "one unified
model").
"""

from __future__ import annotations

import dataclasses

from app.core.config import Settings
from app.documents.figures.caption import associate_captions
from app.documents.figures.chart_understanding import extract_chart_data, refine_chart_type
from app.documents.figures.classification import classify_image
from app.documents.figures.extractor import (
    ExtractedImage,
    extract_images_from_docx,
    extract_images_from_paragraph,
    extract_images_from_pdf_page,
)
from app.documents.models import Block, BlockType, ExtractionSource, VisualObject, VisualObjectType, _stable_object_id

_MIN_CLASSIFICATION_CONFIDENCE_FOR_CHART_REFINEMENT = 0.0  # refine even a low-confidence CHART guess; it's still the best guess we have


def _visual_block_from_image(img: ExtractedImage, source_path: str, order: int, settings: Settings) -> Block:
    classification = classify_image(img.data)
    object_id = _stable_object_id(source_path, "visual", str(img.page_number), str(order))

    chart_type = None
    metadata: dict = {}
    if classification.object_type == VisualObjectType.CHART:
        block_type = BlockType.CHART
    elif classification.object_type == VisualObjectType.FIGURE:
        block_type = BlockType.FIGURE
    else:
        block_type = BlockType.VISUAL  # PHOTO, DIAGRAM, ILLUSTRATION, SCREENSHOT, UNKNOWN_VISUAL

    if classification.object_type == VisualObjectType.CHART:
        chart_type, type_confidence = refine_chart_type(img.data)
        metadata["chart_type_confidence"] = type_confidence
        chart_data = extract_chart_data(img.data, settings)
        if chart_data is not None:
            metadata["chart_data"] = chart_data

    visual = VisualObject(
        object_id=object_id,
        object_type=classification.object_type,
        page_number=img.page_number,
        bbox=img.bbox,
        confidence=classification.confidence,
        source=ExtractionSource.NATIVE,
        extraction_method=img.extraction_method,
        chart_type=chart_type,
        metadata={**metadata, "classification_signals": classification.signals, "width": img.width, "height": img.height},
    )

    return Block(
        block_type=block_type,
        text=_visual_display_text(visual),
        page_number=img.page_number,
        order=order,
        bbox=img.bbox,
        source=ExtractionSource.NATIVE,
        confidence=classification.confidence,
        visual=visual,
        extraction_method=img.extraction_method,
    )


def _visual_display_text(visual: VisualObject) -> str:
    """The text embedded/chunked for this visual (spec section 14,
    "extend chunking for figure/chart/visual evidence") — a compact,
    factual description built ONLY from fields that were actually
    extracted, never inventing a description of image content this
    module didn't actually determine.
    """
    parts = [f"[{visual.object_type.value}]"]
    if visual.chart_type:
        parts.append(f"({visual.chart_type.value} chart)")
    if visual.caption:
        parts.append(visual.caption)
    chart_data = visual.metadata.get("chart_data")
    if isinstance(chart_data, dict):
        if chart_data.get("title"):
            parts.append(f"Title: {chart_data['title']}")
        if chart_data.get("x_axis_label"):
            parts.append(f"X axis: {chart_data['x_axis_label']}")
        if chart_data.get("y_axis_label"):
            parts.append(f"Y axis: {chart_data['y_axis_label']}")
        if chart_data.get("legend"):
            parts.append("Legend: " + ", ".join(chart_data["legend"]))
        for series in chart_data.get("series", []):
            name = series.get("name") or "series"
            values = ", ".join(str(v) for v in series.get("values", []))
            if values:
                parts.append(f"{name}: {values}")
    return " ".join(parts)


def extract_pdf_page_visuals(page, page_number: int, source_path: str, order_start: int, settings: Settings) -> list[Block]:
    if not settings.figure_extraction_enabled:
        return []
    images = extract_images_from_pdf_page(page, page_number)
    return [_visual_block_from_image(img, source_path, order_start + i, settings) for i, img in enumerate(images)]


def extract_docx_visuals(document, source_path: str, order_start: int, settings: Settings) -> list[Block]:
    if not settings.figure_extraction_enabled:
        return []
    images = extract_images_from_docx(document)
    return [_visual_block_from_image(img, source_path, order_start + i, settings) for i, img in enumerate(images)]


def extract_docx_paragraph_visuals(paragraph, document, source_path: str, order: int, settings: Settings) -> list[Block]:
    """Same as `extract_docx_visuals`, scoped to ONE paragraph — see
    `app.documents.figures.extractor.extract_images_from_paragraph`'s
    docstring for why this (not the bulk, whole-document function above)
    is what `docx_parser.py` actually calls, interleaved into its
    paragraph loop: caption association needs the image's `Block.order`
    to be its TRUE position in the document, not appended at the end.
    """
    if not settings.figure_extraction_enabled:
        return []
    images = extract_images_from_paragraph(paragraph, document)
    return [_visual_block_from_image(img, source_path, order + i, settings) for i, img in enumerate(images)]


def associate_all_captions(all_blocks: list[Block]) -> list[Block]:
    """Run caption association across every visual block in a fully
    -assembled document (called once, after all blocks — text, tables,
    visuals — exist and are in reading order) and return an updated
    block list with `.visual.caption`/`caption_bbox`/`caption_confidence`
    filled in where a match was found. Non-visual blocks pass through
    unchanged.
    """
    visual_blocks = [b for b in all_blocks if b.visual is not None]
    if not visual_blocks:
        return all_blocks

    matches = associate_captions(visual_blocks, all_blocks)
    if not matches:
        return all_blocks

    out = list(all_blocks)
    for i, block in enumerate(out):
        if block.visual is None or block.visual.object_id not in matches:
            continue
        match = matches[block.visual.object_id]
        cap_block = all_blocks[match.block_index]
        updated_visual = dataclasses.replace(
            block.visual,
            caption=match.text,
            caption_bbox=cap_block.bbox,
            caption_confidence=match.confidence,
        )
        out[i] = dataclasses.replace(block, visual=updated_visual, text=_visual_display_text(updated_visual))
    return out