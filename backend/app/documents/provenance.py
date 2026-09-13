"""Unified provenance (Phase 4 spec section 3).

The single place every `Block.object_id`/`parent_object_id` gets
assigned, run once, centrally, right after parsing
(`app.documents.pipeline.extract_document` calls this before returning)
— rather than each of the five parsers (`pdf_parser`, `docx_parser`,
`text_parser`, `image_parser`, plus OCR's scanned-page path inside
`pdf_parser`) computing its own IDs with its own convention. Table
`Cell.object_id`s are already assigned at the point of extraction
(`app.documents.tables.reconstruct` — both the native-pdfplumber and
OCR-heuristic paths), since a cell's identity is naturally local to its
table's own extraction step; this module does not touch those, only
top-level `Block`s and `VisualObject`s that don't yet have a parser
-assigned ID.

Stability contract (spec section 3, "All IDs must remain stable
throughout: parser -> block -> chunk -> retrieval -> citation"):
`object_id` is derived ONLY from content that is itself stable across a
re-ingest of an unchanged file — `source_path`, `block_type`,
`page_number`, and `order` (a block's position is deterministic given
the same input bytes and the same parser version) — never from a
wall-clock timestamp or a random UUID. Re-running ingestion on an
unchanged file reproduces the exact same `object_id`s, matching the
idempotency contract `app/chunking.py`'s `chunk_id`s already provide one
level up.
"""

from __future__ import annotations

import dataclasses

from app.documents.models import Block, ExtractedDocument, _stable_object_id


def assign_provenance(doc: ExtractedDocument) -> ExtractedDocument:
    """Return `doc` with every block's `object_id` (and, for a table's
    cells or a visual's caption, `parent_object_id` on the CHILD side)
    filled in. Idempotent and non-destructive: a block that already has
    an `object_id` (a parser that assigns its own — none currently do,
    but this keeps the door open) is left untouched.

    Mutates `doc.blocks` in place (replacing each `Block` with a
    `dataclasses.replace`d copy, since `Block` is frozen) and returns the
    same `doc` object, matching this module's callers' expectation of a
    single linear `extract_document` pipeline rather than a copy-heavy
    one.
    """
    new_blocks: list[Block] = []
    for block in doc.blocks:
        object_id = block.object_id or _stable_object_id(
            doc.source_path, block.block_type.value, str(block.page_number), str(block.order)
        )
        extraction_method = block.extraction_method or _default_extraction_method(block)

        visual = block.visual
        if visual is not None and not visual.parent_object_id:
            visual = dataclasses.replace(visual, parent_object_id=object_id)
            if visual.caption_object_id is None and visual.caption:
                visual = dataclasses.replace(
                    visual, caption_object_id=_stable_object_id(object_id, "caption")
                )

        new_blocks.append(
            dataclasses.replace(
                block,
                object_id=object_id,
                extraction_method=extraction_method,
                visual=visual,
            )
        )
    doc.blocks = new_blocks
    return doc


def _default_extraction_method(block: Block) -> str:
    """A short, honest label for HOW this block's content was obtained —
    not a re-statement of `source` (native/ocr), but specifically which
    code path (spec section 3, `extraction_method` field). Best-effort:
    falls back to a generic native/ocr label when nothing more specific
    is knowable purely from the `Block` itself (visual extraction sets
    its own, more specific `extraction_method` directly on the
    `VisualObject`/`Block` at extraction time — see
    `app.documents.figures.extractor` — this function only fills the gap
    for plain text/table blocks that never set one).
    """
    if block.visual is not None and block.visual.extraction_method:
        return block.visual.extraction_method
    if block.source.value == "ocr":
        return "tesseract_ocr"
    return "native_text_layer" if block.block_type.value != "table" else "pdfplumber_native_table"
