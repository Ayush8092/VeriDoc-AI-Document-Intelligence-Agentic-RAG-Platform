"""Parse `.docx` files into an `ExtractedDocument`, preserving tables.

Unlike the baseline loader (which joined table cells into `" | "`-separated
plain text indistinguishable from prose), tables are kept as structured
`TableData` blocks — see docs/tables.md. Paragraph styles named
`Heading 1..6` (Word's built-in heading styles) become HEADING blocks;
everything else is a paragraph or list item based on the paragraph's list
formatting.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.documents.figures.extract import associate_all_captions, extract_docx_paragraph_visuals
from app.documents.models import Block, BlockType, ExtractedDocument, ExtractionSource, PageInfo, TableData


def _heading_level(style_name: str) -> int | None:
    if not style_name:
        return None
    style_name = style_name.strip().lower()
    if style_name.startswith("heading "):
        try:
            return int(style_name.split(" ", 1)[1])
        except ValueError:
            return 1
    if style_name in ("title",):
        return 1
    return None


def parse_docx(path: Path, settings: Settings | None = None) -> ExtractedDocument:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ValueError("DOCX parsing requires the 'python-docx' package") from exc

    if settings is None:
        settings = Settings(_env_file=None)

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ValueError(f"{path.name}: could not be read as a DOCX ({exc})") from exc

    doc = ExtractedDocument(source_path=str(path), file_type="docx", pages=[PageInfo(page_number=1)])
    order = 0

    # python-docx exposes body children in document order via `element.body`,
    # so we walk paragraphs and tables interleaved rather than
    # `document.paragraphs` then `document.tables` separately (which would
    # silently reorder a document where a table sits between two sections).
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph as DocxParagraph
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            para = DocxParagraph(child, document)

            # Phase 4 (spec Problem 2): extract any inline image embedded
            # IN THIS paragraph before the text-emptiness check below —
            # a paragraph holding only an image (the overwhelmingly
            # common case) has empty `.text` and would otherwise be
            # skipped entirely, silently dropping the image. Never
            # allowed to fail parsing of the rest of the document (spec
            # Problem 23) — a bad embedded image becomes a warning.
            try:
                visual_blocks = extract_docx_paragraph_visuals(para, document, str(path), order, settings)
            except Exception as exc:  # noqa: BLE001
                doc.warnings.append(f"paragraph {order}: visual extraction failed ({exc})")
                visual_blocks = []
            if visual_blocks:
                doc.blocks.extend(visual_blocks)
                order += len(visual_blocks)

            text = para.text.strip()
            if not text:
                continue
            level = _heading_level(para.style.name if para.style else "")
            if level:
                doc.blocks.append(
                    Block(block_type=BlockType.HEADING, text=text, page_number=1, order=order, heading_level=level)
                )
            else:
                is_list = bool(para.style and "list" in para.style.name.lower())
                doc.blocks.append(
                    Block(
                        block_type=BlockType.LIST_ITEM if is_list else BlockType.PARAGRAPH,
                        text=text,
                        page_number=1,
                        order=order,
                    )
                )
            order += 1
        elif isinstance(child, CT_Tbl):
            table = DocxTable(child, document)
            rows: list[tuple[str, ...]] = []
            for row in table.rows:
                rows.append(tuple(cell.text.strip() for cell in row.cells))
            if not rows:
                continue
            table_data = TableData(rows=tuple(rows))
            doc.blocks.append(
                Block(
                    block_type=BlockType.TABLE,
                    text=table_data.to_markdown(),
                    page_number=1,
                    order=order,
                    table=table_data,
                )
            )
            order += 1

    # Phase 4 (spec Problem 1/6): caption association over the full,
    # already-interleaved block list — see
    # `app.documents.figures.extract.associate_all_captions`'s docstring.
    if any(b.visual is not None for b in doc.blocks):
        doc.blocks = associate_all_captions(doc.blocks)

    if not doc.blocks:
        raise ValueError(f"{path.name}: extracted no text (empty or unsupported DOCX)")

    return doc