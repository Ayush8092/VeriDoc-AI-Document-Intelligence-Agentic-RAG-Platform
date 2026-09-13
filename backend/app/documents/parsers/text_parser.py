"""Parse `.txt` and `.md` files into an `ExtractedDocument`.

`.md` headings (`#`, `##`, ...) become HEADING blocks with `heading_level`
set; GitHub-style pipe tables (`| a | b |` rows with a `|---|---|`
separator) become dedicated TABLE blocks, same as every other format (see
docs/tables.md); everything else becomes PARAGRAPH blocks split on blank
lines. `.txt` has no heading/table syntax, so the whole file becomes a
sequence of paragraph blocks under one implicit section — chunking.py's
fallback ("no headings found -> one section") handles that the same way
it always has.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.documents.models import Block, BlockType, ExtractedDocument, ExtractionSource, PageInfo, TableData

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s*[-*+]\s+.+$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _parse_pipe_table(lines: list[str]) -> TableData | None:
    """Parse a run of `| a | b |` lines (with a `|---|---|` separator row)
    into a `TableData`. Returns None if `lines` isn't a valid pipe table.
    """
    if len(lines) < 2 or not _TABLE_ROW_RE.match(lines[0]) or not _TABLE_SEP_RE.match(lines[1]):
        return None
    rows = []
    for line in [lines[0]] + lines[2:]:
        if not _TABLE_ROW_RE.match(line):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(tuple(cells))
    return TableData(rows=tuple(rows)) if rows else None


def parse_text(path: Path) -> ExtractedDocument:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8 text") from exc

    file_type = path.suffix.lower().lstrip(".")
    doc = ExtractedDocument(source_path=str(path), file_type=file_type, pages=[PageInfo(page_number=1)])

    order = 0
    paragraph_lines: list[str] = []
    raw_lines = raw.splitlines()

    def flush_paragraph():
        nonlocal order
        text = "\n".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if not text:
            return
        block_type = BlockType.LIST_ITEM if _LIST_ITEM_RE.match(text.splitlines()[0]) else BlockType.PARAGRAPH
        doc.blocks.append(
            Block(block_type=block_type, text=text, page_number=1, order=order, source=ExtractionSource.NATIVE)
        )
        order += 1

    i = 0
    n = len(raw_lines)
    while i < n:
        line = raw_lines[i]

        heading_match = _HEADING_RE.match(line) if file_type == "md" else None
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            doc.blocks.append(
                Block(
                    block_type=BlockType.HEADING,
                    text=title,
                    page_number=1,
                    order=order,
                    heading_level=level,
                    source=ExtractionSource.NATIVE,
                )
            )
            order += 1
            i += 1
            continue

        if file_type == "md" and _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(raw_lines[i + 1]):
            j = i
            table_lines = []
            while j < n and (_TABLE_ROW_RE.match(raw_lines[j]) or (j == i + 1 and _TABLE_SEP_RE.match(raw_lines[j]))):
                table_lines.append(raw_lines[j])
                j += 1
            table_data = _parse_pipe_table(table_lines)
            if table_data is not None:
                flush_paragraph()
                doc.blocks.append(
                    Block(
                        block_type=BlockType.TABLE,
                        text=table_data.to_markdown(),
                        page_number=1,
                        order=order,
                        table=table_data,
                        source=ExtractionSource.NATIVE,
                    )
                )
                order += 1
                i = j
                continue

        if line.strip() == "":
            flush_paragraph()
            i += 1
            continue
        paragraph_lines.append(line)
        i += 1

    flush_paragraph()

    if not doc.blocks:
        raise ValueError(f"{path.name}: extracted no text (empty file)")

    return doc

