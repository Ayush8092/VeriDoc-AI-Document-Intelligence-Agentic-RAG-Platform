"""Structure-aware chunking with deterministic, context-rich chunks.

Evolved from the baseline project's markdown-aware chunker (see
MIGRATION_PLAN.md): the core ideas are preserved —

- every chunk is prefixed with the document title and section for context
  so it is self-contained once pulled out of the document;
- chunk IDs are deterministic (`<source_root>::<filename>::<section-slug>
  ::<index>`), so re-running ingestion recomputes the same IDs and lands
  on the same vectors instead of creating duplicates;
- oversized text is split into overlapping windows only as a fallback cap.

What's new: chunking now walks the structured `ExtractedDocument` (see
app/documents/models.py) instead of a flat string, so:

- sections are delimited by real HEADING blocks (any supported format),
  falling back to one implicit "content" section when a document has none
  (plain .txt, a PDF with no detected headings) — same graceful fallback
  behavior as the baseline;
- a TABLE block always becomes its OWN chunk, never merged into
  surrounding paragraph text — this is what makes "table questions can be
  answered / table answers cite the correct source" possible, and what
  lets a citation carry the table's page/bbox/OCR-confidence instead of
  losing that once it's flattened into prose;
- a FIGURE/CHART/VISUAL block (Phase 4) is likewise always its own
  dedicated chunk (never merged into surrounding text, never treated as
  an isolated image with no context) — see `_visual_chunk`: its embedded
  text combines the block's own description (object type, chart type,
  caption, extracted chart data — already built by
  `app.documents.figures.extract._visual_display_text`) with the section
  heading (`_context_prefix`, same as every other chunk) and up to
  `RELATED_TEXT_CHARS` of the paragraph text immediately preceding it in
  the same section (`related_text`), so "Figure 3 + caption + surrounding
  paragraph + section heading" really does retrieve as one coherent unit
  rather than a caption-less image with no searchable context;
- every chunk carries page number(s), extraction source, and confidence,
  so a citation can point at the exact page/region it came from (see
  docs/tables.md, docs/ocr.md); a visual chunk additionally carries
  `object_id`/`caption`/`visual_type`/`chart_type`/`coordinate_space` so a
  citation can identify exactly which figure/chart/visual it came from
  (Phase 4, "visual citations").

This module is pure (no network calls) so it stays fully unit-testable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.documents.models import Block, BlockType, ExtractedDocument
from app.documents.pipeline import SUPPORTED_EXTENSIONS, extract_document

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150
RELATED_TEXT_CHARS = 300  # how much preceding paragraph text a visual chunk's `related_text` carries

EXCLUDED_FILENAMES = {"readme.txt", "readme.md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_root: str
    source_file: str
    document_title: str
    section: str
    chunk_index: int
    text: str
    content_hash: str
    block_type: str = "text"          # "text" | "table" | "figure" | "chart" | "visual"
    page_start: int | None = None
    page_end: int | None = None
    source: str = "native"            # "native" | "ocr"
    confidence: float = 1.0
    table_rows: tuple[tuple[str, ...], ...] | None = None
    bbox: dict | None = None
    # Phase 4: populated only for figure/chart/visual chunks (see
    # `_visual_chunk`). All `None`/empty for text/table chunks — every
    # pre-Phase-4 consumer of `Chunk` is unaffected.
    object_id: str | None = None
    caption: str | None = None
    visual_type: str | None = None    # VisualObjectType value, e.g. "photo", "diagram", "chart"
    chart_type: str | None = None     # ChartType value, e.g. "bar", "line"; only set when visual_type == "chart"
    coordinate_space: str | None = None
    related_text: str | None = None   # nearby paragraph text used to enrich retrieval (see module docstring)
    # Phase 6 (spec item 10, "figure/chart intelligence"): the structured
    # extraction dict `app.documents.figures.chart_understanding` produced
    # for this chart (title/axis labels/legend/series-values — see that
    # module's docstring for the exact shape), propagated end-to-end
    # alongside `chunk.text`'s natural-language description of the same
    # data. Populated only for `block_type == "chart"`; `None` for every
    # other chunk type. This is what lets `app.rag.figure_reasoning`
    # compute a chart's max/min/trend deterministically from real
    # extracted numbers instead of re-parsing them back out of prose —
    # the same "structured data alongside the text description" pattern
    # `table_rows` already established for tables.
    chart_data: dict | None = None
    # Phase 5: `None` (serialized as `""` everywhere it's persisted, same
    # convention as every other optional string field on this dataclass)
    # means "shared/public corpus" — visible to every request. A non-None
    # value is a user id (as a string) and scopes this chunk to exactly
    # that user; see app.security.auth.allowed_owner_ids,
    # app/vectorstore.py's upsert_chunks/query, and
    # app/lexical_index.py's search.
    owner_id: str | None = None


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Section:
    title: str
    blocks: list[Block] = field(default_factory=list)


def _split_into_sections(doc: ExtractedDocument) -> tuple[str, list[_Section]]:
    """Split blocks into (document_title, sections) using HEADING blocks.

    The document title is the first level-1 heading. Any block before the
    first section-worthy heading (or the whole document, if there are no
    headings at all) becomes a single "content" section — this is the
    direct equivalent of the baseline chunker's "no ## headings -> one
    header chunk" fallback, generalized to every format.
    """
    title = ""
    sections: list[_Section] = []
    current: _Section | None = None

    for block in doc.blocks:
        if block.block_type == BlockType.HEADING:
            if block.heading_level == 1 and not title:
                title = block.text
                continue
            if current is not None:
                sections.append(current)
            current = _Section(title=block.text)
            continue
        if current is None:
            current = _Section(title="content")
        current.blocks.append(block)

    if current is not None:
        sections.append(current)

    return title, sections


def _split_oversized(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + MAX_CHUNK_CHARS])
        start += MAX_CHUNK_CHARS - OVERLAP_CHARS
    return parts


def _context_prefix(title: str, section: str) -> str:
    lines = []
    if title:
        lines.append(f"Document: {title}")
    if section:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


def _page_range(blocks: list[Block]) -> tuple[int | None, int | None]:
    pages = [b.page_number for b in blocks if b.page_number is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


def _make_chunk_id(source_root: str, filename: str, slug: str, index: int, owner_id: str | None = None) -> str:
    """Deterministic chunk id (unchanged, byte-for-byte, when `owner_id`
    is `None` — every existing corpus/public-upload chunk_id, and every
    Pinecone/BM25-snapshot record already written before Phase 5, is
    unaffected). Phase 5: a per-user upload namespaces the filename
    component with the owner id, so two different users' files sharing a
    filename (e.g. both uploading "report.pdf") can never collide —
    critical, since chunk_id doubles as the Pinecone vector id
    (`app/vectorstore.py`'s `upsert_chunks`) and a collision there would
    silently let one user's chunk overwrite another user's vector.
    `Chunk.source_file` itself stays the plain original filename (this
    only affects the id, never what's displayed in a citation).
    """
    if owner_id:
        return f"{source_root}::{owner_id}/{filename}::{slug}::{index}"
    return f"{source_root}::{filename}::{slug}::{index}"


def _visual_chunk(
    block: Block,
    prefix: str,
    filename: str,
    source_root: str,
    slug: str,
    section_title: str,
    document_title: str,
    index: int,
    related_text: str | None,
    owner_id: str | None = None,
) -> Chunk:
    """Build the ONE chunk for a FIGURE/CHART/VISUAL block (Phase 4,
    "true multimodal chunking"). Mirrors the TABLE branch's shape
    (dedicated chunk, never merged into a text run) with visual-specific
    provenance carried alongside — see `Chunk`'s new fields.

    `block.text` is already the visual's compact factual description
    (object type + chart type + caption + extracted chart data — see
    `app.documents.figures.extract._visual_display_text`); this only adds
    the section-context prefix and `related_text` on top, the same
    enrichment every other chunk type already gets from `prefix`.
    """
    visual = block.visual
    body_parts = [block.text]
    if related_text:
        body_parts.append(f"Nearby text: {related_text}")
    body = "\n\n".join(p for p in body_parts if p.strip())
    text = f"{prefix}\n\n{body}".strip() if prefix else body

    return Chunk(
        chunk_id=_make_chunk_id(source_root, filename, slug, index, owner_id=owner_id),
        source_root=source_root,
        source_file=filename,
        document_title=document_title,
        section=section_title,
        chunk_index=index,
        text=text,
        content_hash=_content_hash(text),
        block_type=block.block_type.value,  # "figure" | "chart" | "visual"
        page_start=block.page_number,
        page_end=block.page_number,
        source=block.source.value,
        confidence=block.confidence,
        bbox=block.bbox.to_dict() if block.bbox else None,
        object_id=block.object_id or (visual.object_id if visual else None),
        caption=visual.caption if visual else None,
        visual_type=visual.object_type.value if visual else None,
        chart_type=visual.chart_type.value if visual and visual.chart_type else None,
        coordinate_space=block.bbox.coordinate_space.value if block.bbox else None,
        related_text=related_text,
        owner_id=owner_id,
        chart_data=(visual.metadata if visual and visual.chart_type and visual.metadata else None),
    )


def chunk_document(
    doc: ExtractedDocument, filename: str, source_root: str = "corpus", owner_id: str | None = None
) -> list[Chunk]:
    """Chunk one already-parsed `ExtractedDocument`.

    `owner_id` (Phase 5): stamped onto every produced chunk via
    `dataclasses.replace` at the very end, rather than threaded through
    each of the several internal `Chunk(...)` construction sites above —
    deliberately the smallest possible change to this already-complex
    function, and it can never miss a chunk type (text/table/visual) a
    future edit adds here, since it applies uniformly to the final list.
    """
    title, sections = _split_into_sections(doc)
    stem = Path(filename).stem
    chunks: list[Chunk] = []

    for section in sections:
        slug = _slugify(section.title)
        prefix = _context_prefix(title or stem, section.title)
        index = 0

        text_run: list[Block] = []

        def flush_text_run(run: list[Block]) -> list[Chunk]:
            nonlocal index
            if not run:
                return []
            body = "\n\n".join(b.text for b in run if b.text.strip())
            if not body.strip():
                return []
            out = []
            page_start, page_end = _page_range(run)
            confidence = min((b.confidence for b in run), default=1.0)
            source = "ocr" if any(b.source.value == "ocr" for b in run) else "native"
            for part in _split_oversized(body):
                text = f"{prefix}\n\n{part}".strip() if prefix else part
                out.append(
                    Chunk(
                        chunk_id=_make_chunk_id(source_root, filename, slug, index, owner_id=owner_id),
                        source_root=source_root,
                        source_file=filename,
                        document_title=title or stem,
                        section=section.title,
                        chunk_index=index,
                        text=text,
                        content_hash=_content_hash(text),
                        block_type="text",
                        page_start=page_start,
                        page_end=page_end,
                        source=source,
                        confidence=round(confidence, 3),
                        owner_id=owner_id,
                    )
                )
                index += 1
            return out

        for block in section.blocks:
            if block.block_type == BlockType.TABLE:
                chunks.extend(flush_text_run(text_run))
                text_run = []
                table_text = f"{prefix}\n\n{block.text}".strip() if prefix else block.text
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(source_root, filename, slug, index, owner_id=owner_id),
                        source_root=source_root,
                        source_file=filename,
                        document_title=title or stem,
                        section=section.title,
                        chunk_index=index,
                        text=table_text,
                        content_hash=_content_hash(table_text),
                        block_type="table",
                        page_start=block.page_number,
                        page_end=block.page_number,
                        source=block.source.value,
                        confidence=block.confidence,
                        table_rows=block.table.rows if block.table else None,
                        bbox=block.bbox.to_dict() if block.bbox else None,
                        owner_id=owner_id,
                    )
                )
                index += 1
            elif block.block_type in (BlockType.FIGURE, BlockType.CHART, BlockType.VISUAL):
                # Phase 4: a visual block is always its own dedicated
                # chunk (same "never merge into surrounding prose" rule
                # as TABLE, above) — but unlike a table, it's enriched
                # WITH a snippet of that surrounding prose (`related_text`)
                # rather than replacing it, since an image has no
                # searchable text of its own beyond what extraction
                # already determined (caption/chart data). The preceding
                # paragraph text is whatever's still pending in
                # `text_run` at this point — captured BEFORE flushing so
                # both the standalone text chunk AND the visual chunk's
                # enrichment see it.
                related_text = None
                if text_run:
                    preceding = " ".join(b.text for b in text_run if b.text.strip())
                    if preceding:
                        related_text = preceding[-RELATED_TEXT_CHARS:]
                chunks.extend(flush_text_run(text_run))
                text_run = []
                chunks.append(
                    _visual_chunk(
                        block,
                        prefix=prefix,
                        filename=filename,
                        source_root=source_root,
                        slug=slug,
                        section_title=section.title,
                        document_title=title or stem,
                        index=index,
                        related_text=related_text,
                        owner_id=owner_id,
                    )
                )
                index += 1
            else:
                text_run.append(block)

        chunks.extend(flush_text_run(text_run))

    if owner_id is not None:
        import dataclasses

        chunks = [dataclasses.replace(c, owner_id=owner_id) for c in chunks]

    return chunks


def chunk_file(path: Path, settings: Settings, source_root: str = "corpus") -> list[Chunk]:
    """Parse and chunk a single file (`.md`/`.txt`/`.pdf`/`.docx`/`.png`/`.jpg`/`.jpeg`)."""
    doc = extract_document(path, settings)
    return chunk_document(doc, filename=path.name, source_root=source_root)


def chunk_corpus(corpus_dir: Path, settings: Settings, source_root: str = "corpus") -> list[Chunk]:
    """Chunk every supported file in a directory. Missing directory -> no chunks."""
    if not corpus_dir.is_dir():
        return []

    paths = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and p.name.lower() not in EXCLUDED_FILENAMES
    )

    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_file(path, settings, source_root=source_root))
    return chunks


def chunk_directories(directories: list[tuple[Path, str]], settings: Settings) -> list[Chunk]:
    """Chunk several directories together into one logical corpus.

    Raises `ValueError` on a chunk_id collision across directories — should
    be unreachable given `source_root` is baked into every ID.
    """
    chunks: list[Chunk] = []
    seen: dict[str, str] = {}
    for directory, source_root in directories:
        for chunk in chunk_corpus(directory, settings, source_root=source_root):
            if chunk.chunk_id in seen:
                raise ValueError(
                    f"chunk_id collision: '{chunk.chunk_id}' was produced by both "
                    f"{seen[chunk.chunk_id]} and {directory}"
                )
            seen[chunk.chunk_id] = str(directory)
            chunks.append(chunk)
    return chunks


def preview(chunks: list[Chunk]) -> str:
    return "\n".join(
        f"{c.chunk_id}  ({len(c.text)} chars, {c.block_type})  [{c.source_file} / {c.section}]" for c in chunks
    )