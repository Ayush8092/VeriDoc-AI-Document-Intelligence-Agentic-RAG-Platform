"""Detect whether a PDF page needs OCR, and render pages to images.

Per-page decision (docs/ocr.md, "Detecting scanned pages"): a page is
treated as scanned when its native text layer yields fewer than
`settings.ocr_min_native_chars_per_page` characters after stripping
whitespace — a page can be native while a neighboring page in the same
PDF is a scanned image (e.g. a signature page), so this is decided per
page, not once for the whole file.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def render_pdf_pages(path: Path, dpi: int = 200) -> list[Image.Image]:
    """Render every page of a PDF to a PIL image at the given DPI."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Scanned-PDF OCR requires the 'pdf2image' package and poppler-utils") from exc

    return convert_from_path(str(path), dpi=dpi)


def is_scanned_page(native_text: str, min_chars: int) -> bool:
    return len((native_text or "").strip()) < min_chars
