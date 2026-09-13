"""Extract embedded raster images (candidate figures/charts) from a page.

This module only extracts and locates images — it does not classify or
understand them (see `classification.py`, `chart_understanding.py`). One
function per source format, both returning the same
`(image_bytes, bbox_or_None, page_number)` tuple shape, so downstream
code (`classification.classify_image`, `caption.associate_captions`) is
format-agnostic.

Embedded images below `MIN_FIGURE_DIMENSION_PX` in either dimension are
skipped — these are overwhelmingly bullet-point icons, logos in a
letterhead, or decorative rules, not genuine figures/charts, and
including them would flood chunking/retrieval with noise no user query
is ever about.
"""

from __future__ import annotations

import io
import logging

from app.documents.models import BoundingBox, CoordinateSpace

_log = logging.getLogger(__name__)

MIN_FIGURE_DIMENSION_PX = 48
MIN_FIGURE_AREA_PX = MIN_FIGURE_DIMENSION_PX * MIN_FIGURE_DIMENSION_PX * 2


class ExtractedImage:
    __slots__ = ("data", "bbox", "page_number", "width", "height", "extraction_method")

    def __init__(self, data: bytes, bbox: BoundingBox | None, page_number: int, width: int, height: int, extraction_method: str):
        self.data = data
        self.bbox = bbox
        self.page_number = page_number
        self.width = width
        self.height = height
        self.extraction_method = extraction_method


def _decode_and_check(raw: bytes) -> tuple[bytes, int, int] | None:
    """Decode with Pillow to get real pixel dimensions (a PDF/DOCX image
    stream's declared size can lie or be absent) and re-encode as PNG so
    every downstream consumer (classification, chart understanding, the
    frontend) deals with exactly one, universally-supported format
    regardless of the source's original encoding (JPEG, CCITT, etc.).
    Returns None for anything Pillow can't decode, or too small to be a
    genuine figure.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
        width, height = img.size
        if width < MIN_FIGURE_DIMENSION_PX or height < MIN_FIGURE_DIMENSION_PX or width * height < MIN_FIGURE_AREA_PX:
            return None
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), width, height
    except Exception as exc:  # noqa: BLE001 - a single corrupt embedded image must not fail the whole page
        _log.debug("Could not decode candidate embedded image: %s", exc)
        return None


def extract_images_from_pdf_page(page, page_number: int) -> list[ExtractedImage]:
    """`page` is a pdfplumber `Page`. Uses `page.images` (pdfplumber's own
    inventory of embedded XObject images, each with a real, native PDF
    -points bbox — the same coordinate space `pdf_parser.py` already uses
    for text/table bboxes, so no `CoordinateTransform` is needed here,
    unlike OCR-derived bboxes) and `page.crop(bbox).to_image()` to
    rasterize each one to real pixel bytes (pdfplumber's `image["stream"]`
    gives the raw XObject bytes, which can be in encodings — JPXDecode,
    CCITTFaxDecode — Pillow can't always decode directly; cropping and
    re-rendering the page region is slower but format-agnostic and
    reliable).
    """
    out: list[ExtractedImage] = []
    try:
        images = page.images
    except Exception as exc:  # noqa: BLE001
        _log.debug("page.images failed on page %s: %s", page_number, exc)
        return out

    for img in images:
        try:
            x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
            if x1 <= x0 or bottom <= top:
                continue
            # Clip to the page's own bounds -- an image element can report
            # a bbox that slightly overhangs the page edge.
            x0, top = max(x0, 0), max(top, 0)
            x1, bottom = min(x1, page.width), min(bottom, page.height)
            if x1 <= x0 or bottom <= top:
                continue
            cropped = page.crop((x0, top, x1, bottom))
            pil_image = cropped.to_image(resolution=150).original
        except Exception as exc:  # noqa: BLE001 - one bad image must not sink the whole page
            _log.debug("Could not rasterize embedded image on page %s: %s", page_number, exc)
            continue

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        decoded = _decode_and_check(buf.getvalue())
        if decoded is None:
            continue
        png_bytes, width, height = decoded
        out.append(
            ExtractedImage(
                data=png_bytes,
                bbox=BoundingBox(x0=x0, y0=top, x1=x1, y1=bottom, coordinate_space=CoordinateSpace.PDF_POINTS),
                page_number=page_number,
                width=width,
                height=height,
                extraction_method="pdfplumber_embedded_image",
            )
        )
    return out

def extract_images_from_paragraph(paragraph, document) -> list[ExtractedImage]:
    """Extract images embedded in one DOCX paragraph.

    python-docx exposes inline shapes globally through
    ``document.inline_shapes``, but does not expose a direct
    paragraph -> image API.

    We therefore inspect the paragraph XML for ``a:blip`` elements,
    resolve their ``r:embed`` relationship IDs through
    ``document.part.related_parts``, and decode each image using the
    same validation path as the bulk DOCX extractor.

    DOCX does not provide stable page coordinates during parsing, so
    extracted images use page_number=1 and bbox=None.
    """
    out: list[ExtractedImage] = []

    try:
        from docx.oxml.ns import qn

        # Search only inside this paragraph. This is important because
        # the DOCX parser processes paragraphs in document order.
        for blip in paragraph._p.iter():
            if blip.tag != qn("a:blip"):
                continue

            rid = blip.get(qn("r:embed"))
            if not rid:
                continue

            try:
                image_part = document.part.related_parts[rid]
                raw = image_part.blob
            except Exception as exc:  # noqa: BLE001
                _log.debug(
                    "Could not resolve DOCX image relationship %s: %s",
                    rid,
                    exc,
                )
                continue

            decoded = _decode_and_check(raw)
            if decoded is None:
                continue

            png_bytes, width, height = decoded

            out.append(
                ExtractedImage(
                    data=png_bytes,
                    bbox=None,
                    page_number=1,
                    width=width,
                    height=height,
                    extraction_method="docx_inline_shape",
                )
            )

    except Exception as exc:  # noqa: BLE001
        # A malformed paragraph/image must never abort the entire document.
        _log.debug(
            "Could not inspect paragraph images: %s",
            exc,
        )

    return out

def extract_images_from_docx(document) -> list[ExtractedImage]:
    """`document` is a `docx.Document`. DOCX has no per-page/bbox concept
    (a flow document reflows), so every extracted image gets
    `page_number=1, bbox=None` — `caption.py`'s DOCX path falls back to
    document-order adjacency instead of geometric proximity for
    associating a caption, since that's the only signal DOCX offers.
    """
    out: list[ExtractedImage] = []
    try:
        inline_shapes = document.inline_shapes
    except Exception as exc:  # noqa: BLE001
        _log.debug("document.inline_shapes unavailable: %s", exc)
        return out

    for shape in inline_shapes:
        try:
            blip = shape._inline.graphic.graphicData.pic.blipFill.blip
            rId = blip.embed
            image_part = document.part.related_parts[rId]
            raw = image_part.blob
        except Exception as exc:  # noqa: BLE001
            _log.debug("Could not extract one inline DOCX image: %s", exc)
            continue

        decoded = _decode_and_check(raw)
        if decoded is None:
            continue
        png_bytes, width, height = decoded
        out.append(
            ExtractedImage(
                data=png_bytes,
                bbox=None,
                page_number=1,
                width=width,
                height=height,
                extraction_method="docx_inline_shape",
            )
        )
    return out