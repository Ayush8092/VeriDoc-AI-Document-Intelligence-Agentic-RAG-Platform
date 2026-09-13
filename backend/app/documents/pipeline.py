"""Ingestion entry point: file path -> `ExtractedDocument`.

This is the one place that dispatches by extension. Every parser produces
the same `ExtractedDocument` shape (see app/documents/models.py), so
everything after this point — chunking, embedding, DB persistence — is
format-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.documents.models import ExtractedDocument
from app.documents.parsers.docx_parser import parse_docx
from app.documents.parsers.image_parser import parse_image
from app.documents.parsers.pdf_parser import parse_pdf
from app.documents.parsers.text_parser import parse_text
from app.documents.provenance import assign_provenance

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}


class DocumentParseError(ValueError):
    """Raised for an unsupported extension or a file that fails to parse."""


def extract_document(path: Path, settings: Settings) -> ExtractedDocument:
    suffix = path.suffix.lower()
    try:
        if suffix in (".md", ".txt"):
            doc = parse_text(path)
        elif suffix == ".docx":
            doc = parse_docx(path, settings=settings)
        elif suffix == ".pdf":
            doc = parse_pdf(
                path,
                ocr_language=settings.ocr_language,
                ocr_dpi=settings.ocr_dpi,
                min_native_chars=settings.ocr_min_native_chars_per_page,
                ocr_confidence_floor=settings.ocr_confidence_floor,
                settings=settings,
            )
        elif suffix in (".png", ".jpg", ".jpeg"):
            doc = parse_image(path, ocr_language=settings.ocr_language, ocr_confidence_floor=settings.ocr_confidence_floor)
        else:
            raise DocumentParseError(
                f"{path.name}: unsupported file type '{suffix}' "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize every parser failure to one error type
        raise DocumentParseError(f"{path.name}: failed to parse ({exc})") from exc

    # Phase 4: assign every block's object_id/parent_object_id exactly
    # once, centrally, right here — see app.documents.provenance's
    # docstring for why this lives in the ONE shared entry point rather
    # than each parser computing its own IDs. This call was previously
    # documented but not actually wired in; fixed as part of this pass
    # (docs/architecture.md notes it under "Fixed in this pass").
    return assign_provenance(doc)