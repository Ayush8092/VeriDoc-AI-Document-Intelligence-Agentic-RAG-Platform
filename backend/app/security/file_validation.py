"""Upload content validation: magic bytes + docx zip-bomb guard.

Phase 5 completion pass, item 6 ("Validate ... the actual MIME/type where
practical ... don't trust the extension alone"). `app/api/documents.py`
already validates the extension (`SUPPORTED_EXTENSIONS`), size
(`max_upload_file_size_bytes`), and filename (`_safe_filename` — path
-traversal safe, confines every write to `upload_dir`); this module adds
the one thing those checks can't catch: a file whose *content* doesn't
match what its extension claims (e.g. `malware.exe` renamed to
`report.pdf`), and — specifically for `.docx`, since it's a zip archive —
a decompression-bomb-shaped entry.

Both checks run BEFORE the staged file is ever handed to
`extract_document`/python-docx/pdfplumber, so a malformed or hostile
upload never reaches those parsers at all.

`FileValidationError` is a `ValueError` subclass so it's caught by the
same `except (DocumentParseError, FileValidationError, ValueError)`
per-file handler `app/api/documents.py::upload_documents` already uses
for every other validation failure — one rejection path, not two.
"""

from __future__ import annotations

import zipfile
from io import BytesIO


class FileValidationError(ValueError):
    """A file's actual content doesn't match what its extension/format claims."""


# Known file-signature ("magic bytes") prefixes, keyed by lowercase
# extension. Formats with no reliable universal signature (.txt, .md —
# plain text can start with anything) are intentionally absent: nothing
# to check, so `check_file_signature` is a no-op for them.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    # JPEG's only universal constant is the SOI marker (FF D8 FF); the
    # 4th byte varies by JFIF/EXIF/Adobe variant, so it's deliberately
    # not part of the match.
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    # .docx is a zip archive (OOXML) — PK\x03\x04 is the standard local
    # -file-header signature every non-empty zip starts with.
    ".docx": (b"PK\x03\x04",),
}


def check_file_signature(data: bytes, suffix: str) -> None:
    """Raise `FileValidationError` if `data` doesn't start with a known
    signature for `suffix`. A no-op for extensions with no fixed
    signature (.txt, .md) or one this module doesn't recognize.
    """
    signatures = _SIGNATURES.get(suffix.lower())
    if not signatures:
        return
    if not any(data.startswith(sig) for sig in signatures):
        raise FileValidationError(
            f"file content does not match the expected signature for '{suffix}' "
            "(the extension and the actual file content disagree)"
        )


# Mutable at runtime (not a `Final`) so tests can lower it to exercise
# the guard without constructing an actual multi-hundred-MB fixture —
# see tests/test_upload_security.py::test_docx_zip_bomb_is_rejected.
_MAX_DOCX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500MB decompressed, well above any legitimate document


def check_docx_not_a_zip_bomb(data: bytes) -> None:
    """Raise `FileValidationError` if `data` (a .docx/zip archive) is
    unreadable as a zip, or declares more total uncompressed content
    than `_MAX_DOCX_UNCOMPRESSED_BYTES` across its entries.

    Reads each `ZipInfo.file_size` (the size the archive's own central
    directory CLAIMS an entry decompresses to) rather than actually
    inflating anything — this is deliberately cheap: a real zip bomb is
    specifically an entry that claims/produces an enormous decompressed
    size from a tiny compressed one, so the claimed size alone is
    exactly the signal worth checking before ever calling `.read()` on
    an entry.
    """
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            total_declared = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise FileValidationError(f"file is not a valid .docx/zip archive: {exc}") from exc
    except (OSError, EOFError, zipfile.LargeZipFile) as exc:
        raise FileValidationError(f"file could not be read as a .docx/zip archive: {exc}") from exc

    if total_declared > _MAX_DOCX_UNCOMPRESSED_BYTES:
        raise FileValidationError(
            f"file declares {total_declared:,} bytes of uncompressed content, exceeding the "
            f"{_MAX_DOCX_UNCOMPRESSED_BYTES:,}-byte limit (rejected as a possible zip-bomb / "
            "decompression attack, before decompressing anything)"
        )


def validate_upload_bytes(data: bytes, suffix: str) -> None:
    """The one entry point `app/api/documents.py` calls: run every
    content-level check applicable to `suffix`. Raises
    `FileValidationError` on the first failure.
    """
    check_file_signature(data, suffix)
    if suffix.lower() == ".docx":
        check_docx_not_a_zip_bomb(data)
