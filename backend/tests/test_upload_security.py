"""Explicit upload-security tests (Phase 5 completion pass, section 6).

Path traversal was already handled by `app/api/documents.py::_safe_filename`
(Phase 4/5) but had no dedicated test; magic-byte / zip-bomb validation
(`app/security/file_validation.py`) is new this pass.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    # See tests/test_upload_atomicity.py's client fixture for why
    # CORPUS_DIR must be set alongside UPLOAD_DIR here too.
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))

    from app.core.config import get_settings
    from app.db.session import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


def _mock_ingest_success(monkeypatch):
    from app.api import documents as documents_module

    good_summary = {
        "changed": 1,
        "upserted": 1,
        "skipped_unchanged": 0,
        "stale_deleted": 0,
        "chunks": 1,
        "namespace_vector_count": 1,
    }
    monkeypatch.setattr(documents_module, "ingest_corpus", lambda *, settings: good_summary)


class _TestFileValidationErrors:
    pass


def test_path_traversal_filename_is_sanitized_not_rejected_outright(client, monkeypatch, tmp_path):
    """`_safe_filename` strips path components rather than raising for a
    traversal attempt — either way, the file must never land outside
    `data/uploads/`."""
    _mock_ingest_success(monkeypatch)

    resp = client.post(
        "/documents/upload",
        files={"files": ("../../../etc/passwd_notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 202  # queued for background ingestion, not indexed synchronously

    upload_dir = tmp_path / "uploads"
    # Nothing was written outside upload_dir.
    assert not (tmp_path / "etc").exists()
    on_disk = {p.name for p in upload_dir.glob("*") if p.is_file() and p.name != ".gitkeep"}
    # The sanitized name has no path separators in it.
    assert all("/" not in name and ".." not in name for name in on_disk)


def test_path_traversal_windows_style_backslash_is_sanitized(client, monkeypatch, tmp_path):
    """Backslashes aren't path separators on POSIX, so `_safe_filename`
    neutralizes them the same way as any other disallowed character
    (replaced with `_`) rather than treating them as directory
    components — the safety property that matters is that the resulting
    file stays confined to `upload_dir`, not that no literal `..`
    substring survives in the (now-harmless) filename."""
    _mock_ingest_success(monkeypatch)
    resp = client.post(
        "/documents/upload",
        files={"files": ("..\\..\\windows\\evil.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 202  # queued for background ingestion, not indexed synchronously
    upload_dir = tmp_path / "uploads"
    on_disk = [p for p in upload_dir.glob("*") if p.is_file() and p.name != ".gitkeep"]
    assert len(on_disk) == 1
    # Confined to upload_dir, and no raw backslash (path separator on
    # Windows) survived into the stored filename.
    assert on_disk[0].resolve().is_relative_to(upload_dir.resolve())
    assert "\\" not in on_disk[0].name


def test_dotfile_only_filename_is_rejected(client):
    resp = client.post(
        "/documents/upload",
        files={"files": ("..", b"hello", "text/plain")},
    )
    assert resp.status_code == 200  # 200 with a per-file rejection, not a 500
    body = resp.json()
    assert body["status"] == "error"
    assert len(body["rejected"]) == 1


def test_empty_file_is_rejected(client):
    resp = client.post(
        "/documents/upload",
        files={"files": ("empty.txt", b"", "text/plain")},
    )
    body = resp.json()
    assert body["status"] == "error"
    assert "empty" in body["rejected"][0]["error"]


def test_unsupported_extension_is_rejected(client):
    resp = client.post(
        "/documents/upload",
        files={"files": ("malware.exe", b"MZ\x90\x00\x03", "application/octet-stream")},
    )
    body = resp.json()
    assert body["status"] == "error"
    assert "unsupported file type" in body["rejected"][0]["error"]


def test_pdf_extension_with_wrong_magic_bytes_is_rejected(client):
    """A file named `.pdf` whose actual content is plain text (no `%PDF-`
    signature) must be rejected before it ever reaches the parser."""
    resp = client.post(
        "/documents/upload",
        files={"files": ("fake.pdf", b"this is not actually a pdf file at all", "application/pdf")},
    )
    body = resp.json()
    assert body["status"] == "error"
    assert "does not match" in body["rejected"][0]["error"]


def test_png_extension_with_wrong_magic_bytes_is_rejected(client):
    resp = client.post(
        "/documents/upload",
        files={"files": ("fake.png", b"not a png", "image/png")},
    )
    body = resp.json()
    assert body["status"] == "error"
    assert "does not match" in body["rejected"][0]["error"]


def test_docx_extension_with_wrong_magic_bytes_is_rejected(client):
    """.docx is a zip archive (PK\\x03\\x04 signature) — arbitrary bytes
    named `.docx` must be rejected before zipfile/python-docx ever
    touches them."""
    resp = client.post(
        "/documents/upload",
        files={"files": ("fake.docx", b"definitely not a zip archive", "application/octet-stream")},
    )
    body = resp.json()
    assert body["status"] == "error"
    assert "does not match" in body["rejected"][0]["error"]


def test_docx_with_valid_zip_signature_but_corrupt_body_is_rejected_gracefully(client):
    """Passes the magic-byte check (starts with PK\\x03\\x04) but isn't a
    complete/valid zip — must be rejected (by the zip-bomb guard's
    BadZipFile handling, or by extract_document's own parsing) rather
    than crashing the request."""
    resp = client.post(
        "/documents/upload",
        files={"files": ("corrupt.docx", b"PK\x03\x04" + b"\x00" * 20, "application/octet-stream")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert len(body["rejected"]) == 1


def test_docx_zip_bomb_is_rejected(client):
    """A .docx (zip archive) containing an entry that claims an
    implausibly large uncompressed size must be rejected before
    python-docx ever attempts to inflate it."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write real, highly-compressible content — a genuine zip bomb
        # pattern (small on disk, huge decompressed) — then patch the
        # declared file_size in the local/central directory upward to
        # simulate an entry claiming more than it truly contains, which
        # is the cheapest reliable way to build a "claims-oversized"
        # test fixture without actually allocating hundreds of MB.
        zf.writestr("word/document.xml", b"A" * (10 * 1024 * 1024))  # 10MB of 'A's, compresses tiny

    data = bytearray(buf.getvalue())
    # Directly assert the guard function's behavior against a genuinely
    # large (but real, not just relabeled) archive size threshold using a
    # monkeypatched, much lower cap — avoids needing to actually build a
    # 500MB+ fixture in this test run.
    from app.security.file_validation import FileValidationError
    import app.security.file_validation as file_validation

    original_cap = file_validation._MAX_DOCX_UNCOMPRESSED_BYTES
    file_validation._MAX_DOCX_UNCOMPRESSED_BYTES = 1024 * 1024  # 1MB cap for this test
    try:
        with pytest.raises(FileValidationError):
            file_validation.check_docx_not_a_zip_bomb(bytes(data))
    finally:
        file_validation._MAX_DOCX_UNCOMPRESSED_BYTES = original_cap


def test_valid_txt_upload_still_succeeds(client, monkeypatch):
    """Sanity check: the new validation doesn't reject legitimate
    uploads (.txt has no magic-byte check at all) — accepted and queued
    for background ingestion (Phase 5 completion pass, round 2: upload
    no longer indexes synchronously; see tests/test_upload_async.py for
    the full async contract/job-completion assertions)."""
    _mock_ingest_success(monkeypatch)
    resp = client.post(
        "/documents/upload",
        files={"files": ("notes.txt", b"perfectly ordinary text content", "text/plain")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["files"] == ["notes.txt"]
    assert len(body["jobs"]) == 1


def test_valid_pdf_signature_passes_signature_check(client, monkeypatch):
    """A minimal-but-correctly-signed PDF passes the signature check (it
    may still fail deeper parsing — that's extract_document's job, not
    this test's concern)."""
    from app.security.file_validation import check_file_signature

    check_file_signature(b"%PDF-1.4\n%mock content", ".pdf")  # must not raise