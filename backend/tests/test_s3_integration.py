"""Real S3 integration tests (Phase 5 completion pass, item 3 — "Do not
simply mock the entire S3 implementation and claim S3 is complete...
Prefer MinIO / another S3-compatible local service").

This is a DIFFERENT, COMPLEMENTARY tier from
`tests/test_storage_minio_integration.py`, not a replacement for it:

  - `tests/test_storage.py`'s `S3StorageBackend` tests use a fully
    hand-written fake boto3 client — exactly the "mock the entire S3
    implementation" this item warns against on its own.
  - THIS file drives the REAL `boto3` client, inside the REAL
    `S3StorageBackend`, against `ThreadedMotoServer` (from the `moto`
    library, an existing test dependency) — a genuine in-process HTTP
    server that speaks the real S3 API surface. No boto3 method is
    patched/mocked anywhere in this file. Its advantage over the MinIO
    file below: it needs no external service or env var, so it runs
    unconditionally in this sandbox and in CI, catching real boto3
    request-signing/serialization bugs (like the credentials bug this
    file itself had until this pass — see below) that a hand-mock could
    never catch.
  - `tests/test_storage_minio_integration.py` is the OPT-IN tier against
    a genuine external MinIO/AWS S3 endpoint (`VERIDOC_TEST_S3_ENDPOINT_URL`),
    for verifying against a real, independently-implemented S3-compatible
    server rather than moto's (also real, but Python-based) implementation
    of the S3 API. Keep both; they test different things.

**Bug fixed this pass**: every test in this file was failing with
`botocore.exceptions.NoCredentialsError` — `S3StorageBackend` was
constructed without `access_key`/`secret_key`, so boto3 fell back to its
default credential discovery chain (environment variables, `~/.aws/
credentials`, instance metadata, ...), found nothing in this sandboxed
environment, and refused to sign requests even though `moto_server`
itself doesn't validate credentials at all. Fixed by passing explicit
dummy credentials, same as `test_storage_minio_integration.py`'s
`minioadmin`/`minioadmin` default already does for the same reason.
"""

from __future__ import annotations

import os

import pytest

boto3 = pytest.importorskip("boto3")

from app.storage.base import StorageError, local_path  # noqa: E402
from app.storage.s3 import S3StorageBackend  # noqa: E402

_DUMMY_ACCESS_KEY = "moto-test-access-key"
_DUMMY_SECRET_KEY = "moto-test-secret-key"


@pytest.fixture(scope="session")
def moto_server():
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=0)
    server.start()
    port = server._server.server_port
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture()
def s3_backend(moto_server, request):
    # A distinct bucket per test (named after the test itself) rather
    # than per-key namespacing within one shared bucket — closer to how
    # a real deployment isolates buckets, and means two tests can never
    # interfere with each other even if run out of order.
    bucket = f"veridoc-test-{request.node.name}".lower().replace("[", "-").replace("]", "-")[:63]
    backend = S3StorageBackend(
        bucket=bucket,
        endpoint_url=moto_server,
        region="us-east-1",
        access_key=_DUMMY_ACCESS_KEY,
        secret_key=_DUMMY_SECRET_KEY,
    )
    backend._client.create_bucket(Bucket=bucket)
    return backend


# ---------------------------------------------------------------------------
# Every StorageBackend operation, against a real S3-API HTTP server
# ---------------------------------------------------------------------------


def test_save_and_read_roundtrip(s3_backend):
    s3_backend.save("uploads/1/report.pdf", b"pdf bytes here")
    assert s3_backend.read("uploads/1/report.pdf") == b"pdf bytes here"


def test_save_overwrites_existing(s3_backend):
    s3_backend.save("file.txt", b"v1")
    s3_backend.save("file.txt", b"v2")
    assert s3_backend.read("file.txt") == b"v2"


def test_read_missing_key_raises_file_not_found(s3_backend):
    with pytest.raises(FileNotFoundError):
        s3_backend.read("does/not/exist.txt")


def test_exists(s3_backend):
    assert s3_backend.exists("file.txt") is False
    s3_backend.save("file.txt", b"data")
    assert s3_backend.exists("file.txt") is True


def test_delete_is_idempotent(s3_backend):
    s3_backend.save("file.txt", b"data")
    s3_backend.delete("file.txt")
    assert not s3_backend.exists("file.txt")
    s3_backend.delete("file.txt")  # must not raise


def test_stream_yields_full_content_in_chunks(s3_backend):
    content = b"x" * 50_000
    s3_backend.save("big.bin", content)
    chunks = list(s3_backend.stream("big.bin", chunk_size=8192))
    assert len(chunks) > 1
    assert b"".join(chunks) == content


def test_stream_missing_key_raises(s3_backend):
    with pytest.raises(FileNotFoundError):
        list(s3_backend.stream("nope.bin"))


def test_move(s3_backend):
    s3_backend.save("src.txt", b"data")
    s3_backend.move("src.txt", "dst.txt")
    assert not s3_backend.exists("src.txt")
    assert s3_backend.read("dst.txt") == b"data"


def test_move_missing_source_raises(s3_backend):
    with pytest.raises(FileNotFoundError):
        s3_backend.move("nope.txt", "dst.txt")


def test_copy_preserves_source(s3_backend):
    s3_backend.save("src.txt", b"data")
    s3_backend.copy("src.txt", "dst.txt")
    assert s3_backend.read("src.txt") == b"data"
    assert s3_backend.read("dst.txt") == b"data"


def test_list_keys_returns_all_under_prefix(s3_backend):
    s3_backend.save("a/1.txt", b"1")
    s3_backend.save("a/2.txt", b"2")
    s3_backend.save("b/3.txt", b"3")
    assert set(s3_backend.list_keys("a")) == {"a/1.txt", "a/2.txt"}
    assert set(s3_backend.list_keys()) == {"a/1.txt", "a/2.txt", "b/3.txt"}


def test_size(s3_backend):
    s3_backend.save("file.txt", b"12345")
    assert s3_backend.size("file.txt") == 5


def test_size_missing_key_raises(s3_backend):
    with pytest.raises(FileNotFoundError):
        s3_backend.size("nope.txt")


# ---------------------------------------------------------------------------
# get_local_path / release_local_path: real temp-file lifecycle
# ---------------------------------------------------------------------------


def test_get_local_path_downloads_real_bytes_to_a_real_temp_file(s3_backend):
    s3_backend.save("report.pdf", b"%PDF-1.4 fake pdf content")
    path = s3_backend.get_local_path("report.pdf")
    try:
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == b"%PDF-1.4 fake pdf content"
        # The temp file must NOT be the literal storage key/path — it's a
        # genuinely separate local copy.
        assert "report.pdf" not in path or path.endswith(".pdf")
    finally:
        s3_backend.release_local_path(path)


def test_release_local_path_actually_deletes_the_temp_file(s3_backend):
    s3_backend.save("report.pdf", b"data")
    path = s3_backend.get_local_path("report.pdf")
    assert os.path.exists(path)
    s3_backend.release_local_path(path)
    assert not os.path.exists(path)


def test_release_local_path_is_idempotent(s3_backend):
    s3_backend.save("report.pdf", b"data")
    path = s3_backend.get_local_path("report.pdf")
    s3_backend.release_local_path(path)
    s3_backend.release_local_path(path)  # must not raise


def test_local_path_context_manager_cleans_up_on_success(s3_backend):
    s3_backend.save("report.pdf", b"data")
    with local_path(s3_backend, "report.pdf") as path:
        assert os.path.exists(path)
        captured_path = path
    assert not os.path.exists(captured_path)


def test_local_path_context_manager_cleans_up_on_exception(s3_backend):
    s3_backend.save("report.pdf", b"data")
    captured_path = None
    with pytest.raises(ValueError):
        with local_path(s3_backend, "report.pdf") as path:
            captured_path = path
            raise ValueError("simulated parse failure")
    assert not os.path.exists(captured_path)


def test_get_local_path_missing_key_raises_without_leaving_a_temp_file(s3_backend, tmp_path):
    import glob
    import tempfile

    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "veridoc-s3-*")))
    with pytest.raises(FileNotFoundError):
        s3_backend.get_local_path("nope.pdf")
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "veridoc-s3-*")))
    assert after == before  # no leaked temp file for a download that never happened


# ---------------------------------------------------------------------------
# Tenant key collisions / path safety — through the REAL ingestion key
# construction, against the REAL S3-compatible server
# ---------------------------------------------------------------------------


def test_tenant_specific_keys_cannot_collide_between_users(s3_backend):
    """Two different users' identically-named uploads must land at
    distinct keys and never overwrite each other — the same property
    tests/test_tenant_isolation.py already verifies against
    LocalStorageBackend, re-verified here against the real S3 API."""
    from app.api.documents import _upload_key

    key_user_1 = _upload_key("report.pdf", owner_id=1)
    key_user_2 = _upload_key("report.pdf", owner_id=2)
    assert key_user_1 != key_user_2

    s3_backend.save(key_user_1, b"user 1's content")
    s3_backend.save(key_user_2, b"user 2's content")

    assert s3_backend.read(key_user_1) == b"user 1's content"
    assert s3_backend.read(key_user_2) == b"user 2's content"


def test_s3_key_traversal_via_dotdot_does_not_escape_the_bucket_namespace(s3_backend):
    """S3 has no filesystem to escape (unlike LocalStorageBackend, which
    defends against this explicitly — see test_storage.py), but a
    `../`-laden key must still behave as an ordinary, harmless key
    within THIS backend's own bucket — it must never be interpreted as
    a relative-path escape by anything downstream that later calls
    get_local_path() and hands the result to a filesystem-based parser.
    """
    key = "../../etc/passwd"
    s3_backend.save(key, b"not actually /etc/passwd")
    assert s3_backend.read(key) == b"not actually /etc/passwd"

    # Critically: downloading it must produce an ordinary temp file
    # inside the system temp directory — never anything that resolves
    # outside it.
    path = s3_backend.get_local_path(key)
    try:
        import tempfile

        real_tmp_dir = os.path.realpath(tempfile.gettempdir())
        assert os.path.realpath(path).startswith(real_tmp_dir)
    finally:
        s3_backend.release_local_path(path)


# ---------------------------------------------------------------------------
# End-to-end: real ingestion through S3StorageBackend
# ---------------------------------------------------------------------------


def test_ingestion_service_scans_and_parses_files_via_real_s3_backend(s3_backend, tmp_path, monkeypatch):
    """The actual gap this whole item exists to close: does
    ingestion_service.py's storage-backend-based scanning
    (app/services/ingestion_service.py::_list_source_files/
    _chunk_source_root) work end-to-end against a REAL S3-compatible
    service, not just against LocalStorageBackend? Uploads two real
    files, points ingestion directly at this S3 backend, and confirms
    they're actually parsed and chunked.
    """
    from app.core.config import get_settings
    from app.services.ingestion_service import _chunk_source_root

    s3_backend.save("corpus/note.txt", b"This is a plain text corpus document about storage testing.")
    s3_backend.save("corpus/other.md", b"# Heading\n\nSome markdown content for the second document.")

    settings = get_settings()
    chunks = _chunk_source_root(s3_backend, "corpus", "corpus", settings, session=None)

    assert len(chunks) >= 2
    filenames = {c.source_file for c in chunks}
    assert "note.txt" in filenames
    assert "other.md" in filenames