"""Real S3-compatible integration tests for `S3StorageBackend`, against
an actual running MinIO (or any other S3-compatible) endpoint — NOT the
hand-written fake boto3 client `tests/test_storage.py` uses.

Phase 5 completion pass (final round), item 3: "Do not simply mock the
entire S3 implementation and claim S3 is complete... Prefer MinIO for
local automated integration testing if appropriate."

**Every test in this file is skipped, not failed,** when its
prerequisites aren't available: `boto3` not installed, or no reachable
S3-compatible endpoint configured via `VERIDOC_TEST_S3_ENDPOINT_URL`.
This is deliberate — this sandbox this file was authored in has neither
`boto3` nor network access to run a MinIO container, so these tests were
written but NOT executed here (see the Phase 5 completion report for the
honest "implemented, not verified in this environment" distinction).
They are written to be genuinely meaningful the first time they DO run
against a real endpoint, not decorative.

To actually run these:

    docker run -p 9000:9000 -p 9001:9001 \\
        -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \\
        minio/minio server /data --console-address ":9001"

    # then, in another shell (create the test bucket once):
    pip install boto3
    python -c "
import boto3
c = boto3.client('s3', endpoint_url='http://localhost:9000',
                  aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin')
c.create_bucket(Bucket='veridoc-test')
"

    VERIDOC_TEST_S3_ENDPOINT_URL=http://localhost:9000 \\
    VERIDOC_TEST_S3_BUCKET=veridoc-test \\
    VERIDOC_TEST_S3_ACCESS_KEY=minioadmin \\
    VERIDOC_TEST_S3_SECRET_KEY=minioadmin \\
        pytest tests/test_storage_minio_integration.py -v

`docker-compose.yml`'s `minio` profile (see that file) starts exactly
this, with these exact credentials/bucket, for local development.
"""

from __future__ import annotations

import os
import uuid

import pytest

boto3 = pytest.importorskip("boto3", reason="boto3 is an optional dependency for STORAGE_BACKEND=s3")

from app.storage.s3 import S3StorageBackend  # noqa: E402

_ENDPOINT_URL = os.environ.get("VERIDOC_TEST_S3_ENDPOINT_URL")
_BUCKET = os.environ.get("VERIDOC_TEST_S3_BUCKET", "veridoc-test")
_ACCESS_KEY = os.environ.get("VERIDOC_TEST_S3_ACCESS_KEY", "minioadmin")
_SECRET_KEY = os.environ.get("VERIDOC_TEST_S3_SECRET_KEY", "minioadmin")

pytestmark = pytest.mark.skipif(
    not _ENDPOINT_URL,
    reason=(
        "VERIDOC_TEST_S3_ENDPOINT_URL not set — no S3-compatible endpoint configured for "
        "integration testing. See this module's docstring to run a local MinIO."
    ),
)


@pytest.fixture
def backend():
    """A real `S3StorageBackend` against the configured MinIO/S3-compatible
    endpoint. Every test below uses a UUID-prefixed key so parallel test
    runs (or a stale key from a previous crashed run) can never collide
    or interfere with each other — the bucket is never wiped/reset by
    this fixture, only individual test keys are cleaned up.
    """
    return S3StorageBackend(
        bucket=_BUCKET,
        endpoint_url=_ENDPOINT_URL,
        access_key=_ACCESS_KEY,
        secret_key=_SECRET_KEY,
    )


def _key(name: str) -> str:
    return f"integration-test/{uuid.uuid4().hex}/{name}"


def test_save_then_read_round_trips_against_real_endpoint(backend):
    key = _key("hello.txt")
    backend.save(key, b"hello from a real S3-compatible endpoint")
    try:
        assert backend.read(key) == b"hello from a real S3-compatible endpoint"
    finally:
        backend.delete(key)


def test_exists_reflects_real_bucket_state(backend):
    key = _key("exists.txt")
    assert backend.exists(key) is False
    backend.save(key, b"x")
    try:
        assert backend.exists(key) is True
    finally:
        backend.delete(key)


def test_delete_then_exists_is_false(backend):
    key = _key("delete-me.txt")
    backend.save(key, b"x")
    backend.delete(key)
    assert backend.exists(key) is False


def test_delete_is_idempotent_against_real_endpoint(backend):
    key = _key("never-existed.txt")
    backend.delete(key)  # must not raise
    backend.delete(key)  # still must not raise


def test_move_relocates_object_on_real_endpoint(backend):
    src, dest = _key("src.txt"), _key("dest.txt")
    backend.save(src, b"payload")
    try:
        backend.move(src, dest)
        assert backend.exists(src) is False
        assert backend.read(dest) == b"payload"
    finally:
        backend.delete(src)
        backend.delete(dest)


def test_copy_leaves_source_intact_on_real_endpoint(backend):
    src, dest = _key("src.txt"), _key("copy.txt")
    backend.save(src, b"payload")
    try:
        backend.copy(src, dest)
        assert backend.read(src) == b"payload"
        assert backend.read(dest) == b"payload"
    finally:
        backend.delete(src)
        backend.delete(dest)


def test_list_keys_filters_by_prefix_on_real_endpoint(backend):
    prefix = _key("listing")
    try:
        backend.save(f"{prefix}/a.txt", b"a")
        backend.save(f"{prefix}/b.txt", b"b")
        backend.save(f"{prefix}-unrelated.txt", b"c")  # deliberately NOT under prefix + "/"
        keys = set(backend.list_keys(f"{prefix}/"))
        assert keys == {f"{prefix}/a.txt", f"{prefix}/b.txt"}
    finally:
        backend.delete(f"{prefix}/a.txt")
        backend.delete(f"{prefix}/b.txt")
        backend.delete(f"{prefix}-unrelated.txt")


def test_size_matches_uploaded_byte_count_on_real_endpoint(backend):
    key = _key("sized.bin")
    payload = os.urandom(4096)
    backend.save(key, payload)
    try:
        assert backend.size(key) == 4096
    finally:
        backend.delete(key)


def test_get_local_path_downloads_a_real_temp_file_that_is_cleaned_up(backend):
    """Phase 5 completion pass, item 3: "Ensure temporary local files
    created from S3 objects are always cleaned up." — the one thing the
    fake-client unit tests literally cannot verify (no real filesystem
    round trip happens against a fake client), and the primary reason
    this integration file exists.
    """
    key = _key("for-local-path.bin")
    payload = os.urandom(1024)
    backend.save(key, payload)
    local_path = None
    try:
        local_path = backend.get_local_path(key)
        assert local_path is not None
        assert os.path.exists(local_path)
        with open(local_path, "rb") as f:
            assert f.read() == payload
    finally:
        if local_path is not None:
            backend.release_local_path(local_path)
            assert not os.path.exists(local_path), "release_local_path must delete the downloaded temp file"
        backend.delete(key)


def test_read_missing_key_raises_file_not_found_on_real_endpoint(backend):
    with pytest.raises(FileNotFoundError):
        backend.read(_key("does-not-exist.txt"))


def test_path_traversal_style_key_is_stored_literally_not_escaped(backend):
    """S3 has no concept of ".." path resolution the way a local
    filesystem does — a key containing "../" is just an unusual key
    name, not a traversal outside the bucket. This test exists to
    confirm that assumption holds against a REAL endpoint (unlike the
    local backend, which must actively defend against this — see
    `tests/test_storage.py::test_path_traversal_is_rejected` — S3 has no
    equivalent vulnerability to defend against by construction, but this
    is worth confirming rather than assuming).
    """
    key = _key("../../etc/passwd")
    backend.save(key, b"not actually a traversal on object storage")
    try:
        assert backend.read(key) == b"not actually a traversal on object storage"
    finally:
        backend.delete(key)
