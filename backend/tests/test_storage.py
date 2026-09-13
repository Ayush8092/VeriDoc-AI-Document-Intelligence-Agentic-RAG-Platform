"""Tests for `app.storage.{base,local,s3}` (Phase 5 completion pass, round
2, item 22 — this file was referenced by name in
`app/storage/base.py`'s module docstring and `validate_storage_config`
since before it existed; this closes that gap for real).

**`LocalStorageBackend`**: tested against a real temp directory
(`tmp_path`) — no mocking, genuine filesystem behavior.

**`S3StorageBackend`**: neither `boto3` nor `moto` are installed in the
environment this was authored in (no network access to install them —
see `app/storage/base.py`'s module docstring). Rather than skip S3
coverage entirely, `_FakeBoto3Client`/`_FakeClientError` below are a
minimal, hand-rolled stand-in for exactly the boto3 API surface
`app/storage/s3.py` calls (`put_object`, `get_object`, `head_object`,
`delete_object`, `copy_object`, `get_paginator("list_objects_v2")`).
Since `app/storage/s3.py` imports `boto3`/`botocore.exceptions.ClientError`
LAZILY (inside `S3StorageBackend.__init__`, not at module level — see
that file), injecting fake `boto3`/`botocore` modules into `sys.modules`
before constructing `S3StorageBackend` is sufficient to exercise the
REAL `S3StorageBackend` code — every line of `save`/`read`/`exists`/
`delete`/`move`/`copy`/`get_local_path`/`list_keys`/`size` in that file
actually runs.

**What this does and does NOT prove**: this verifies `S3StorageBackend`
calls boto3's API correctly and handles its documented error shapes
(`ClientError` with a `404`/`NoSuchKey`/`NotFound` response code) the
way `StorageBackend`'s contract requires. It does NOT verify boto3
itself, real AWS S3/MinIO/R2 semantics, network error handling, request
signing, or credential resolution — that requires a real or `moto`-mocked
S3 endpoint, neither of which was reachable here. Treat a first real
deployment against S3 as this backend's true first end-to-end test.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.storage.base import StorageBackend, local_path
from app.storage.local import LocalStorageBackend


# =============================================================================
# LocalStorageBackend — real filesystem, fully executable
# =============================================================================


@pytest.fixture()
def backend(tmp_path) -> LocalStorageBackend:
    return LocalStorageBackend(root=tmp_path / "storage-root")


def test_save_then_read_round_trips(backend):
    backend.save("uploads/report.pdf", b"hello world")
    assert backend.read("uploads/report.pdf") == b"hello world"


def test_read_missing_key_raises_file_not_found(backend):
    with pytest.raises(FileNotFoundError):
        backend.read("uploads/does-not-exist.pdf")


def test_save_overwrites_existing_key(backend):
    backend.save("uploads/report.pdf", b"version 1")
    backend.save("uploads/report.pdf", b"version 2")
    assert backend.read("uploads/report.pdf") == b"version 2"


def test_save_creates_parent_directories(backend):
    backend.save("uploads/42/nested/report.pdf", b"data")
    assert backend.read("uploads/42/nested/report.pdf") == b"data"


def test_exists_true_for_present_key_false_for_missing(backend):
    backend.save("uploads/a.txt", b"x")
    assert backend.exists("uploads/a.txt") is True
    assert backend.exists("uploads/b.txt") is False


def test_delete_removes_key(backend):
    backend.save("uploads/a.txt", b"x")
    backend.delete("uploads/a.txt")
    assert backend.exists("uploads/a.txt") is False


def test_delete_is_idempotent_for_missing_key(backend):
    backend.delete("uploads/never-existed.txt")  # must not raise


def test_move_relocates_and_removes_source(backend):
    backend.save("uploads/old.txt", b"content")
    backend.move("uploads/old.txt", "uploads/new.txt")
    assert backend.exists("uploads/old.txt") is False
    assert backend.read("uploads/new.txt") == b"content"


def test_move_missing_source_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.move("uploads/missing.txt", "uploads/dest.txt")


def test_move_overwrites_existing_dest(backend):
    backend.save("uploads/old.txt", b"new content")
    backend.save("uploads/dest.txt", b"stale content")
    backend.move("uploads/old.txt", "uploads/dest.txt")
    assert backend.read("uploads/dest.txt") == b"new content"


def test_copy_leaves_source_intact(backend):
    backend.save("uploads/original.txt", b"content")
    backend.copy("uploads/original.txt", "uploads/copy.txt")
    assert backend.read("uploads/original.txt") == b"content"
    assert backend.read("uploads/copy.txt") == b"content"


def test_copy_missing_source_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.copy("uploads/missing.txt", "uploads/dest.txt")


def test_list_keys_lists_everything_under_prefix(backend):
    backend.save("uploads/1/a.txt", b"a")
    backend.save("uploads/1/b.txt", b"b")
    backend.save("uploads/2/c.txt", b"c")
    backend.save("corpus/d.txt", b"d")

    upload_keys = set(backend.list_keys("uploads"))
    assert upload_keys == {"uploads/1/a.txt", "uploads/1/b.txt", "uploads/2/c.txt"}

    all_keys = set(backend.list_keys())
    assert all_keys == {"uploads/1/a.txt", "uploads/1/b.txt", "uploads/2/c.txt", "corpus/d.txt"}


def test_list_keys_empty_prefix_directory_returns_empty_list(backend):
    assert backend.list_keys("nonexistent-prefix") == []


def test_size_returns_byte_count(backend):
    backend.save("uploads/a.txt", b"12345")
    assert backend.size("uploads/a.txt") == 5


def test_size_missing_key_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.size("uploads/missing.txt")


def test_get_local_path_returns_a_real_path_to_the_actual_file(backend, tmp_path):
    backend.save("uploads/a.txt", b"content")
    path = backend.get_local_path("uploads/a.txt")
    assert path is not None
    from pathlib import Path

    assert Path(path).read_bytes() == b"content"


def test_release_local_path_is_a_noop_for_local_backend(backend):
    """The REAL file must survive `release_local_path` for
    `LocalStorageBackend` — see `StorageBackend.release_local_path`'s
    docstring: unlike S3's temp-file download, the local backend's
    `get_local_path` IS the stored file, so "releasing" it must never
    delete actual data.
    """
    backend.save("uploads/a.txt", b"content")
    path = backend.get_local_path("uploads/a.txt")
    backend.release_local_path(path)
    assert backend.read("uploads/a.txt") == b"content"


def test_path_traversal_is_rejected(backend):
    """Defense in depth (see `LocalStorageBackend._resolve`'s docstring)
    — a key that would resolve outside the storage root must be
    rejected, not silently escape it.
    """
    with pytest.raises(ValueError):
        backend.save("../../etc/passwd", b"malicious")


def test_path_traversal_via_exists_returns_false_not_raise(backend):
    """`exists()` specifically swallows the traversal ValueError and
    returns False — matches `StorageBackend.exists`'s documented
    contract ("never raises for a missing key... only raises for a
    genuine backend failure") and `LocalStorageBackend.exists`'s own
    `except ValueError: return False`.
    """
    assert backend.exists("../../etc/passwd") is False


def test_local_path_context_manager_yields_real_path_and_does_not_delete_it(backend):
    backend.save("uploads/a.txt", b"content")
    with local_path(backend, "uploads/a.txt") as path:
        from pathlib import Path

        assert Path(path).read_bytes() == b"content"
    # Still readable after the context exits (local backend: release is
    # a no-op, per test_release_local_path_is_a_noop_for_local_backend).
    assert backend.read("uploads/a.txt") == b"content"


def test_stream_yields_full_content_in_chunks(backend):
    backend.save("uploads/a.txt", b"0123456789")
    chunks = list(backend.stream("uploads/a.txt", chunk_size=4))
    assert b"".join(chunks) == b"0123456789"
    assert len(chunks) == 3  # 4 + 4 + 2


def test_stream_missing_key_raises_before_yielding(backend):
    with pytest.raises(FileNotFoundError):
        list(backend.stream("uploads/missing.txt"))


# =============================================================================
# S3StorageBackend — real code, hand-rolled fake boto3 (see module docstring)
# =============================================================================


class _FakeClientError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            chunk, self._pos = self._data[self._pos :], len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def paginate(self, Bucket: str, Prefix: str = ""):  # noqa: N803 - matches boto3's param casing
        contents = [{"Key": k} for k in sorted(self._objects) if k.startswith(Prefix)]
        yield {"Contents": contents}


class _FakeS3Client:
    """Minimal in-memory stand-in for exactly what `app/storage/s3.py`
    calls on a real boto3 S3 client. See module docstring for scope.
    """

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _FakeClientError("NoSuchKey")
        return {"Body": _FakeBody(self.objects[Key])}

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _FakeClientError("404")
        return {"ContentLength": len(self.objects[Key])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)

    def copy_object(self, Bucket, CopySource, Key):  # noqa: N803
        source_key = CopySource["Key"]
        if source_key not in self.objects:
            raise _FakeClientError("NoSuchKey")
        self.objects[Key] = self.objects[source_key]

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self.objects)


@pytest.fixture()
def fake_s3_backend(monkeypatch):
    """Injects fake `boto3`/`botocore.exceptions` modules into
    `sys.modules` so `S3StorageBackend.__init__`'s lazy `import boto3` /
    `from botocore.exceptions import ClientError` picks these up instead
    of failing with `ImportError` — see module docstring.
    """
    fake_client = _FakeS3Client()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service, **kwargs: fake_client  # noqa: ARG005

    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
    fake_botocore_exceptions.ClientError = _FakeClientError
    fake_botocore.exceptions = fake_botocore_exceptions

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exceptions)

    from app.storage.s3 import S3StorageBackend

    backend = S3StorageBackend(bucket="test-bucket")
    backend._fake_client = fake_client  # for assertions that want to inspect state directly
    return backend


def test_s3_save_then_read_round_trips(fake_s3_backend):
    fake_s3_backend.save("uploads/report.pdf", b"hello s3")
    assert fake_s3_backend.read("uploads/report.pdf") == b"hello s3"


def test_s3_read_missing_key_raises_file_not_found(fake_s3_backend):
    with pytest.raises(FileNotFoundError):
        fake_s3_backend.read("uploads/missing.pdf")


def test_s3_exists_true_and_false(fake_s3_backend):
    fake_s3_backend.save("uploads/a.txt", b"x")
    assert fake_s3_backend.exists("uploads/a.txt") is True
    assert fake_s3_backend.exists("uploads/b.txt") is False


def test_s3_delete_is_idempotent(fake_s3_backend):
    fake_s3_backend.save("uploads/a.txt", b"x")
    fake_s3_backend.delete("uploads/a.txt")
    fake_s3_backend.delete("uploads/a.txt")  # second delete must not raise
    assert fake_s3_backend.exists("uploads/a.txt") is False


def test_s3_move_copies_then_deletes_source(fake_s3_backend):
    fake_s3_backend.save("uploads/old.txt", b"content")
    fake_s3_backend.move("uploads/old.txt", "uploads/new.txt")
    assert fake_s3_backend.exists("uploads/old.txt") is False
    assert fake_s3_backend.read("uploads/new.txt") == b"content"


def test_s3_copy_leaves_source_intact(fake_s3_backend):
    fake_s3_backend.save("uploads/original.txt", b"content")
    fake_s3_backend.copy("uploads/original.txt", "uploads/copy.txt")
    assert fake_s3_backend.read("uploads/original.txt") == b"content"
    assert fake_s3_backend.read("uploads/copy.txt") == b"content"


def test_s3_copy_missing_source_raises(fake_s3_backend):
    with pytest.raises(FileNotFoundError):
        fake_s3_backend.copy("uploads/missing.txt", "uploads/dest.txt")


def test_s3_get_local_path_downloads_to_a_real_temp_file(fake_s3_backend):
    fake_s3_backend.save("uploads/a.txt", b"content")
    path = fake_s3_backend.get_local_path("uploads/a.txt")
    assert path is not None
    from pathlib import Path

    assert Path(path).read_bytes() == b"content"
    fake_s3_backend.release_local_path(path)
    assert not Path(path).exists()  # release actually deletes the temp copy, unlike local backend


def test_s3_release_local_path_is_idempotent(fake_s3_backend):
    fake_s3_backend.save("uploads/a.txt", b"content")
    path = fake_s3_backend.get_local_path("uploads/a.txt")
    fake_s3_backend.release_local_path(path)
    fake_s3_backend.release_local_path(path)  # second release must not raise


def test_s3_list_keys_filters_by_prefix(fake_s3_backend):
    fake_s3_backend.save("uploads/1/a.txt", b"a")
    fake_s3_backend.save("uploads/2/b.txt", b"b")
    fake_s3_backend.save("corpus/c.txt", b"c")

    assert set(fake_s3_backend.list_keys("uploads/")) == {"uploads/1/a.txt", "uploads/2/b.txt"}
    assert set(fake_s3_backend.list_keys("corpus/")) == {"corpus/c.txt"}


def test_s3_size_returns_byte_count(fake_s3_backend):
    fake_s3_backend.save("uploads/a.txt", b"12345")
    assert fake_s3_backend.size("uploads/a.txt") == 5


def test_s3_size_missing_key_raises(fake_s3_backend):
    with pytest.raises(FileNotFoundError):
        fake_s3_backend.size("uploads/missing.txt")


def test_s3_local_path_context_manager_cleans_up_temp_file(fake_s3_backend):
    fake_s3_backend.save("uploads/a.txt", b"content")
    captured_path = None
    with local_path(fake_s3_backend, "uploads/a.txt") as path:
        captured_path = path
        from pathlib import Path

        assert Path(path).read_bytes() == b"content"
    from pathlib import Path

    assert not Path(captured_path).exists()  # cleaned up on context exit, unlike the local backend


def test_s3_backend_requires_bucket():
    from app.storage.s3 import S3StorageBackend

    with pytest.raises(ValueError):
        S3StorageBackend(bucket=None)


def test_s3_backend_raises_clear_error_without_boto3_installed(monkeypatch):
    """If `boto3` genuinely isn't importable, constructing
    `S3StorageBackend` must raise a clear `ImportError` naming the
    missing dependency and how to install it — not a bare
    `ModuleNotFoundError` several frames deep.

    `boto3` is, in fact, installed in this project's normal dev/test
    environment (it's in `requirements.txt`, unconditionally — see
    `app/storage/s3.py`'s module docstring for why the *code* still
    treats it as an optional, lazily-imported dependency regardless:
    `STORAGE_BACKEND=s3` might be selected in an environment that never
    installed the S3 extras, and `get_backend`/every LOCAL-backend code
    path must keep working either way). Merely deleting `sys.modules`
    entries (`monkeypatch.delitem`) does NOT simulate "not installed" —
    it only clears the import cache, so Python happily re-imports the
    still-present package from disk and the test would falsely pass
    regardless of whether the ImportError path works at all. Setting the
    module name to `None` in `sys.modules` is the standard idiom that
    actually forces the next `import boto3` to raise `ImportError`,
    exactly like a genuinely-missing package would.
    """
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setitem(sys.modules, "botocore", None)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", None)

    from app.storage.s3 import S3StorageBackend

    with pytest.raises(ImportError, match="boto3"):
        S3StorageBackend(bucket="test-bucket")


# =============================================================================
# StorageBackend is a real abstract base — both implementations satisfy it
# =============================================================================


def test_local_backend_is_a_storage_backend(backend):
    assert isinstance(backend, StorageBackend)


def test_s3_backend_is_a_storage_backend(fake_s3_backend):
    assert isinstance(fake_s3_backend, StorageBackend)