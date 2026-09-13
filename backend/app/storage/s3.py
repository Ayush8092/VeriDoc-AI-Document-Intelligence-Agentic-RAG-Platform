"""S3-compatible `StorageBackend`, via boto3.

Works against real AWS S3 or any S3-compatible endpoint (MinIO,
Cloudflare R2, DigitalOcean Spaces, ...) by setting `S3_ENDPOINT_URL`.
`boto3` is an optional dependency (like `sentence-transformers`/`opik`
elsewhere in this project — see `app/rag/cross_encoder.py`,
`app/observability/opik_integration.py`): importing this module without
`boto3` installed raises a clear `ImportError`-derived message at
construction time, not at import time, so `app/storage/base.py::
get_backend` can still be imported freely by code that only ever uses
the local backend.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterator

from app.storage.base import StorageBackend, StorageError


class S3StorageBackend(StorageBackend):
    def __init__(
        self,
        bucket: str | None,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        if not bucket:
            raise ValueError("S3_BUCKET must be set to use STORAGE_BACKEND=s3")
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise ImportError(
                "STORAGE_BACKEND=s3 requires the optional 'boto3' dependency. "
                "Install it with: pip install boto3"
            ) from exc

        self._ClientError = ClientError
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
        )

    def _is_not_found(self, exc) -> bool:
        code = exc.response.get("Error", {}).get("Code", "")
        return code in ("404", "NoSuchKey", "NotFound")

    def save(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        except self._ClientError as exc:
            raise StorageError(f"failed to save {key!r} to s3://{self.bucket}: {exc}") from exc

    def read(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except self._ClientError as exc:
            if self._is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise StorageError(f"failed to read {key!r} from s3://{self.bucket}: {exc}") from exc

    def stream(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
        except self._ClientError as exc:
            if self._is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise StorageError(f"failed to stream {key!r} from s3://{self.bucket}: {exc}") from exc

        body = resp["Body"]

        def _iter():
            while chunk := body.read(chunk_size):
                yield chunk

        return _iter()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._ClientError as exc:
            if self._is_not_found(exc):
                return False
            raise StorageError(f"failed to check existence of {key!r} in s3://{self.bucket}: {exc}") from exc

    def delete(self, key: str) -> None:
        # S3 DeleteObject is already idempotent (no error for a missing
        # key) — matches StorageBackend.delete's documented contract with
        # no extra handling needed.
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except self._ClientError as exc:
            raise StorageError(f"failed to delete {key!r} from s3://{self.bucket}: {exc}") from exc

    def move(self, source_key: str, dest_key: str) -> None:
        self.copy(source_key, dest_key)
        self.delete(source_key)

    def copy(self, source_key: str, dest_key: str) -> None:
        if not self.exists(source_key):
            raise FileNotFoundError(source_key)
        try:
            self._client.copy_object(
                Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": source_key}, Key=dest_key
            )
        except self._ClientError as exc:
            raise StorageError(f"failed to copy {source_key!r} to {dest_key!r} in s3://{self.bucket}: {exc}") from exc

    def get_local_path(self, key: str) -> str | None:
        """S3 objects have no local path — downloads `key` to a fresh
        temp file and returns THAT path. Unlike `LocalStorageBackend`,
        this temp file is a copy, not the source of truth: the CALLER
        owns cleaning it up via `release_local_path` (or, preferably, the
        `app.storage.base.local_path()` context manager, which calls it
        automatically) — this is exactly the "escape hatch" tradeoff
        `StorageBackend.get_local_path`'s docstring describes. Raises
        `FileNotFoundError` if `key` does not exist, same as `read`.
        """
        data = self.read(key)  # raises FileNotFoundError itself if missing
        suffix = os.path.splitext(key)[1]
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="veridoc-s3-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except BaseException:
            os.unlink(tmp_path)
            raise
        return tmp_path

    def release_local_path(self, path: str) -> None:
        """Delete the temp file `get_local_path` created. Safe to call
        more than once or on a path that's already gone (mirrors
        `StorageBackend.delete`'s idempotency contract) — a caller
        racing a retry/cleanup path should never crash on double-release.
        """
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
        except self._ClientError as exc:
            raise StorageError(f"failed to list keys under {prefix!r} in s3://{self.bucket}: {exc}") from exc
        return keys

    def size(self, key: str) -> int:
        try:
            resp = self._client.head_object(Bucket=self.bucket, Key=key)
            return int(resp["ContentLength"])
        except self._ClientError as exc:
            if self._is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise StorageError(f"failed to get size of {key!r} in s3://{self.bucket}: {exc}") from exc