"""Local filesystem `StorageBackend` — the default, and functionally
equivalent to what the codebase already does with `pathlib` directly
(see `app/storage/base.py`'s "Honest scope note" for what's NOT yet
routed through this).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

from app.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """Resolve `key` to a path under `self.root`, refusing to resolve
        OUTSIDE it. `key` is meant to be a trusted, already-sanitized
        relative path (the same contract `app/api/documents.py::
        _safe_filename` already enforces upstream for user-supplied
        filenames) — this check is defense in depth against a `key`
        containing `../` ever reaching here, not a substitute for
        sanitizing at the point a filename is first accepted from a
        request.
        """
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"storage key {key!r} resolves outside the storage root")
        return candidate

    def save(self, key: str, data: bytes) -> None:
        """Writes via a temp file in the SAME directory + `os.replace`
        (through `Path.write_bytes` on the temp file, then
        `Path.replace`), so a reader that opens `key` mid-write either
        sees the old complete contents or the new complete contents,
        never a partial write — atomic for THIS SINGLE FILE (a stronger,
        multi-file guarantee like the upload flow's backup-and-rollback
        is still the caller's job; see `StorageBackend.save`'s
        docstring).
        """
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{id(data)}")
        try:
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def stream(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(key)

        def _iter():
            with path.open("rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk

        return _iter()

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ValueError:
            return False

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def move(self, source_key: str, dest_key: str) -> None:
        source = self._resolve(source_key)
        dest = self._resolve(dest_key)
        if not source.is_file():
            raise FileNotFoundError(source_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.replace(dest)

    def copy(self, source_key: str, dest_key: str) -> None:
        source = self._resolve(source_key)
        dest = self._resolve(dest_key)
        if not source.is_file():
            raise FileNotFoundError(source_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    def get_local_path(self, key: str) -> str | None:
        path = self._resolve(key)
        # Deliberately does NOT check existence — callers that are about
        # to CREATE a file at this path (e.g. an OCR page-image cache
        # writer) need the path back too; `read`/`exists`/`stream` are
        # the existence-checked read operations.
        return str(path)

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._resolve(prefix) if prefix else self.root
        if base.is_file():
            return [str(base.relative_to(self.root)).replace("\\", "/")]
        if not base.is_dir():
            return []
        return [str(p.relative_to(self.root)).replace("\\", "/") for p in base.rglob("*") if p.is_file()]

    def size(self, key: str) -> int:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.stat().st_size
