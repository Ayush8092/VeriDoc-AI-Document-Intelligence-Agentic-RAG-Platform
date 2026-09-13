"""Storage abstraction (Phase 5 section 4).

This module defines the `StorageBackend` interface every file-touching
code path in this project depends on (`app/api/documents.py` — upload
persist/rollback; `app/api/source.py` — source serving + page-image
rendering; `app/services/ingestion_service.py` — corpus/upload directory
scanning + parsing), so the application doesn't need to know whether a
file lives on local disk or in S3-compatible object storage.

Two implementations ship:
  - `LocalStorageBackend` (`app/storage/local.py`) — a thin wrapper over
    `pathlib`, functionally identical to what the codebase did directly
    before this abstraction existed. Default backend; zero behavior
    change for local dev/tests.
  - `S3StorageBackend` (`app/storage/s3.py`) — boto3-based, works against
    AWS S3 or any S3-compatible endpoint (MinIO, Cloudflare R2, ...) via
    `S3_ENDPOINT_URL`.

Selected by `STORAGE_BACKEND` (`local` | `s3` — see
`app.core.config.Settings.storage_backend`) via `get_backend()` below.

**Integration status** (see the Phase 5 spec's "do not fabricate
results"): as of the Phase 5 completion pass (round 2), every
file-touching call site listed above routes through this abstraction —
`app/services/ingestion_service.py`'s corpus/upload scanning now uses
`list_keys()`/`local_path()` rather than `Path.iterdir()`, and
`app/api/source.py` handles both `source_root == "corpus"` and
`"uploads"` uniformly through `get_backend(settings)`. `validate_storage_config`
(below) therefore now ALLOWS `STORAGE_BACKEND=s3` — see its own
docstring for exactly what was re-audited and, importantly, what has
NOT been runtime-verified (no live/mocked S3 endpoint was reachable in
the environment this pass was authored in — `boto3` itself was not
installable there). `LocalStorageBackend` is independently tested
(`tests/test_storage.py`, real filesystem, fully executable); the same
file's `S3StorageBackend` tests are written against a hand-rolled fake
`boto3` client (since neither `boto3` nor `moto` are installed in this
sandbox) — see that file's module docstring for what that does and
doesn't prove.
"""

from __future__ import annotations

import abc
import contextlib
from dataclasses import dataclass
from typing import BinaryIO, Iterator


class StorageError(Exception):
    """Raised for a storage operation failure that isn't better expressed
    as a builtin exception (`FileNotFoundError`, etc.) — e.g. an
    S3-compatible provider returning an unexpected error."""


@dataclass(frozen=True)
class StorageObjectInfo:
    key: str
    size: int
    exists: bool = True


class StorageBackend(abc.ABC):
    """Content-addressed-by-key file storage. `key` is always a POSIX-style
    relative path (e.g. `"uploads/42/report.pdf"`) — never an absolute
    local filesystem path or a backend-specific URL; each implementation
    maps `key` onto whatever "a file" means for that backend (a path
    under a root directory for `LocalStorageBackend`, an object key
    within a bucket for `S3StorageBackend`).

    Every method's docstring states its exact failure behavior — a
    storage abstraction that's vague about "what happens if the file
    doesn't exist" just relocates bugs rather than preventing them.
    """

    @abc.abstractmethod
    def save(self, key: str, data: bytes) -> None:
        """Write `data` to `key`, creating/overwriting it. Creates any
        needed parent structure. Not guaranteed atomic on its own — see
        `LocalStorageBackend.save`'s docstring for what "atomic" means
        for that implementation specifically; callers that need the
        stronger cross-file atomicity `app/api/documents.py` implements
        today must keep doing so themselves (this method is a primitive,
        not a transaction)."""

    @abc.abstractmethod
    def read(self, key: str) -> bytes:
        """Return the full contents of `key`. Raises `FileNotFoundError`
        if `key` does not exist — never returns `None` or an empty
        `bytes` for a missing key, so a caller can't mistake "not found"
        for "empty file"."""

    @abc.abstractmethod
    def stream(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """Yield `key`'s contents in `chunk_size`-byte pieces, for a large
        file a caller doesn't want to hold entirely in memory (e.g.
        serving a source download). Raises `FileNotFoundError` up front
        (before yielding anything) if `key` does not exist."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Never raises for a missing key — that's exactly what `False`
        means. Only raises for a genuine backend failure (e.g. the
        provider is unreachable)."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove `key`. Idempotent: deleting a key that doesn't exist is
        NOT an error (matches `Path.unlink(missing_ok=True)`, which
        every current direct-filesystem call site already relies on for
        idempotent cleanup/rollback — see
        `app/api/documents.py::_revert_moved_uploads`)."""

    @abc.abstractmethod
    def move(self, source_key: str, dest_key: str) -> None:
        """Move `source_key` to `dest_key`, overwriting `dest_key` if it
        already exists. Raises `FileNotFoundError` if `source_key`
        doesn't exist. This is the primitive `app/api/documents.py`'s
        atomic-upload flow is built from (`Path.replace`, a same-
        filesystem rename) — see that module's docstring for why
        overwrite-on-move is the behavior a caller doing its own
        backup-before-move needs, not a footgun to guard against here."""

    @abc.abstractmethod
    def copy(self, source_key: str, dest_key: str) -> None:
        """Copy `source_key` to `dest_key`, overwriting `dest_key` if it
        already exists. Raises `FileNotFoundError` if `source_key`
        doesn't exist. Unlike `move`, `source_key` still exists
        afterward."""

    @abc.abstractmethod
    def get_local_path(self, key: str) -> str | None:
        """Return a real local filesystem path usable with ordinary
        `open()`/`Path()`-based libraries (pdfplumber, pytesseract,
        python-docx, ...) that don't know about this abstraction, OR
        `None` if `key` isn't backed by a local path at all (a remote
        backend must download to a temp file first — see
        `S3StorageBackend.get_local_path`, which does exactly that and
        returns the temp path; the caller is responsible for that temp
        file's cleanup — see `release_local_path`/`local_path` below).
        This is the deliberate escape hatch: the document-parsing
        pipeline (`app/documents/pipeline.py`) needs a real path today
        and rewriting every parser to take bytes/streams instead is out
        of scope for this pass (see this module's docstring, "Honest
        scope note")."""

    def release_local_path(self, path: str) -> None:
        """Clean up a path previously returned by `get_local_path`, if
        (and only if) this backend created a temporary copy for it.

        Concrete (not abstract) with a no-op default: for
        `LocalStorageBackend`, `get_local_path` returns the REAL file —
        "releasing" it must do nothing, or every caller would delete
        actual stored files out from under themselves.
        `S3StorageBackend` overrides this to delete the temp file it
        downloaded. Every caller that uses `get_local_path` for more
        than an instant should call this when done (or, better, use the
        `local_path()` context manager below, which calls it
        automatically) — otherwise a remote backend leaks one temp file
        per call, which matters most for a loop that processes many
        files (see `app/services/ingestion_service.py`).
        """
        return None

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """Every key starting with `prefix`, in no particular guaranteed
        order. `prefix=""` lists everything."""

    @abc.abstractmethod
    def size(self, key: str) -> int:
        """Size of `key` in bytes. Raises `FileNotFoundError` if `key`
        does not exist."""


@contextlib.contextmanager
def local_path(backend: "StorageBackend", key: str):
    """`with local_path(backend, key) as path:` — a real local path to
    `key`, automatically released (see `StorageBackend.release_local_path`)
    on exit, success or failure. The correct default way to get a local
    path for a *computational* use (parsing, rendering, hashing) that
    finishes before the `with` block ends.

    NOT appropriate for a path that must outlive this function — e.g.
    `app/api/source.py`'s `get_source_file`, which hands the path to
    Starlette's `FileResponse` and needs cleanup deferred until AFTER the
    response is actually sent. That call site uses
    `get_local_path`/`release_local_path` directly, via
    `FileResponse(..., background=BackgroundTask(backend.release_local_path, path))`,
    for exactly that reason — see that function's docstring.
    """
    path = backend.get_local_path(key)
    if path is None:
        raise StorageError(f"{type(backend).__name__} could not resolve a local path for {key!r}")
    try:
        yield path
    finally:
        backend.release_local_path(path)


def get_backend(settings) -> StorageBackend:
    """Select a `StorageBackend` from `settings.storage_backend`
    (`"local"` | `"s3"`). Raises `ValueError` for anything else —
    silently falling back to local storage for a typo'd production
    config value (e.g. `STORAGE_BACKEND=S3` vs `s3`) would be a much
    worse failure mode than refusing to start.

    Phase 5 completion pass, item 3.4: does NOT validate configuration
    consistency itself (that happens once, at startup — see
    `validate_storage_config`) so that calling this repeatedly per-request
    stays cheap; call `validate_storage_config` once during app startup
    (`app/main.py`'s `lifespan`) instead of relying on this function to
    catch a misconfiguration.
    """
    backend = (settings.storage_backend or "local").strip().lower()
    if backend == "local":
        from app.storage.local import LocalStorageBackend

        return LocalStorageBackend(root=_local_root(settings))
    if backend == "s3":
        from app.storage.s3 import S3StorageBackend

        return S3StorageBackend(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key=settings.s3_access_key.get_secret_value() if settings.s3_access_key else None,
            secret_key=settings.s3_secret_key.get_secret_value() if settings.s3_secret_key else None,
        )
    raise ValueError(f"Unknown STORAGE_BACKEND {backend!r} — must be 'local' or 's3'")


def _local_root(settings):
    """The actual directory `LocalStorageBackend` roots itself at.

    Preference order: (1) an explicitly-configured `storage_local_root`;
    (2) a root derived from `upload_dir`+`corpus_dir` sharing a common
    parent (`Settings.effective_storage_local_root`); (3) a root derived
    from `upload_dir` ALONE (`effective_storage_local_root_for("uploads")`)
    — since uploads is the one prefix actually wired through this backend
    as of this pass (see `app/api/documents.py`; `app/api/source.py`'s
    corpus reads are NOT yet routed through here — see this module's
    "Honest scope note"), a `CORPUS_DIR` that doesn't happen to share
    `UPLOAD_DIR`'s parent must not block deriving a perfectly-usable
    upload root. Falls back to the literal `storage_local_root` setting
    if none of the above apply.

    Relative paths are resolved against the backend project root, the
    same convention `app.services.ingestion_service._resolve_dir` uses
    for `UPLOAD_DIR`/`CORPUS_DIR` — so "relative" means the same thing
    everywhere a path/root setting appears in this project.
    """
    from pathlib import Path

    derived = (
        settings.effective_storage_local_root()
        or settings.effective_storage_local_root_for("uploads")
    )
    root = derived if derived is not None else Path(settings.storage_local_root)
    if not root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        root = project_root / root
    return root


class StorageConfigError(RuntimeError):
    """Raised by `validate_storage_config` — a config problem serious
    enough that the app should refuse to start rather than run with
    silently inconsistent or partially-working storage.
    """


# Fixed key-prefix convention every LOCAL-backend-routed call site uses —
# see `app/api/documents.py::_upload_key`/`_corpus_key` and
# `app/api/source.py::_source_key`. Kept here (not re-derived from
# `settings.upload_dir`/`settings.corpus_dir`, which are pathlib-flavored
# and Windows-path-aware) so every caller agrees on the exact same keys
# without each one re-deriving them slightly differently.
UPLOADS_KEY_PREFIX = "uploads"
CORPUS_KEY_PREFIX = "corpus"


def validate_storage_config(settings) -> None:
    """Called once, at app startup (`app/main.py`'s `lifespan`) —
    refuses to start with a storage configuration that would silently
    misbehave rather than failing loudly.

    Two checks:

    1. `STORAGE_BACKEND=s3` is now ALLOWED (Phase 5 completion pass,
       round 2, item 1 — previously outright rejected here). Re-auditing
       every application code path against the actual current code (not
       an earlier snapshot of it) found the two reasons this used to be
       rejected are no longer true:
         - `app.services.ingestion_service._list_source_files`/
           `_chunk_source_root` now enumerate via `backend.list_keys()`
           and materialize each file via `local_path()`, not a direct
           `pathlib` directory walk.
         - `app/api/source.py`'s `_source_key`/`_resolve_source_path`
           handle BOTH `source_root == "corpus"` and `"uploads"`
           uniformly through `get_backend(settings)` — corpus files are
           not special-cased to bypass the abstraction.
       `app/api/documents.py` (upload persist + rollback) and
       `app.services.ingestion_jobs.run_job` (background worker) were
       already confirmed storage-abstracted in the pass before this one.
       **What this check's removal does NOT claim**: `S3StorageBackend`
       has not been runtime-exercised against a real or mocked S3
       endpoint in the environment this pass was authored in (`boto3` is
       listed in `requirements.txt` but was not installable there — no
       network access). This is a code-path audit, not a live-tested
       guarantee — see `tests/test_storage.py` for what IS/ISN'T
       actually executable there, and treat a first real deployment
       against S3 as that feature's true first test, not a foregone
       conclusion.
    2. For `STORAGE_BACKEND=local` (the default), `storage_local_root`
       must actually agree with `upload_dir`/`corpus_dir` — i.e.
       `<storage_local_root>/uploads` must resolve to the same directory
       as `upload_dir`, and `<storage_local_root>/corpus` to
       `corpus_dir`. Both `app/api/documents.py` and `app/api/source.py`
       now read/write via the fixed `uploads/`/`corpus/` key prefixes
       above UNCONDITIONALLY (not just uploads — see check 1); a
       deployer who set `UPLOAD_DIR`/`CORPUS_DIR` to something that
       doesn't match `STORAGE_LOCAL_ROOT` would have files written to
       one directory while other code scans/serves a DIFFERENT one.
       Both are now checked strictly (corpus used to be checked only
       conditionally, back when it wasn't storage-abstracted at all —
       that condition no longer applies).
    """
    backend = (settings.storage_backend or "local").strip().lower()

    if backend not in ("local", "s3"):
        raise StorageConfigError(f"Unknown STORAGE_BACKEND {backend!r} — must be 'local' or 's3'.")

    if backend == "s3":
        # Configuration-completeness check only (item 24: "S3
        # credentials/bucket/endpoint must be environment-driven, never
        # hardcoded, never silently defaulted for a production backend
        # selection"). Only S3_BUCKET is actually required — access/
        # secret key are deliberately optional here (see
        # `app.storage.base.get_backend`/`app.storage.s3.S3StorageBackend`:
        # when unset, boto3 falls back to its standard credential chain
        # — environment variables, an IAM role, `~/.aws/credentials`,
        # etc. — which is the RECOMMENDED way to run in production, not
        # a misconfiguration to reject).
        #
        # Does NOT attempt to actually connect — that happens lazily on
        # first real use (`S3StorageBackend`), consistent with every
        # other external-provider client in this codebase (Pinecone,
        # Groq, Gemini) never being pinged at startup either.
        if not settings.s3_bucket:
            raise StorageConfigError(
                "STORAGE_BACKEND=s3 requires S3_BUCKET to be set (see .env.example). "
                "S3_ACCESS_KEY/S3_SECRET_KEY/S3_ENDPOINT_URL/S3_REGION are optional — "
                "when unset, boto3's standard credential chain (env vars, IAM role, "
                "~/.aws/credentials) is used."
            )
        return

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent

    def _resolve(p) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (project_root / path).resolve()

    root = _local_root(settings)

    # UPLOAD_DIR: the prefix app/api/documents.py writes uploads through
    # and app.services.ingestion_service scans — a mismatch here means
    # uploaded files would be written somewhere ingestion never scans.
    expected_uploads = (root / UPLOADS_KEY_PREFIX).resolve()
    actual_uploads = _resolve(settings.upload_dir)
    if expected_uploads != actual_uploads:
        raise StorageConfigError(
            f"Storage misconfiguration: the effective local storage root ({root}) implies "
            f"UPLOAD_DIR={expected_uploads}, but UPLOAD_DIR={settings.upload_dir!r} resolves to "
            f"{actual_uploads}. app/api/documents.py writes uploads via that root + "
            f"'{UPLOADS_KEY_PREFIX}/...' keys — a mismatch here means uploaded files would be "
            f"written somewhere app.services.ingestion_service never scans. Either set "
            f"STORAGE_LOCAL_ROOT explicitly to match, or leave it unset and make sure UPLOAD_DIR "
            f"is named '.../uploads'."
        )

    # CORPUS_DIR: checked unconditionally now (see this function's
    # docstring, check 2) — app/api/source.py and
    # app.services.ingestion_service both route corpus reads through
    # this same root + '{CORPUS_KEY_PREFIX}/...' keys, same as uploads.
    expected_corpus = (root / CORPUS_KEY_PREFIX).resolve()
    actual_corpus = _resolve(settings.corpus_dir)
    if expected_corpus != actual_corpus:
        raise StorageConfigError(
            f"Storage misconfiguration: the effective local storage root ({root}) implies "
            f"CORPUS_DIR={expected_corpus}, but CORPUS_DIR={settings.corpus_dir!r} resolves to "
            f"{actual_corpus}. Align CORPUS_DIR with STORAGE_LOCAL_ROOT (or leave both at their "
            f"defaults)."
        )