"""POST /documents/upload, GET /documents — upload, queue-for-ingestion,
and list documents.

**Asynchronous upload flow (Phase 5 completion pass, round 3, item 1).**
A batch of uploaded files goes through:

    upload -> staging (temp) -> validate -> parse-check -> persist to
    storage (with replacement backup) -> create IngestionJob row(s),
    QUEUED -> COMMIT -> 202 Accepted

Nothing about ingestion itself (parse -> OCR -> chunk -> embed ->
Pinecone -> BM25) runs in this request or in a FastAPI `BackgroundTasks`
callback tied to this process. The durable execution mechanism is the
STANDALONE WORKER (`python -m app.services.ingestion_jobs worker` — see
that module's docstring for the full architecture, including the
documented SQLite-vs-PostgreSQL concurrent-claiming caveat), running as
its own process/container (`docker-compose.yml`'s `worker` service),
polling for QUEUED jobs independently of whether the API process that
accepted a given upload is even still running. If the API process
crashes between "202 returned" and the worker's next poll, the job is
already durably QUEUED in the database — nothing is lost, and no
in-memory callback needed to survive.

See `upload_documents`'s inline comment (right before
`_create_jobs_for_batch` is called) for the full reasoning, and
`tests/test_upload_atomicity.py`'s module docstring for exactly what
"atomicity" means for the SYNCHRONOUS part of this request now that the
response can no longer depend on ingestion's eventual outcome: job row
creation + file bytes durably saved to storage is still all-or-nothing;
a failed BACKGROUND ingestion no longer deletes the uploaded file (it
stays for inspection/retry, with the job marked FAILED), except when it
was a replacement of an already-successful file — see `run_job`'s
"Replacement-file backup/restore" in `app.services.ingestion_jobs`,
which now owns that restore logic (moved there from this module so it
survives an API-process crash the same way job processing itself does).

**Storage abstraction (Phase 5 completion pass, item 3.4).** Uploaded
bytes are persisted via `app.storage.base.get_backend(settings)` —
`_upload_key`/`get_backend` — not written directly with `pathlib`. See
`app/storage/base.py`'s module docstring for exactly which backends
(local disk; S3, once `STORAGE_BACKEND=s3` is selected) this supports
and what's been verified for each.

**Duplicate-filename replacement bug (fixed — originally Phase 3B/4,
re-verified against the storage abstraction + worker-based rewrite
above).** Persisting a NEW file at a key that already has content would
otherwise silently overwrite that pre-existing, previously-successful
file's bytes. Fixed by reading the existing bytes back before
overwriting and durably recording them as a restorable backup ON THE JOB
ROW itself (`IngestionJob.backup_key`/`restore_to_key` — not held in
this request's memory, so a crash between "backup taken" and "job
processed" doesn't lose the backup):

    existing bytes at key (if any)
         |
         v  read into memory, saved to a distinct backup_key,
         |  and RECORDED on the new IngestionJob row
    new bytes written to the same key
         |
         v  worker picks up the QUEUED job whenever it does
         |
         +-- job reaches COMPLETED -> backup discarded (see run_job)
         +-- job reaches terminal FAILED -> backup bytes restored (see run_job)

This guarantees the required invariant: a failed upload/ingestion
operation must never corrupt or permanently destroy a previously
successful document. See
`tests/test_upload_atomicity.py::test_failed_reupload_of_existing_filename_restores_original_bytes_once_terminally_failed`.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import groq
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from google.genai import errors as genai_errors
from pinecone.exceptions import PineconeException

from app.core.config import get_settings
from app.db.models import Document, DocumentVersion, IngestionJob, User
from app.db.session import init_db, session_scope
from app.documents.pipeline import DocumentParseError, SUPPORTED_EXTENSIONS, extract_document
from app.rag.llm import LLMProviderError
from app.schemas.documents import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentOut,
    DocumentVersionOut,
    UploadJobRef,
    UploadRejection,
    UploadResponse,
)
from app.security.auth import get_current_user_optional, get_current_user_required
from app.security.file_validation import FileValidationError, validate_upload_bytes
from app.services.ingestion_jobs import create_job
from app.services.ingestion_service import _resolve_dir, ingest_corpus
from app.storage.base import UPLOADS_KEY_PREFIX, get_backend

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_UPSTREAM_ERRORS = (LLMProviderError, groq.APIError, genai_errors.APIError, PineconeException)

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(original: str | None) -> str:
    name = Path((original or "").strip()).name
    if not name or name in {".", ".."}:
        raise ValueError("missing or invalid filename")
    safe = _SAFE_FILENAME_RE.sub("_", name).lstrip(".")
    if not safe:
        raise ValueError("missing or invalid filename")
    return safe


def _upload_key(safe_name: str, owner_id: int | None) -> str:
    """The storage key (`app.storage.base.StorageBackend`) an upload's
    persisted copy lives at — Phase 5 completion pass, item 3.4. Mirrors
    the pre-existing per-user subdirectory layout
    (`data/uploads/<owner_id>/<filename>` for an owned upload,
    `data/uploads/<filename>` for anonymous) as a storage KEY rather than
    a local path: `"uploads/<owner_id>/<filename>"` /
    `"uploads/<filename>"`. `app.services.ingestion_service` and
    `app/api/source.py` must agree on this exact same convention — see
    `app.storage.base.validate_storage_config`'s docstring for how a
    `STORAGE_LOCAL_ROOT`/`UPLOAD_DIR` mismatch is caught at startup
    rather than silently misrouting files.
    """
    if owner_id is not None:
        return f"{UPLOADS_KEY_PREFIX}/{owner_id}/{safe_name}"
    return f"{UPLOADS_KEY_PREFIX}/{safe_name}"


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    response: Response,
    files: list[UploadFile] = File(...),
    current_user: User | None = Depends(get_current_user_optional),
) -> UploadResponse:
    settings = get_settings()
    upload_dir = _resolve_dir(settings, settings.upload_dir)
    # Phase 5 tenant isolation: an authenticated upload goes into its own
    # per-user subdirectory (`data/uploads/<user_id>/`) — this is what
    # `app.services.ingestion_service._chunk_directory_with_metadata`
    # scans to assign `owner_id` (see its docstring/comment there). An
    # anonymous upload (no token) keeps the exact pre-Phase-5 behavior:
    # written straight into `data/uploads/`, publicly visible, matching
    # every pre-Phase-5 test.
    if current_user is not None:
        upload_dir = upload_dir / str(current_user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    resolved_upload_dir = upload_dir.resolve()
    max_bytes = settings.max_upload_file_size_bytes

    accepted: list[str] = []
    rejected: list[UploadRejection] = []
    # (staged_path, safe_name) for every file that passed validation —
    # moved into storage below via the storage key derived from
    # safe_name (+ owner subdirectory), and rolled back from there if
    # ingestion fails. Kept as `safe_name` rather than a resolved local
    # `Path` (pre Phase 5 completion pass) since the persisted copy is
    # now written via `app.storage.base.get_backend`, which may not be
    # local disk at all — see `_upload_key`.
    staged: list[tuple[Path, str]] = []

    # Stage in a temp directory OUTSIDE upload_dir, so a half-validated
    # batch is never visible to a concurrent ingestion run or directory
    # scan (`app.services.ingestion_service` lists `upload_dir` directly).
    staging_root = Path(tempfile.mkdtemp(prefix="veridoc-upload-"))
    try:
        for upload in files:
            staged_path: Path | None = None
            try:
                safe_name = _safe_filename(upload.filename)
                suffix = Path(safe_name).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise ValueError(
                        f"unsupported file type '{suffix}' (supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
                    )

                data = await upload.read(max_bytes + 1)
                if not data:
                    raise ValueError("file is empty")
                if len(data) > max_bytes:
                    raise ValueError(f"file exceeds the {max_bytes // (1024 * 1024)}MB upload limit")

                # Phase 5 completion pass, section 6: don't trust the
                # extension alone — check the file's actual magic bytes,
                # and (for .docx specifically, since it's a zip archive)
                # guard against a zip-bomb-style decompression attack.
                # Raises FileValidationError (a ValueError) on failure,
                # caught by the same handler below as every other
                # per-file validation error.
                validate_upload_bytes(data, suffix)

                final_dest = upload_dir / safe_name
                if not final_dest.resolve().is_relative_to(resolved_upload_dir):
                    raise ValueError("resolved destination is outside the upload directory")

                # Unique staged name (uuid prefix) so two files with the
                # same original name in one batch can't collide in staging.
                staged_path = staging_root / f"{uuid.uuid4().hex}_{safe_name}"
                staged_path.write_bytes(data)

                # Fail fast on a corrupt/unreadable file, per-file, with the
                # same parser ingestion itself will run — against the
                # STAGED copy, before it ever touches data/uploads/.
                extract_document(staged_path, settings)

                staged.append((staged_path, safe_name))
                accepted.append(safe_name)

            except (DocumentParseError, FileValidationError, ValueError) as exc:
                if staged_path is not None and staged_path.exists():
                    staged_path.unlink(missing_ok=True)
                rejected.append(UploadRejection(filename=upload.filename or "(unnamed)", error=str(exc)))
            except Exception as exc:  # noqa: BLE001
                _log.exception("unexpected error handling upload %s", upload.filename)
                if staged_path is not None and staged_path.exists():
                    staged_path.unlink(missing_ok=True)
                rejected.append(
                    UploadRejection(filename=upload.filename or "(unnamed)", error=f"unexpected error: {exc}")
                )
            finally:
                await upload.close()

        if not staged:
            response.status_code = 200
            return UploadResponse(status="error", files=[], rejected=rejected)

        # Phase 5 completion pass (round 3), item 1: the STANDALONE
        # WORKER (`python -m app.services.ingestion_jobs worker` —
        # `app.services.ingestion_jobs.poll_and_process_once`/`run_job`)
        # is now the durable ingestion execution mechanism, not FastAPI
        # `BackgroundTasks` (superseded — see MIGRATION_PLAN.md / this
        # module's git history for the round-2 BackgroundTasks design
        # this replaces). This request does exactly three things, in
        # order, each rolled back if a LATER step fails:
        #
        #   1. persist accepted files' bytes via the storage backend
        #      (backing up any pre-existing bytes at the same key first)
        #   2. create one QUEUED IngestionJob per file, referencing that
        #      backup (if any)
        #   3. return 202 — nothing further happens in THIS process
        #
        # The standalone worker picks up QUEUED jobs on its own polling
        # loop, independently of whether the process that accepted this
        # upload is even still running. If the API process crashes
        # between step 3 and the worker's next poll, the job is durably
        # QUEUED in the database — the worker (running as ITS OWN
        # process/container — see docker-compose.yml's `worker` service)
        # picks it up exactly as if nothing had happened. See
        # `app.services.ingestion_jobs`'s module docstring for the full
        # architecture (including the documented SQLite-vs-PostgreSQL
        # concurrent-claiming caveat) and `run_job`'s docstring for the
        # backup/restore guarantee step 1's backups feed into.
        backend = get_backend(settings)
        moved: list[tuple[str, str | None]] = []  # (key, backup_key_or_None)
        try:
            for staged_path, safe_name in staged:
                key = _upload_key(safe_name, current_user.id if current_user else None)
                backup_key: str | None = None
                if backend.exists(key):
                    # Durably back up the PRE-UPLOAD bytes before
                    # overwriting — read + re-save under a distinct key
                    # (not held in memory across the request, unlike the
                    # round-2 design: this must survive the request
                    # process exiting — see `run_job`'s docstring,
                    # "Replacement-file backup/restore"). Unique suffix
                    # per backup so two concurrent replacements of the
                    # same filename can't collide.
                    backup_key = f"{UPLOADS_KEY_PREFIX}/.backups/{uuid.uuid4().hex}_{safe_name}"
                    backend.save(backup_key, backend.read(key))
                backend.save(key, staged_path.read_bytes())
                moved.append((key, backup_key))
        except Exception as exc:  # noqa: BLE001 - storage backend can raise more than OSError (e.g. S3)
            _log.exception("failed to persist uploaded file(s) to storage; rolling back batch")
            _rollback_moved_uploads(backend, moved)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to persist uploaded file(s) to storage ({type(exc).__name__}). Nothing was queued.",
            ) from exc

        try:
            job_ids = _create_jobs_for_batch(
                settings,
                accepted,
                owner_id=current_user.id if current_user else None,
                backup_keys={name: backup_key for (_key, backup_key), name in zip(moved, accepted)},
            )
        except Exception:  # noqa: BLE001 - DB error creating job rows; nothing must be left "queued" without a job
            job_ids = []
        if not job_ids:
            # For an async, queue-only endpoint, a job that failed to
            # persist is not an acceptable "queued" response (item 1/2:
            # "Do not return success if the job was not actually
            # persisted"). Covers BOTH `_create_jobs_for_batch` raising
            # AND it returning an empty list without raising (e.g. a
            # caller-supplied replacement that creates zero job rows for
            # some other reason) — either way, no job exists to process
            # these files, so the files WERE already written to storage
            # above must be rolled back too, restoring any backups taken.
            _log.error("failed to create IngestionJob rows for an upload batch; rolling back storage writes too")
            _rollback_moved_uploads(backend, moved)
            raise HTTPException(
                status_code=503,
                detail="Failed to persist ingestion job record(s) for this upload — nothing was queued. Please retry.",
            )

        job_refs = [UploadJobRef(filename=name, job_id=jid) for name, jid in zip(accepted, job_ids)]

        response.status_code = 202
        return UploadResponse(
            status="queued",
            files=accepted,
            rejected=rejected,
            jobs=job_refs,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _create_jobs_for_batch(
    settings, filenames: list[str], *, owner_id: int | None, backup_keys: dict[str, str | None]
) -> list[int]:
    """Create one QUEUED `IngestionJob` per accepted filename — the
    durable hand-off point to the standalone worker (Phase 5 completion
    pass, round 3, item 1). Unlike the round-2 helper this replaces, job
    creation is NOT best-effort/non-fatal: for a queue-only endpoint, a
    job that silently failed to persist must not be reported as
    "queued". Any exception here propagates to the caller (`upload_documents`),
    which rolls back the already-persisted storage writes for this
    batch and returns 503 — see that call site's `except` block.

    `backup_keys[filename]` (set by `upload_documents` when this file
    is REPLACING an existing, previously-successful filename; `None`
    for a brand-new filename) is recorded directly on the new
    `IngestionJob` row as `backup_key`/`restore_to_key` — see
    `app.services.ingestion_jobs.create_job`'s docstring and `run_job`'s
    "Replacement-file backup/restore". Storing it on the job row (not
    kept in this request's memory) is what lets the WORKER — running in
    a different process, possibly started long after this request
    returned — correctly restore or discard that backup when the job
    reaches a terminal state.
    """
    if not filenames:
        return []
    init_db(settings)
    with session_scope(settings) as session:
        jobs = [
            create_job(
                session,
                filename=name,
                source_root="uploads",
                owner_id=owner_id,
                backup_key=backup_keys.get(name),
                restore_to_key=_upload_key(name, owner_id) if backup_keys.get(name) else None,
                settings=settings,
            )
            for name in filenames
        ]
        session.flush()
        return [j.id for j in jobs]


def _rollback_moved_uploads(backend, moved: list[tuple[str, str | None]]) -> None:
    """Undo the staging->storage persist for a batch that could not be
    fully queued (a later step in `upload_documents` — another file's
    storage save, or `IngestionJob` row creation — failed).

    For each `(key, backup_key)` pair already written in this request:
      - `backup_key` set and its blob still exists (this `key` is a
        REPLACEMENT of a pre-existing, previously-successful file):
        restore those original bytes onto `key`, then discard the
        backup blob — the previously-successful document's content ends
        up exactly what it was before this request, byte-for-byte.
      - otherwise (a brand-new filename, nothing to restore): delete
        `key` outright — it never existed before this request.

    This is the one rollback path for "some files in this batch were
    already durably persisted to storage before a later step failed" —
    used both when a storage write itself fails mid-batch and when job
    -row creation fails after every file was already persisted, so the
    two failure points can never silently diverge in behavior again
    (see this module's git history: an earlier version of the
    storage-write failure path called a since-removed helper with the
    wrong argument shape and never actually restored a replaced file's
    backup — fixed here by giving both call sites exactly one correct
    implementation).
    """
    for key, backup_key in moved:
        try:
            if backup_key and backend.exists(backup_key):
                backend.save(key, backend.read(backup_key))
                backend.delete(backup_key)
            else:
                backend.delete(key)
        except Exception:  # noqa: BLE001 - a storage backend can raise more than OSError (e.g. S3StorageBackend)
            _log.exception("failed to roll back storage write for key=%s", key)


@router.get("", response_model=DocumentListResponse)
def list_documents(current_user: User | None = Depends(get_current_user_optional)) -> DocumentListResponse:
    """List every document known to the metadata database, scoped to the
    shared/public corpus plus the current user's own uploads (Phase 5
    tenant isolation — an anonymous request sees only `owner_id IS NULL`,
    which is every pre-Phase-5 row, exactly matching pre-Phase-5
    behavior), with its current version's stats (pages, tables, OCR
    confidence). Returns an empty list before ingestion has ever run.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        query = session.query(Document)
        if current_user is not None:
            query = query.filter(
                (Document.owner_id.is_(None)) | (Document.owner_id == current_user.id)
            )
        else:
            query = query.filter(Document.owner_id.is_(None))
        docs = query.order_by(Document.source_root, Document.filename).all()
        out = []
        for d in docs:
            version_out = None
            if d.current_version_id:
                version = session.get(DocumentVersion, d.current_version_id)
                if version:
                    version_out = DocumentVersionOut(**version.to_dict())
            out.append(
                DocumentOut(
                    id=d.id,
                    source_root=d.source_root,
                    filename=d.filename,
                    file_type=d.file_type,
                    title=d.title,
                    owner_id=d.owner_id,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                    updated_at=d.updated_at.isoformat() if d.updated_at else None,
                    current_version=version_out,
                )
            )
        return DocumentListResponse(documents=out)


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user_required),
) -> DeleteDocumentResponse:
    """Delete a document — its metadata rows, stored file, and every
    trace of it in the retrieval indexes (Phase 5 completion pass, round
    2, item 19: "document lifecycle: delete").

    **Scope, deliberately narrow**: only a document the CALLER owns
    (`owner_id == current_user.id`) can be deleted through this endpoint
    — never a shared/public corpus document (`owner_id IS NULL`). The
    corpus (`data/corpus/`) is the project's shipped, immutable reference
    set (see `data/corpus/README.txt`); allowing any authenticated user
    to delete shared content everyone else relies on would be a
    multi-tenant safety hole, not a feature. A corpus document, or
    another user's upload, is reported as 404 — matching this codebase's
    established tenant-isolation convention (see `app/api/source.py`'s
    `_get_document_or_404`: "a document another user owns is treated as
    404, not 403, so its very existence isn't confirmable by a user who
    shouldn't see it").

    **What gets removed, in order:**
      1. The stored file bytes (`app.storage.base.get_backend`) —
         removed FIRST, deliberately: `ingest_corpus()`'s existing
         reconciliation logic (step 4 below) only recognizes a document
         as "stale, remove its vectors/lexical-index entries" once it's
         genuinely gone from storage — see
         `app.services.ingestion_service._do_ingest`'s
         `stale_ids = existing_ids - current_chunk_ids` computation. This
         reuses that ALREADY-TESTED reconciliation path rather than
         duplicating a second, independently-written "remove this
         document's vectors" implementation that could drift out of
         sync with it.
      2. Every `IngestionJob` row referencing this document (full
         history removal — this is a hard delete, not an archive).
      3. The `DocumentVersion` row(s), then the `Document` row itself.
      4. `ingest_corpus(settings=settings)` — reconciles Pinecone (stale
         vector cleanup) and rewrites the BM25 lexical index snapshot to
         no longer include this document's chunks. Run synchronously
         (not backgrounded): a delete response claiming success ought to
         mean "actually gone from retrieval", not "gone from the DB, but
         still answerable via a stale Pinecone vector for a few more
         minutes until some other request happens to trigger
         reconciliation."

    If the storage delete or the DB rows are removed but the final
    `ingest_corpus()` reconciliation step fails (e.g. a transient
    Pinecone error), this raises 503 — the document is genuinely gone
    from what a NEW request would see (DB, storage), but stale vectors
    may briefly remain retrievable until reconciliation is retried
    (e.g. via the next successful upload/reingest, which runs the same
    `ingest_corpus()` reconciliation, or by an operator re-running
    `python -m app.services.ingestion_service`). This is reported
    honestly rather than silently swallowed.
    """
    settings = get_settings()
    init_db(settings)

    with session_scope(settings) as session:
        doc = session.get(Document, document_id)
        if doc is None or doc.owner_id != current_user.id:
            # Deliberately the SAME 404 for "doesn't exist" and "exists
            # but isn't yours (or is shared corpus content)" — see
            # docstring.
            raise HTTPException(status_code=404, detail=f"No document with id={document_id}")

        filename = doc.filename
        owner_id = doc.owner_id
        key = _upload_key(filename, owner_id)

        job_ids = [j.id for j in session.query(IngestionJob.id).filter(IngestionJob.document_id == document_id)]

    # Step 1: remove the stored bytes FIRST (outside the DB transaction
    # above — a storage failure here must not leave a half-deleted DB
    # state) — see docstring for why this must happen before the
    # reconciliation step's chunk-diffing can recognize the document as
    # gone.
    backend = get_backend(settings)
    try:
        backend.delete(key)
    except Exception as exc:  # noqa: BLE001 - report clearly rather than a bare 500 traceback
        _log.exception("failed to delete storage key %s for document_id=%s", key, document_id)
        raise HTTPException(status_code=503, detail=f"Could not delete the stored file: {exc}") from exc

    # Steps 2-3: remove every DB trace (jobs, versions, the document
    # itself) in one transaction.
    with session_scope(settings) as session:
        session.query(IngestionJob).filter(IngestionJob.document_id == document_id).delete(
            synchronize_session=False
        )
        session.query(DocumentVersion).filter(DocumentVersion.document_id == document_id).delete(
            synchronize_session=False
        )
        session.query(Document).filter(Document.id == document_id).delete(synchronize_session=False)

    # Step 4: reconcile Pinecone + the BM25 lexical index snapshot — see
    # docstring for why this reuses ingest_corpus()'s existing
    # reconciliation rather than a bespoke removal path.
    try:
        summary = ingest_corpus(settings=settings)
        vectors_deleted = summary.get("stale_deleted", 0)
        lexical_remaining = summary.get("chunks", 0)
    except SystemExit:
        # ingest_corpus() raises SystemExit when NO ingestable files
        # remain anywhere in corpus/uploads at all (see its docstring) —
        # a legitimate outcome of deleting the last remaining document,
        # not a failure. Nothing left to reconcile against.
        vectors_deleted = 0
        lexical_remaining = 0
    except _UPSTREAM_ERRORS as exc:
        _log.warning("post-delete reconciliation failed for document_id=%s: %s: %s", document_id, type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Document {filename!r} was deleted from storage and the database, but reconciling "
                f"the vector/lexical indexes failed ({type(exc).__name__}). Stale entries may briefly "
                "remain retrievable — see this endpoint's docstring."
            ),
        ) from exc

    return DeleteDocumentResponse(
        status="deleted",
        document_id=document_id,
        filename=filename,
        jobs_deleted=len(job_ids),
        vectors_deleted=vectors_deleted,
        lexical_index_chunks_remaining=lexical_remaining,
    )