"""GET /documents/{id}/ingestion, POST /documents/{id}/reingest — status
and retry endpoints for the background ingestion job engine (Phase 5
completion pass, item 3.3; referenced by name in
`app.services.ingestion_jobs`'s module docstring since before this file
existed).

Split from `app/api/documents.py` on purpose (not added to that
already-large, carefully-tested module) — these are read/retry
endpoints over `IngestionJob` rows, not part of the upload
validate/stage/move/rollback flow those tests protect.

**Why `POST /documents/{id}/reingest` uses genuine background execution
(`BackgroundTasks` + `app.services.ingestion_jobs.run_job`) while
`POST /documents/upload` does NOT** (see that module's `upload_documents`
for the full reasoning): there is no existing test asserting this
endpoint's response status code reflects the retry's eventual outcome —
it only asserts a job was queued and scheduled, which IS knowable
synchronously. Nothing here blocks on `ingest_corpus()` finishing;
`GET /documents/{id}/ingestion` is how a caller learns the outcome.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.config import get_settings
from app.db.models import Document, IngestionJob, User
from app.db.session import init_db, session_scope
from app.schemas.ingestion import DocumentIngestionStatusOut, IngestionJobOut, ReingestResponse
from app.security.auth import get_current_user_optional
from app.services.ingestion_jobs import JobStatus, create_job, run_job

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["ingestion"])


def _get_document_or_404(session, document_id: int, current_user: User | None) -> Document:
    """Same tenant-isolation enforcement point as
    `app.api.source._get_document_or_404` (a document another user owns
    is treated as 404, not 403 — see that function's docstring for why).
    Duplicated rather than imported across modules to keep
    `app/api/source.py` (a read-only, carefully-scoped module) untouched
    by this pass, per the "do not rewrite already-working Phase 5 code"
    rule — the two implementations must stay behaviorally identical if
    either changes; that's a acceptable small duplication for that
    isolation, not an oversight.
    """
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document with id={document_id}")
    if doc.owner_id is not None and (current_user is None or doc.owner_id != current_user.id):
        raise HTTPException(status_code=404, detail=f"No document with id={document_id}")
    return doc


def _latest_job_for(session, doc: Document) -> IngestionJob | None:
    """Most recent job for this document — matched first by
    `document_id` (once a job has been linked, e.g. by
    `app.api.documents._record_job_outcome` or `run_job`), falling back
    to `(source_root, filename, owner_id)` for a job that was created
    before the `Document` row existed yet (see
    `app.services.ingestion_jobs.create_job`'s docstring — `document_id`
    is nullable for exactly this reason).
    """
    job = (
        session.query(IngestionJob)
        .filter(IngestionJob.document_id == doc.id)
        .order_by(IngestionJob.created_at.desc())
        .first()
    )
    if job is not None:
        return job
    return (
        session.query(IngestionJob)
        .filter(
            IngestionJob.source_root == doc.source_root,
            IngestionJob.filename == doc.filename,
            IngestionJob.owner_id == doc.owner_id,
        )
        .order_by(IngestionJob.created_at.desc())
        .first()
    )


@router.get("/jobs/{job_id}", response_model=IngestionJobOut)
def get_job_status(
    job_id: int,
    current_user: User | None = Depends(get_current_user_optional),
) -> IngestionJobOut:
    """Poll a job directly by id — the only option immediately after
    `POST /documents/upload` (Phase 5 completion pass, round 2, item 2):
    at that point `document_id` isn't known yet (ingestion, which
    creates the `Document`/`DocumentVersion` rows, hasn't run — see
    `app/api/documents.py::upload_documents`'s docstring), so
    `GET /documents/{id}/ingestion` has nothing to key off of yet.
    `UploadResponse.jobs[].job_id` (returned by the upload call) is
    exactly what a caller polls here.

    Declared with the literal path segment `jobs` BEFORE
    `/{document_id}/ingestion` below in route registration order isn't
    actually required for correctness — FastAPI/Starlette route
    matching distinguishes `/documents/jobs/{job_id}` from
    `/documents/{document_id}/ingestion` by looking at what's in the
    LAST segment (`{job_id}` vs. the literal `ingestion`), not
    declaration order — but is kept adjacent to that route for
    readability.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        job = session.get(IngestionJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No ingestion job with id={job_id}")
        # Same tenant-isolation rule as _get_document_or_404: another
        # user's job is reported as 404, never 403 (no confirmation that
        # a given job id even exists for someone who doesn't own it).
        if job.owner_id is not None and (current_user is None or job.owner_id != current_user.id):
            raise HTTPException(status_code=404, detail=f"No ingestion job with id={job_id}")
        return IngestionJobOut(**job.to_dict())


@router.get("/{document_id}/ingestion", response_model=DocumentIngestionStatusOut)
def get_ingestion_status(
    document_id: int,
    current_user: User | None = Depends(get_current_user_optional),
) -> DocumentIngestionStatusOut:
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        job = _latest_job_for(session, doc)
        return DocumentIngestionStatusOut(
            document_id=doc.id,
            filename=doc.filename,
            current_version_id=doc.current_version_id,
            has_current_version=doc.current_version_id is not None,
            latest_job=IngestionJobOut(**job.to_dict()) if job is not None else None,
        )


@router.post("/{document_id}/reingest", response_model=ReingestResponse)
def retry_document_ingestion(
    document_id: int,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(get_current_user_optional),
) -> ReingestResponse:
    """Queue a fresh ingestion attempt for `document_id` and schedule it
    to run in the background. Returns as soon as the job is QUEUED —
    poll `GET /documents/{id}/ingestion` for the outcome.

    Idempotency: `run_job` calls the same `ingest_corpus()` pipeline the
    upload path uses, which is itself idempotent (deterministic chunk
    IDs, content-hash-gated re-embedding — see
    `app.services.ingestion_service`). Re-running it via this endpoint
    when nothing actually changed on disk is a safe, fast no-op, not a
    duplicate document/version/vector.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        job = create_job(
            session,
            filename=doc.filename,
            source_root=doc.source_root,
            owner_id=doc.owner_id,
            document_id=doc.id,
            version_id=doc.current_version_id,
            settings=settings,
        )
        session.flush()
        job_dict = job.to_dict()
        job_id = job.id

    background_tasks.add_task(_run_job_safely, job_id, settings)

    return ReingestResponse(status=JobStatus.QUEUED.value, job=IngestionJobOut(**job_dict))


def _run_job_safely(job_id: int, settings) -> None:
    """`BackgroundTasks` target: `run_job` already catches and records
    every exception from `ingest_corpus()` onto the job row itself (see
    its docstring) — this wrapper exists only to guarantee that if
    `run_job` ITSELF somehow raises (e.g. the job row went missing), that
    never propagates out of a background task and crashes the ASGI
    server's task-cleanup path. Logged, not silently swallowed.
    """
    try:
        run_job(job_id, settings=settings)
    except Exception:  # noqa: BLE001 - background task must never raise past this point
        _log.exception("background run_job(job_id=%s) failed unexpectedly", job_id)
