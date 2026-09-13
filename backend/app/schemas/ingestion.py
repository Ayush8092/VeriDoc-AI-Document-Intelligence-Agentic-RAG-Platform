"""Response schemas for the ingestion-job endpoints (Phase 5 completion
pass, item 3.3) — `app/api/ingestion_jobs.py`. A separate module from
`app/schemas/documents.py` (which already has an unrelated
`DocumentOut`/`DocumentVersionOut`) so this stays easy to find and this
pass's diff stays additive rather than growing an already-large file.

Every field here is read directly off `IngestionJob`/`Document`/
`DocumentVersion` ORM rows (`app/db/models.py`) — never raw ORM objects
returned directly from an endpoint, and never an internal stack trace or
secret in `error` (`IngestionJob.error` is already truncated to 2000
chars and is a `str(exc)`, not a traceback — see
`app.services.ingestion_jobs.run_job`).
"""

from __future__ import annotations

from pydantic import BaseModel


class IngestionJobOut(BaseModel):
    id: int
    document_id: int | None
    version_id: int | None
    filename: str
    source_root: str
    status: str
    attempt_count: int
    max_attempts: int
    error: str
    chunks_total: int
    chunks_changed: int
    vectors_upserted: int
    duration_seconds: float | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None


class DocumentIngestionStatusOut(BaseModel):
    """`GET /documents/{id}/ingestion` response — the document, its
    current version (if any), and its most recent `IngestionJob` (if
    any job has ever been recorded for it — a document ingested before
    this pass, or ingested via the corpus/CLI path rather than an
    upload, may legitimately have `latest_job=None`).
    """

    document_id: int
    filename: str
    current_version_id: int | None
    has_current_version: bool
    latest_job: IngestionJobOut | None


class ReingestResponse(BaseModel):
    """`POST /documents/{id}/reingest` response — confirms a new job was
    queued and scheduled; does NOT wait for/report the job's outcome
    (poll `GET /documents/{id}/ingestion` for that, same as after an
    upload — see `app/api/ingestion_jobs.py`'s docstring for why this
    endpoint is genuinely asynchronous, unlike `POST /documents/upload`).
    """

    status: str  # "queued"
    job: IngestionJobOut
