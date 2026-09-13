"""Background ingestion job lifecycle (Phase 5 completion pass, items 4/5).

**Architecture choice, and why** (spec explicitly asks for this to be
documented): a DB-backed job queue, not Redis + Celery. No Redis instance
was available in this environment to build and integration-test a real
broker/worker split against, and this project's scale (a handful of
ingestion jobs at a time, not a high-throughput multi-tenant SaaS) does
not need a distributed queue yet — introducing Celery/Redis without being
able to verify it actually works would violate the "don't fabricate
results" / "don't introduce unnecessary infrastructure" guidance more
than it would help. `IngestionJob` (app/db/models.py) is the queue table;
`run_job` is the "worker" function. Two ways to invoke it ship:

1. In-process, via FastAPI `BackgroundTasks` (see
   `app/api/ingestion_jobs.py::retry_document_ingestion`) — runs after
   the HTTP response is sent, in the same process. This is genuine
   asynchronous execution (the caller does not block on it), but is
   single-process: it does not survive a process restart mid-job, and
   does not scale across multiple backend replicas without a shared
   queue. Documented, not hidden.
2. `python -m app.services.ingestion_jobs.worker` — a standalone polling
   worker (see `poll_and_process_once`) that claims QUEUED/RETRYING jobs
   from the database. On PostgreSQL this uses `SELECT ... FOR UPDATE SKIP
   LOCKED` so multiple worker processes can run safely in parallel
   without double-processing a job. **On SQLite (the default local-dev
   database), `FOR UPDATE SKIP LOCKED` is not supported** — SQLite has no
   row-level locking — so the polling worker is single-worker-only
   against SQLite; running two worker processes against the same SQLite
   file can race. This is an explicit, documented limitation, not a
   silent one: `poll_and_process_once` detects a non-PostgreSQL engine
   and logs it once.

**Idempotency**: `run_job` calls the SAME `ingest_corpus()` pipeline the
synchronous upload path already uses. That pipeline is already idempotent
(deterministic chunk IDs, `DocumentVersion.file_hash` / chunk
`content_hash` gate re-embedding and re-versioning — see
`app/services/ingestion_service.py`) — a retried job re-running
`ingest_corpus()` therefore never creates duplicate documents, versions,
vectors, or BM25 entries; it either finds nothing changed (fast no-op) or
picks up genuinely new/changed content.

**Scope limitation** (documented, not hidden): `ingest_corpus()` is
corpus-scoped (it reconciles ALL of `corpus_dir` + every user's
`upload_dir` subdirectory together — see that module's docstring), not
per-document. A single job's `run_job` therefore triggers a full
reconciliation pass, not a narrowly-scoped "parse just this one file"
operation. This is safe (idempotent, tenant-isolated, no duplicate
vectors) but means job "progress" is really "the outcome of the run that
happened to include this job's target file", and one job's failure
reflects whatever the whole batch's outcome was. A narrower per-file
ingestion path (skip the rest of the corpus scan entirely) would be a
real improvement but is a larger change to `ingest_corpus` than fits
safely alongside this pass's other constraints — see
docs/architecture.md, "Background ingestion: known limitations".
"""

from __future__ import annotations

import argparse
import logging
import time
from enum import Enum

from app.core.config import Settings, get_settings
from app.db.models import Document, DocumentVersion, IngestionJob
from app.db.session import get_session_factory, init_db

_log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


# Exceptions worth retrying: transient upstream/network failures. A
# permanent validation error (bad file, unsupported format) is NEVER
# retried — see `_is_transient`. Imported lazily inside functions that
# need them to avoid a hard import-time dependency on every optional
# provider SDK for callers that only need the state-machine logic
# (e.g. unit tests of `next_status_after_failure`).
def _transient_exception_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, OSError]
    try:
        import groq

        types.append(groq.APIError)
    except ImportError:
        pass
    try:
        from google.genai import errors as genai_errors

        types.append(genai_errors.APIError)
    except ImportError:
        pass
    try:
        from pinecone.exceptions import PineconeException

        types.append(PineconeException)
    except ImportError:
        pass
    try:
        from app.rag.llm import LLMProviderError

        types.append(LLMProviderError)
    except ImportError:
        pass
    return tuple(types)


def is_transient_error(exc: BaseException) -> bool:
    """Whether `exc` looks like a transient (network/upstream-provider)
    failure worth retrying, as opposed to a permanent one (bad input —
    `ValueError`, `app.documents.pipeline.DocumentParseError`, a
    `SystemExit` for "no ingestable files") that would fail identically
    on every retry and should never be auto-retried."""
    from app.documents.pipeline import DocumentParseError

    if isinstance(exc, (DocumentParseError, ValueError, SystemExit)):
        return False
    return isinstance(exc, _transient_exception_types())


def next_status_after_failure(attempt_count: int, max_attempts: int, transient: bool) -> JobStatus:
    """Pure state-machine decision, unit-testable without a DB/network:
    a transient failure that hasn't exhausted `max_attempts` yet becomes
    RETRYING (eligible for another attempt); anything else becomes
    FAILED (terminal, requires an explicit manual retry via
    `POST /documents/{id}/reingest`)."""
    if transient and attempt_count < max_attempts:
        return JobStatus.RETRYING
    return JobStatus.FAILED


def create_job(
    session,
    *,
    filename: str,
    source_root: str = "uploads",
    owner_id: int | None = None,
    document_id: int | None = None,
    version_id: int | None = None,
    max_attempts: int | None = None,
    backup_key: str | None = None,
    restore_to_key: str | None = None,
    settings: Settings | None = None,
) -> IngestionJob:
    """Create a new job in QUEUED state. Does not commit — caller controls
    the transaction (matches every other `session_scope`-based write in
    this codebase).

    `backup_key`/`restore_to_key`: see `app/db/models.py::IngestionJob`
    and `run_job`'s docstring, "Replacement-file backup/restore" — set
    together by `app/api/documents.py::upload_documents` when this job
    is REPLACING an existing, previously-successful filename; left `None`
    (the default) for a brand-new filename, where there is nothing to
    restore.
    """
    settings = settings or get_settings()
    job = IngestionJob(
        filename=filename,
        source_root=source_root,
        owner_id=owner_id,
        document_id=document_id,
        version_id=version_id,
        status=JobStatus.QUEUED.value,
        attempt_count=0,
        max_attempts=max_attempts or settings.ingestion_max_attempts,
        backup_key=backup_key,
        restore_to_key=restore_to_key,
    )
    session.add(job)
    session.flush()
    return job


def run_job(job_id: int, settings: Settings | None = None) -> IngestionJob:
    """Execute one ingestion job: QUEUED/RETRYING -> PROCESSING -> COMPLETED
    | RETRYING | FAILED. Each state transition is committed in its own
    short transaction (same "audit row survives even if the work itself
    rolls back" pattern `app/services/ingestion_service.py::
    _record_ingestion_run` already uses), so a crash mid-run leaves the
    job visibly PROCESSING (detectable/re-claimable) rather than silently
    stuck QUEUED forever.

    **Replacement-file backup/restore** (Phase 5 completion pass, round
    3, item 1): if this job's `backup_key`/`restore_to_key` are set (see
    `app/db/models.py::IngestionJob` — populated by
    `app/api/documents.py::upload_documents` when a job is REPLACING an
    existing, previously-successful filename), the pre-upload bytes at
    `backup_key` are restored onto `restore_to_key` when this job reaches
    a TERMINAL FAILED state (not RETRYING — a job that may still succeed
    on its next attempt must not have its in-flight replacement reverted
    out from under it), and discarded (deleted from storage) once the job
    reaches COMPLETED. This is deliberately done HERE, inside `run_job`
    itself — not in the caller that created the job — so the guarantee
    holds no matter WHICH execution path actually processes this job:
    the standalone worker (`app.services.ingestion_jobs.worker`, the
    primary/durable mechanism as of this pass), `poll_and_process_once`'s
    `--once` CLI mode, or (for `POST /documents/{id}/reingest`, which
    still uses `BackgroundTasks` — see `app/api/ingestion_jobs.py`'s
    module docstring for why that one case is fine) an in-process
    background task. One code path, three callers, no duplicated logic.
    """
    settings = settings or get_settings()
    init_db(settings)
    factory = get_session_factory(settings)

    session = factory()
    try:
        job = session.get(IngestionJob, job_id)
        if job is None:
            raise ValueError(f"ingestion job {job_id} not found")
        if job.status == JobStatus.CANCELLED.value:
            return job
        job.status = JobStatus.PROCESSING.value
        job.attempt_count += 1
        import datetime as dt

        job.started_at = dt.datetime.now(dt.timezone.utc)
        session.commit()
    finally:
        session.close()

    started = time.monotonic()
    try:
        from app.services.ingestion_service import ingest_corpus

        summary = ingest_corpus(settings=settings)
        error: str | None = None
    except BaseException as exc:  # noqa: BLE001
        summary = None
        error = str(exc)
        transient = is_transient_error(exc)
        _log.warning(
            "ingestion_job id=%s attempt=%s failed transient=%s error=%s",
            job_id,
            "?",
            transient,
            error,
        )
    else:
        transient = False

    session = factory()
    try:
        import datetime as dt

        job = session.get(IngestionJob, job_id)
        if job is None:
            raise ValueError(f"ingestion job {job_id} not found")
        job.duration_seconds = round(time.monotonic() - started, 3)
        job.finished_at = dt.datetime.now(dt.timezone.utc)

        if error is None:
            job.status = JobStatus.COMPLETED.value
            job.error = ""
            if summary:
                job.chunks_total = summary.get("chunks", 0)
                job.chunks_changed = summary.get("changed", 0)
                job.vectors_upserted = summary.get("upserted", 0)
            # Best-effort: attach the document/version this job's
            # filename now resolves to, if not already set (covers the
            # "job created before the Document row existed" case).
            if job.document_id is None:
                doc = (
                    session.query(Document)
                    .filter(
                        Document.source_root == job.source_root,
                        Document.filename == job.filename,
                        Document.owner_id == job.owner_id,
                    )
                    .one_or_none()
                )
                if doc is not None:
                    job.document_id = doc.id
                    job.version_id = doc.current_version_id
        else:
            job.error = error[:2000]  # never store an unbounded stack trace
            job.status = next_status_after_failure(job.attempt_count, job.max_attempts, transient).value

        backup_key = job.backup_key
        restore_to_key = job.restore_to_key
        terminal_status = job.status

        session.commit()
        session.refresh(job)
        session.expunge(job)
    finally:
        session.close()

    # Storage side-effect happens AFTER the job row is committed and the
    # session closed — a failure here (e.g. the storage backend is briefly
    # unreachable) must not roll back the job status transition that just
    # correctly recorded what actually happened. Best-effort, logged, never
    # raised past this point: `backup_key` staying around a little longer
    # than ideal on a storage hiccup is a much smaller problem than a job
    # whose recorded outcome doesn't match what `ingest_corpus()` actually
    # did.
    if backup_key:
        try:
            from app.storage.base import get_backend

            backend = get_backend(settings)
            if terminal_status == JobStatus.COMPLETED.value:
                backend.delete(backup_key)
            elif terminal_status == JobStatus.FAILED.value and restore_to_key:
                backend.save(restore_to_key, backend.read(backup_key))
                backend.delete(backup_key)
            # RETRYING: backup_key is left in place untouched — this job
            # may still succeed on a future attempt, at which point this
            # same branch runs again with the (unchanged) backup_key.
        except Exception:  # noqa: BLE001 - see comment above
            _log.exception(
                "failed to reconcile replacement-file backup for job_id=%s (backup_key=%s, status=%s)",
                job_id,
                backup_key,
                terminal_status,
            )

    return job


def _reclaim_stale_processing_jobs(session, settings: Settings, is_postgres: bool) -> int:
    """Reclaim jobs stuck in PROCESSING because their worker died mid-run
    (Phase 5 completion pass, round 4, item 2 — "stale-processing
    recovery", added because it was genuinely needed: `run_job` commits
    the PROCESSING transition in its own short transaction before doing
    any real work — see that function's docstring — so a worker process
    that's killed (OOM, deploy, crash) between that commit and the job's
    terminal-state commit leaves the row visibly PROCESSING forever with
    nothing to ever move it out of that state otherwise).

    Uses the EXISTING `started_at` timestamp (no new column) — a
    PROCESSING job whose `started_at` is older than
    `settings.ingestion_stale_processing_seconds` is presumed dead and is
    reset to QUEUED (if it still has retry budget) or straight to FAILED
    (if `attempt_count` already reached `max_attempts` — the crash itself
    counts as a failed attempt, matching `next_status_after_failure`'s
    existing transient-failure logic rather than granting an extra,
    uncounted attempt for free).

    Returns the number of jobs reclaimed. On PostgreSQL this SELECT also
    uses `FOR UPDATE SKIP LOCKED` (same reasoning as the main claim query
    below) so two worker processes polling at once can't both try to
    reclaim — and therefore both re-process — the same stale job.
    """
    import datetime as dt

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=settings.ingestion_stale_processing_seconds)
    query = session.query(IngestionJob).filter(
        IngestionJob.status == JobStatus.PROCESSING.value,
        IngestionJob.started_at.isnot(None),
        IngestionJob.started_at < cutoff,
    )
    if is_postgres:
        query = query.with_for_update(skip_locked=True)
    stale = query.all()
    for job in stale:
        if job.attempt_count >= job.max_attempts:
            job.status = JobStatus.FAILED.value
            job.error = (
                f"Reclaimed after being stuck in PROCESSING for longer than "
                f"{settings.ingestion_stale_processing_seconds}s (worker likely crashed); "
                f"max_attempts ({job.max_attempts}) already reached."
            )
        else:
            job.status = JobStatus.QUEUED.value
            job.error = (
                f"Reclaimed after being stuck in PROCESSING for longer than "
                f"{settings.ingestion_stale_processing_seconds}s (worker likely crashed); requeued."
            )
    if stale:
        session.commit()
        _log.warning(
            "poll_and_process_once: reclaimed %s stale PROCESSING job(s) (job_ids=%s)",
            len(stale),
            [j.id for j in stale],
        )
    return len(stale)


def poll_and_process_once(settings: Settings | None = None) -> int:
    """Claim and process every currently QUEUED/RETRYING job once
    (reclaiming any stale PROCESSING jobs first — see
    `_reclaim_stale_processing_jobs`). Returns the number of jobs
    processed (including reclaimed jobs that got requeued and then
    picked up in this same call).

    On PostgreSQL, claims jobs with `SELECT ... FOR UPDATE SKIP LOCKED`
    so multiple worker processes can safely run this concurrently. On
    SQLite (no row-level locking support), this degrades to a plain
    query + update — safe for a SINGLE worker process only; see module
    docstring.
    """
    settings = settings or get_settings()
    init_db(settings)
    factory = get_session_factory(settings)
    session = factory()
    is_postgres = str(session.bind.dialect.name) == "postgresql" if session.bind is not None else False
    if not is_postgres:
        _log.info(
            "poll_and_process_once: non-PostgreSQL engine (%s) — SELECT ... FOR UPDATE SKIP LOCKED is "
            "unavailable; this polling worker is single-worker-only against this database.",
            session.bind.dialect.name if session.bind is not None else "unknown",
        )

    try:
        _reclaim_stale_processing_jobs(session, settings, is_postgres)

        query = session.query(IngestionJob).filter(
            IngestionJob.status.in_([JobStatus.QUEUED.value, JobStatus.RETRYING.value])
        )
        if is_postgres:
            query = query.with_for_update(skip_locked=True)
        claimable = query.all()
        job_ids = [j.id for j in claimable]
        for job in claimable:
            job.status = JobStatus.PROCESSING.value
        session.commit()
    finally:
        session.close()

    for job_id in job_ids:
        run_job(job_id, settings=settings)
    return len(job_ids)


def worker(poll_interval_seconds: float = 5.0, max_iterations: int | None = None) -> None:
    """Standalone polling worker entrypoint:
    `python -m app.services.ingestion_jobs worker`. Loops
    `poll_and_process_once` forever (or `max_iterations` times, for
    tests). See module docstring for the SQLite single-worker caveat.
    """
    settings = get_settings()
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        processed = poll_and_process_once(settings)
        if processed:
            _log.info("ingestion worker: processed %s job(s)", processed)
        iterations += 1
        if max_iterations is None or iterations < max_iterations:
            time.sleep(poll_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the background ingestion job worker.")
    parser.add_argument("--once", action="store_true", help="Process pending jobs once and exit.")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds.")
    args = parser.parse_args()
    if args.once:
        n = poll_and_process_once()
        print(f"Processed {n} job(s).")
    else:
        worker(poll_interval_seconds=args.interval)


if __name__ == "__main__":
    main()