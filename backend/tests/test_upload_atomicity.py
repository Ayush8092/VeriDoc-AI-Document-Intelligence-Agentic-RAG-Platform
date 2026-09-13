"""Regression tests for the upload workflow's consistency guarantees.

**Round 3 architecture** (the current, real behavior — see
`app/api/documents.py`'s module docstring and
`app/services/ingestion_jobs.py`'s): `POST /documents/upload` no longer
runs `ingest_corpus()` inline, and no longer schedules a FastAPI
`BackgroundTasks` callback either (that was round 2, since superseded).
It validates, durably persists the file(s) + one QUEUED `IngestionJob`
row per file, and returns `202` immediately — nothing further happens in
that request or that process. The STANDALONE WORKER
(`python -m app.services.ingestion_jobs worker`, or its underlying
`poll_and_process_once`/`run_job` functions) is the only thing that
actually runs `ingest_corpus()` and advances a job to a terminal state.

This means a test can no longer just call `client.post(...)` and
immediately assert on a job's OUTCOME — the outcome doesn't exist yet.
Every test below therefore does exactly the two steps that together
model what production actually does:
  1. `client.post("/documents/upload", ...)` — assert `202` + `queued`.
  2. `_run_queued_job(job_id)` — directly call `run_job` (the same
     function the standalone worker calls per job) to synchronously
     drive that one job to its terminal state, exactly as the worker
     would on its own polling loop, just without the wait.

What's preserved from before this rewrite:
  - A failed ingestion must never leave `GET /documents` reporting a
    document that was never actually indexed.
  - A failed re-upload of an EXISTING filename must never permanently
    destroy that filename's previously-successful content (the Phase
    3B/4 "duplicate-filename replacement bug" fix) — now enforced by
    `run_job`'s backup/restore logic once the job reaches its terminal
    FAILED state, rather than synchronously in the request.

What's different (redefined, not dropped — see module docstring above):
  - A failed ingestion of a BRAND NEW filename no longer deletes the
    uploaded file — it stays in storage, unindexed, with the job marked
    FAILED, so it can be inspected or retried
    (`POST /documents/{id}/reingest` — though for a document that was
    never actually created, since ingestion never succeeded once,
    there's no `document_id` to retry against yet; re-uploading the
    same filename is the practical recovery path).
  - The response code for an accepted batch is `202`, not `200`/`503` —
    the eventual success/failure is not knowable at response time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import reset_engine_for_tests


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    # CORPUS_DIR must share UPLOAD_DIR's parent for
    # app.storage.base.validate_storage_config's unconditional
    # local-root-consistency check to pass (both are now storage-
    # abstracted — see that function's docstring, check 2) — this test
    # never touches the corpus directory itself, but app startup
    # (this fixture boots the real app via TestClient) still validates
    # storage config regardless of which directory a given test cares
    # about.
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


def _job_status(client, job_id: int) -> dict:
    resp = client.get(f"/documents/jobs/{job_id}")
    assert resp.status_code == 200
    return resp.json()


def _run_queued_job(job_id: int) -> dict:
    """Synchronously drive job `job_id` to its terminal state, exactly as
    the standalone worker's `run_job` would — see module docstring.
    Returns the job as a plain dict (mirroring what `GET
    /documents/jobs/{id}` returns) so callers can assert on it the same
    way regardless of whether they got it from this helper or from an
    API call.
    """
    from app.services.ingestion_jobs import run_job

    job = run_job(job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "error": job.error,
        "attempt_count": job.attempt_count,
    }


def test_failed_ingestion_marks_job_failed_but_keeps_file_for_retry(client, monkeypatch, tmp_path):
    import app.services.ingestion_service as ingestion_service_module

    def _boom(*, settings):
        raise RuntimeError("Pinecone index unavailable")

    # `run_job` (what actually processes a queued job — see module
    # docstring) imports `ingest_corpus` from `app.services.
    # ingestion_service` directly (a local import inside the function),
    # so that's the module to patch — NOT `app.api.documents`, which
    # only still calls `ingest_corpus` from the separate
    # `POST /documents/{id}/reingest` endpoint, not from upload.
    monkeypatch.setattr(ingestion_service_module, "ingest_corpus", _boom)

    resp = client.post(
        "/documents/upload",
        files={"files": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert len(body["jobs"]) == 1

    job = _run_queued_job(body["jobs"][0]["job_id"])
    assert job["status"] == "failed"
    assert "Pinecone" in job["error"]

    # A brand-new filename's failed job leaves the file in place
    # (redefined atomicity — see module docstring) rather than deleting
    # it; a RuntimeError is treated as non-transient here since it's not
    # in `app.services.ingestion_jobs`'s transient-exception allowlist.
    upload_dir = tmp_path / "uploads"
    remaining = [p for p in upload_dir.glob("*") if p.is_file() and p.name != ".gitkeep"]
    assert [p.name for p in remaining] == ["notes.txt"]

    # But GET /documents must still not report a document that was
    # never actually indexed — Document rows are only created by a
    # successful ingest_corpus() run, which never happened here.
    docs_resp = client.get("/documents")
    assert docs_resp.json() == {"documents": []}


def test_failed_ingestion_does_not_disturb_prior_successful_upload(client, monkeypatch, tmp_path):
    """A second, failing upload batch must not remove or corrupt files
    from an earlier, already-completed batch."""
    import app.services.ingestion_service as ingestion_service_module

    good_summary = {
        "changed": 1,
        "upserted": 1,
        "skipped_unchanged": 0,
        "stale_deleted": 0,
        "chunks": 1,
        "namespace_vector_count": 1,
    }
    monkeypatch.setattr(ingestion_service_module, "ingest_corpus", lambda *, settings: good_summary)

    first = client.post(
        "/documents/upload",
        files={"files": ("first.txt", b"first file content", "text/plain")},
    )
    assert first.status_code == 202
    first_job_id = first.json()["jobs"][0]["job_id"]
    assert _run_queued_job(first_job_id)["status"] == "completed"

    def _boom(*, settings):
        raise RuntimeError("Pinecone index unavailable")

    monkeypatch.setattr(ingestion_service_module, "ingest_corpus", _boom)

    second = client.post(
        "/documents/upload",
        files={"files": ("second.txt", b"second file content", "text/plain")},
    )
    assert second.status_code == 202
    second_job_id = second.json()["jobs"][0]["job_id"]
    assert _run_queued_job(second_job_id)["status"] == "failed"

    upload_dir = tmp_path / "uploads"
    remaining = {p.name for p in upload_dir.glob("*") if p.is_file() and p.name != ".gitkeep"}
    # Both files are present — first.txt because it succeeded, second.txt
    # because a brand-new filename's failed job keeps the file for retry
    # (see the test above). first.txt's content is what matters here.
    assert remaining == {"first.txt", "second.txt"}
    assert (upload_dir / "first.txt").read_bytes() == b"first file content"


def test_failed_reupload_of_existing_filename_restores_original_bytes_once_terminally_failed(
    client, monkeypatch, tmp_path
):
    """The Phase 3B/4 bug this test has always guarded against: uploading
    a file with a filename that already exists, where the new upload's
    ingestion fails, must never permanently destroy the first
    (successful) file's content — see this module's docstring,
    "What's preserved from before this rewrite".
    """
    import app.services.ingestion_service as ingestion_service_module

    good_summary = {
        "changed": 1,
        "upserted": 1,
        "skipped_unchanged": 0,
        "stale_deleted": 0,
        "chunks": 1,
        "namespace_vector_count": 1,
    }
    monkeypatch.setattr(ingestion_service_module, "ingest_corpus", lambda *, settings: good_summary)

    original_bytes = b"ORIGINAL CONTENT - must survive"
    first = client.post(
        "/documents/upload",
        files={"files": ("report.txt", original_bytes, "text/plain")},
    )
    assert first.status_code == 202
    assert _run_queued_job(first.json()["jobs"][0]["job_id"])["status"] == "completed"

    upload_dir = tmp_path / "uploads"
    filename = "report.txt"
    assert (upload_dir / filename).read_bytes() == original_bytes

    def _boom(*, settings):
        raise RuntimeError("Pinecone index unavailable")

    monkeypatch.setattr(ingestion_service_module, "ingest_corpus", _boom)

    replacement_bytes = b"REPLACEMENT CONTENT - ingestion will fail"
    second = client.post(
        "/documents/upload",
        files={"files": (filename, replacement_bytes, "text/plain")},
    )
    assert second.status_code == 202
    second_job_id = second.json()["jobs"][0]["job_id"]
    assert _run_queued_job(second_job_id)["status"] == "failed"

    # The critical assertion: once the replacement's job is terminally
    # FAILED, the file on disk must be restored to the ORIGINAL content —
    # not left as the failed replacement, and not deleted.
    assert (upload_dir / filename).exists(), "the original, previously-successful file must still exist"
    assert (upload_dir / filename).read_bytes() == original_bytes, (
        "a failed re-upload of an existing filename permanently destroyed the previously valid document"
    )

    on_disk_names = {p.name for p in upload_dir.glob("*") if p.is_file() and p.name != ".gitkeep"}
    assert on_disk_names == {filename}

    docs_resp = client.get("/documents")
    assert docs_resp.status_code == 200


def test_job_creation_failure_leaves_nothing_queued(client, monkeypatch):
    """Item 2: "Do not return success if the job was not actually
    persisted." If `_create_jobs_for_batch` cannot create any job rows
    at all (whether by raising, or — the bug fixed alongside this test —
    by returning an empty list WITHOUT raising), the request must fail
    loudly (503) rather than silently claim files were queued when
    nothing was."""
    from app.api import documents as documents_module

    monkeypatch.setattr(documents_module, "_create_jobs_for_batch", lambda *a, **k: [])

    resp = client.post(
        "/documents/upload",
        files={"files": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 503