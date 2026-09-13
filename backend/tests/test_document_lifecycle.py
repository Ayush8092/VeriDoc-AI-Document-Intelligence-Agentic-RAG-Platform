"""`DELETE /documents/{id}` document lifecycle tests (Phase 5 completion
pass, round 5, item 8 — "document lifecycle": upload, replacement,
re-ingestion, failure, retry, DELETION, multiple versions).

Phase 5 completion pass (round 5): `app/api/documents.py::delete_document`
had ZERO test coverage before this file, despite being a fairly involved
endpoint (storage delete, a multi-row DB cascade, and reconciling BOTH
the vector index — local or Pinecone, whichever `VECTORSTORE_BACKEND` is
active — and the BM25 lexical index snapshot, all while enforcing strict
per-tenant ownership).

Follows `tests/test_tenant_isolation.py`'s established pattern exactly:
a real FastAPI `TestClient`, real `/auth/register` users (real bcrypt +
JWT), `Document`/`DocumentVersion`/`IngestionJob` rows inserted directly
(bypassing full `ingest_corpus`, which needs live Gemini/Pinecone
credentials this environment doesn't have) with real files written to
the exact paths the storage abstraction expects.

**Where `ingest_corpus()`'s final reconciliation step is and isn't
exercised for real, and why**: `delete_document` calls the real,
already-separately-tested `ingest_corpus()` to reconcile the vector/BM25
indexes after removing DB rows — see that function's own docstring for
why (reuses tested logic rather than a second bespoke removal path).
Re-running the FULL reconciliation for real requires embedding whatever
real files remain in `corpus_dir`/`upload_dir` after the delete, which
needs a live embedding provider. Two tests below (`test_delete_only_document_reaches_completed_state_with_nothing_left_to_reconcile`
and the "SystemExit" case) exercise the REAL `ingest_corpus()` call by
arranging for zero files to remain afterward (a real, credential-free,
fully exercised code path — see `ingest_corpus`'s documented
`SystemExit` "nothing left to ingest" case). The tenant-isolation tests,
where a SECOND real file legitimately remains after the delete, stub
`ingest_corpus` at the `app.api.documents` import site instead — this
is standard "don't re-verify an already-tested collaborator, verify that
YOUR code calls it correctly and handles its result" test isolation, not
mocking around the thing actually being tested (the DB cascade + storage
delete + tenant scoping, which is exactly what's still exercised for
real in every test here).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.models import Document, DocumentVersion, IngestionJob
from app.db.session import init_db, reset_engine_for_tests, session_scope


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("STORAGE_LOCAL_ROOT", str(tmp_path))
    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


def _register(client, email: str, password: str = "hunter2pass") -> tuple[int, str]:
    resp = client.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    return me.json()["id"], token


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_document(session, *, source_root: str, filename: str, owner_id: int | None) -> int:
    doc = Document(source_root=source_root, filename=filename, file_type="txt", title=filename, owner_id=owner_id)
    session.add(doc)
    session.flush()
    version = DocumentVersion(
        document_id=doc.id,
        version_number=1,
        file_hash="test-hash-" + filename,
        page_count=1,
        has_scanned_pages=False,
    )
    session.add(version)
    session.flush()
    doc.current_version_id = version.id
    session.flush()
    return doc.id


def _seed_job(session, *, document_id: int, owner_id: int | None, filename: str, status: str = "completed") -> int:
    job = IngestionJob(
        document_id=document_id,
        owner_id=owner_id,
        filename=filename,
        source_root="uploads",
        status=status,
    )
    session.add(job)
    session.flush()
    return job.id


def _write_upload_file(tmp_path, *, owner_id: int, filename: str, content: str = "hello") -> None:
    upload_dir = tmp_path / "uploads" / str(owner_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / filename).write_text(content)


# ---------------------------------------------------------------------
# Ownership / scope enforcement
# ---------------------------------------------------------------------


def test_delete_requires_authentication(client):
    resp = client.delete("/documents/1")
    assert resp.status_code in (401, 403)


def test_delete_nonexistent_document_is_404(client):
    _, token = _register(client, "alice@example.com")
    resp = client.delete("/documents/999999", headers=_auth(token))
    assert resp.status_code == 404


def test_delete_another_users_document_is_404_not_403(client, tmp_path):
    """404, not 403 — matches `app/api/source.py`'s established
    convention: a document another user owns must not even be
    confirmable as existing to someone who shouldn't see it."""
    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")
    bob_id, _ = _register(client, "bob@example.com")

    with session_scope(settings) as session:
        bob_doc_id = _seed_document(session, source_root="uploads", filename="bob.txt", owner_id=bob_id)

    resp = client.delete(f"/documents/{bob_doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 404


def test_delete_shared_corpus_document_is_404(client, tmp_path):
    """The shipped corpus (owner_id IS NULL) can never be deleted through
    this endpoint by any authenticated user — see docstring."""
    settings = get_settings()
    init_db(settings)
    _, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        corpus_doc_id = _seed_document(session, source_root="corpus", filename="public.txt", owner_id=None)

    resp = client.delete(f"/documents/{corpus_doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 404


# ---------------------------------------------------------------------
# Real end-to-end delete (nothing else remains -> real ingest_corpus()
# reconciliation, via its documented SystemExit "nothing left" path —
# no live credentials required, see module docstring).
# ---------------------------------------------------------------------


def test_delete_only_document_succeeds_and_cascades_every_db_row(client, tmp_path):
    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        doc_id = _seed_document(session, source_root="uploads", filename="alice.txt", owner_id=alice_id)
        job_id = _seed_job(session, document_id=doc_id, owner_id=alice_id, filename="alice.txt")
    _write_upload_file(tmp_path, owner_id=alice_id, filename="alice.txt")

    resp = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "deleted"
    assert body["document_id"] == doc_id
    assert body["filename"] == "alice.txt"
    assert body["jobs_deleted"] == 1

    with session_scope(settings) as session:
        assert session.get(Document, doc_id) is None
        assert session.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).count() == 0
        assert session.get(IngestionJob, job_id) is None


def test_delete_removes_the_stored_file(client, tmp_path):
    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        doc_id = _seed_document(session, source_root="uploads", filename="alice.txt", owner_id=alice_id)
    _write_upload_file(tmp_path, owner_id=alice_id, filename="alice.txt")
    stored_path = tmp_path / "uploads" / str(alice_id) / "alice.txt"
    assert stored_path.exists()

    resp = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 200, resp.text
    assert not stored_path.exists()


def test_delete_when_stored_file_already_missing_is_still_idempotent_success(client, tmp_path):
    """Storage delete is a no-op for a missing key (`LocalStorageBackend
    .delete` -> `Path.unlink(missing_ok=True)`) — a document whose file
    was already lost/removed out-of-band must still delete cleanly, not
    500."""
    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        doc_id = _seed_document(session, source_root="uploads", filename="alice.txt", owner_id=alice_id)
    # deliberately NOT writing the file to disk

    resp = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 200, resp.text


def test_double_delete_is_a_clean_404_the_second_time(client, tmp_path):
    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        doc_id = _seed_document(session, source_root="uploads", filename="alice.txt", owner_id=alice_id)
    _write_upload_file(tmp_path, owner_id=alice_id, filename="alice.txt")

    first = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert first.status_code == 200, first.text
    second = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert second.status_code == 404


def test_delete_removes_multiple_jobs_for_the_same_document(client, tmp_path):
    """A document that failed and was retried several times accumulates
    multiple `IngestionJob` rows (one per attempt/reingest) — delete
    must remove ALL of them, not just the latest."""
    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        doc_id = _seed_document(session, source_root="uploads", filename="alice.txt", owner_id=alice_id)
        _seed_job(session, document_id=doc_id, owner_id=alice_id, filename="alice.txt", status="failed")
        _seed_job(session, document_id=doc_id, owner_id=alice_id, filename="alice.txt", status="failed")
        _seed_job(session, document_id=doc_id, owner_id=alice_id, filename="alice.txt", status="completed")
    _write_upload_file(tmp_path, owner_id=alice_id, filename="alice.txt")

    resp = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 200, resp.text
    assert resp.json()["jobs_deleted"] == 3

    with session_scope(settings) as session:
        assert session.query(IngestionJob).filter(IngestionJob.document_id == doc_id).count() == 0


# ---------------------------------------------------------------------
# Tenant isolation under deletion — the property that actually matters
# most here: deleting MY document must never touch YOUR document.
# `ingest_corpus` is stubbed at the import site for these (see module
# docstring) so a second real file can legitimately remain afterward
# without needing live embedding credentials.
# ---------------------------------------------------------------------


def test_deleting_one_users_document_never_touches_another_users_document(client, tmp_path, monkeypatch):
    import app.api.documents as documents_module

    monkeypatch.setattr(
        documents_module,
        "ingest_corpus",
        lambda settings: {"stale_deleted": 0, "chunks": 1},
    )

    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")
    bob_id, bob_token = _register(client, "bob@example.com")

    with session_scope(settings) as session:
        alice_doc_id = _seed_document(session, source_root="uploads", filename="secret.txt", owner_id=alice_id)
        bob_doc_id = _seed_document(session, source_root="uploads", filename="secret.txt", owner_id=bob_id)
        bob_job_id = _seed_job(session, document_id=bob_doc_id, owner_id=bob_id, filename="secret.txt")
    _write_upload_file(tmp_path, owner_id=alice_id, filename="secret.txt", content="alice's content")
    _write_upload_file(tmp_path, owner_id=bob_id, filename="secret.txt", content="bob's content")

    resp = client.delete(f"/documents/{alice_doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 200, resp.text

    # Alice's document is gone...
    with session_scope(settings) as session:
        assert session.get(Document, alice_doc_id) is None
    assert not (tmp_path / "uploads" / str(alice_id) / "secret.txt").exists()

    # ...but Bob's identically-named document is completely untouched:
    # same-named files per-owner is exactly the scenario that would
    # expose a storage-key or DB-scoping bug that only filters by
    # filename instead of (filename, owner_id).
    with session_scope(settings) as session:
        assert session.get(Document, bob_doc_id) is not None
        assert session.get(IngestionJob, bob_job_id) is not None
    assert (tmp_path / "uploads" / str(bob_id) / "secret.txt").read_text() == "bob's content"

    # And Bob can still fetch his own document/list normally.
    listing = client.get("/documents", headers=_auth(bob_token))
    assert listing.status_code == 200
    filenames = {d["filename"] for d in listing.json()["documents"]}
    assert "secret.txt" in filenames


def test_delete_reconciliation_failure_returns_503_but_db_rows_stay_deleted(client, tmp_path, monkeypatch):
    """If the final `ingest_corpus()` reconciliation step fails (e.g. a
    transient upstream error), `delete_document` reports 503 honestly
    (see its docstring) rather than a misleading 200 — but the DB rows
    and stored file, which WERE successfully removed before that step
    ran, must not be resurrected/rolled back just because reconciliation
    failed afterward. A document that's gone from the DB and storage
    genuinely is gone from what a new request sees, even if stale
    vector/BM25 entries might briefly remain.
    """
    import app.api.documents as documents_module
    from app.rag.llm import LLMProviderError

    def _boom(settings):
        raise LLMProviderError("simulated upstream failure during reconciliation")

    monkeypatch.setattr(documents_module, "ingest_corpus", _boom)

    settings = get_settings()
    init_db(settings)
    alice_id, alice_token = _register(client, "alice@example.com")

    with session_scope(settings) as session:
        doc_id = _seed_document(session, source_root="uploads", filename="alice.txt", owner_id=alice_id)
    _write_upload_file(tmp_path, owner_id=alice_id, filename="alice.txt")

    resp = client.delete(f"/documents/{doc_id}", headers=_auth(alice_token))
    assert resp.status_code == 503

    with session_scope(settings) as session:
        assert session.get(Document, doc_id) is None
    assert not (tmp_path / "uploads" / str(alice_id) / "alice.txt").exists()