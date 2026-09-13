"""Mandatory free/local-mode proof: the project must start and serve
/health + /ready with NO Pinecone, S3, or Redis credentials anywhere in
the environment — only the free-tier AI provider keys (Gemini/Groq),
which this requirement explicitly allows.

This does NOT make live Gemini/Groq API calls (no network access to
those providers in a CI/sandboxed test environment, and doing so would
need real API keys this environment doesn't have) — it proves what this
requirement actually asks for: that STARTUP and the app's own
health/readiness plumbing never require PINECONE_API_KEY / S3
credentials / a Redis URL. The `vector_store` /ready check
(`app/api/health.py::_check_vector_store`) constructs the real
`DocumentQAService` singleton — including its local vector backend — end
to end, so this exercises the actual free-mode wiring, not a mock of it.

Config note (`app/core/config.py::Settings.effective_storage_local_root`):
`UPLOAD_DIR`/`CORPUS_DIR` are set to sibling `uploads`/`corpus`
directories under one shared temp parent, and `STORAGE_LOCAL_ROOT` is
left at its default — that's the exact shape `effective_storage_local_root`
auto-derives a local storage root from, so this fixture doesn't also
have to set STORAGE_LOCAL_ROOT in lockstep.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def free_mode_client(tmp_path, monkeypatch):
    # The point of this fixture: delete/never-set every paid-service
    # credential, then start the real app.
    for var in (
        "PINECONE_API_KEY",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "RATE_LIMIT_REDIS_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("LOCAL_VECTOR_INDEX_PATH", str(tmp_path / "vector_index"))
    # Explicitly confirm free-mode defaults rather than relying on
    # Settings' own defaults silently matching — if these two ever
    # regress to something else, this test should fail loudly, not
    # pass by accident.
    monkeypatch.setenv("VECTORSTORE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    # Gemini/Groq keys ARE allowed by this requirement ("the only
    # external credentials allowed in the normal free demo") — dummy
    # values here since this test proves STARTUP wiring, not live
    # provider calls.
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-free-tier-key")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-free-tier-key")

    from app.core.config import get_settings
    from app.db.session import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


def test_settings_have_no_pinecone_s3_or_redis_configured(free_mode_client):
    from app.core.config import get_settings

    settings = get_settings()
    assert settings.pinecone_api_key is None
    assert settings.s3_access_key is None
    assert settings.s3_bucket is None
    assert settings.rate_limit_redis_url is None
    assert settings.vectorstore_backend == "local"
    assert settings.storage_backend == "local"


def test_health_works_with_zero_cloud_credentials(free_mode_client):
    resp = free_mode_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_vector_store_check_succeeds_via_local_backend(free_mode_client):
    """The critical assertion: /ready's vector_store check — which
    constructs the real DocumentQAService, including its vector backend
    — must succeed using ONLY the local backend, with no Pinecone key
    anywhere. Before VECTORSTORE_BACKEND=local existed, this same check
    would fail (missing PINECONE_API_KEY) unless VECTORSTORE_BACKEND=local
    was set explicitly; it is now the default."""
    resp = free_mode_client.get("/ready")
    body = resp.json()
    assert body["checks"]["vector_store"] is True, body


def test_upload_document_endpoint_reachable_with_zero_cloud_credentials(free_mode_client):
    """Not a full ingestion (that needs a live Gemini call this sandbox
    can't make) — just confirms the endpoint itself, including its
    storage-backend and job-creation plumbing, never trips over a
    missing Pinecone/S3/Redis credential before it even gets to calling
    the embedding provider."""
    resp = free_mode_client.post(
        "/documents/upload",
        files={"files": ("notes.txt", b"hello free mode", "text/plain")},
    )
    # 202 (queued) is success; the background ingestion job itself may
    # go on to fail against the dummy Gemini key (no live network in
    # this sandbox) — that's a provider-call failure, not a
    # paid-infrastructure dependency, and is exactly why the ingestion
    # job's FAILED/RETRYING states exist.
    assert resp.status_code == 202


def test_compare_endpoint_reachable_with_zero_cloud_credentials(free_mode_client, monkeypatch):
    """Same proof, extended to a Phase 6 reasoning endpoint: /compare
    must not require Pinecone/S3/Redis either — it fails (or succeeds)
    on document resolution / the LLM call, never on missing paid
    infrastructure. Requesting a comparison over documents that don't
    exist yet is expected to 404 — the point is WHICH error it is.
    """
    resp = free_mode_client.post("/compare", json={"topic": "test", "document_ids": [1, 2]})
    assert resp.status_code == 404
    assert "PINECONE" not in resp.text.upper()
    assert "S3" not in resp.text.upper()
    assert "REDIS" not in resp.text.upper()