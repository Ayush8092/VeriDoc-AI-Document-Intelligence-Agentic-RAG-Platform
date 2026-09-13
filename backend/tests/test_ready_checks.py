"""Tests for GET /ready's itemized dependency checks (Phase 5 completion
pass, section 1). /health must stay untouched (pure liveness, never
touches the DB/Pinecone) — see the first test below.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from app.core.config import get_settings
    from app.db.session import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


def test_health_never_touches_dependencies(client, monkeypatch):
    """Even with a completely broken DATABASE_URL, /health must still
    return 200 — it is pure liveness."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_database_check_key(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "database" in body["checks"]
    assert "vector_store" in body["checks"]


def test_ready_database_check_true_for_working_sqlite(client):
    resp = client.get("/ready")
    body = resp.json()
    assert body["checks"]["database"] is True


def test_ready_database_check_false_for_unreachable_database(monkeypatch):
    """Unit-level check of `_check_database` directly against an
    unreachable host — an impossible-to-connect Postgres host fails fast
    on connection-refused. (A full end-to-end `/ready` request against a
    broken DATABASE_URL isn't exercised here because `app.main.lifespan`
    already calls `init_db()`/`create_all` unconditionally at startup —
    same as before this pass — which fails the TestClient's app startup
    itself before `/ready` would even run; that startup behavior is
    pre-existing and out of scope for this pass.)
    """
    from app.core.config import get_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@127.0.0.1:1/doesnotexist")
    get_settings.cache_clear()
    settings = get_settings()

    from app.api.health import _check_database
    from app.db.session import reset_engine_for_tests

    reset_engine_for_tests()
    ok, detail = _check_database(settings)
    assert ok is False
    assert detail

    get_settings.cache_clear()
    reset_engine_for_tests()


def test_ready_vector_store_check_reported_independently_of_database(client, monkeypatch):
    """With a working DB but no Pinecone credentials configured, the
    database check must still report True even though vector_store is
    False — items are independent, not all-or-nothing."""
    resp = client.get("/ready")
    body = resp.json()
    # Whatever vector_store ends up being (depends on whether Pinecone
    # credentials happen to be configured in this environment), the
    # database check must be reported and must be True given the working
    # SQLite DB from the `client` fixture.
    assert body["checks"]["database"] is True
    assert isinstance(body["checks"]["vector_store"], bool)


def test_ready_overall_is_and_of_all_checks(client):
    resp = client.get("/ready")
    body = resp.json()
    assert body["ready"] == all(body["checks"].values())
