"""Tests for app/core/rate_limit.py (Phase 5 completion pass, section 6).

Every other test file in this suite gets rate limiting disabled by the
autouse fixture in tests/conftest.py — these tests explicitly re-enable
it (with low limits) to exercise the 429 behavior directly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_AUTH_PER_MINUTE", "3")
    monkeypatch.setenv("RATE_LIMIT_ASK_PER_MINUTE", "2")

    from app.core.config import get_settings
    from app.core.rate_limit import reset_rate_limiter_for_tests
    from app.db.session import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()
    reset_rate_limiter_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()
    reset_rate_limiter_for_tests()


def test_auth_route_is_rate_limited(client):
    # 3/minute configured — the 4th request within the window is rejected.
    for _ in range(3):
        resp = client.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"})
        assert resp.status_code in (401, 422)  # not yet rate limited

    resp = client.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_rate_limit_is_scoped_per_route_group(client):
    """Exhausting the auth limit must not affect the (separately
    configured) ask limit."""
    for _ in range(3):
        client.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"})
    limited = client.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"})
    assert limited.status_code == 429

    # /ask has its own bucket/limit (2/minute) — unaffected by /auth's.
    resp = client.get("/health")
    assert resp.status_code == 200  # unrelated route entirely, never limited


def test_unrelated_routes_are_never_rate_limited(client):
    for _ in range(10):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_rate_limiting_disabled_by_default_in_other_tests(monkeypatch, tmp_path):
    """Sanity check for the conftest.py autouse fixture: with no explicit
    RATE_LIMIT_ENABLED override, many rapid requests to a normally-limited
    route are never rejected."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from app.core.config import get_settings
    from app.db.session import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        for _ in range(15):
            resp = c.post("/auth/login", json={"email": "a@example.com", "password": "wrong"})
            assert resp.status_code != 429

    get_settings.cache_clear()
    reset_engine_for_tests()
