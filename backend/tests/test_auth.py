"""Tests for app/api/auth.py and app/security/auth.py.

No mocking of the auth machinery itself — these exercise real bcrypt
hashing and real JWT encode/decode against a real (SQLite, per-test)
database, through the actual HTTP endpoints.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.session import reset_engine_for_tests
from app.security.auth import create_access_token, decode_access_token, hash_password, verify_password


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


# --- password hashing (app.security.auth) -----------------------------


def test_hash_password_is_not_plaintext_and_verifies():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("wrong password", hashed) is False


def test_hash_password_uses_a_per_call_salt():
    """Two hashes of the SAME password must differ (bcrypt's per-call
    random salt) -- a fixed/no salt would make identical passwords
    trivially identifiable in a database dump."""
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_verify_password_fails_closed_on_corrupt_hash():
    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


# --- register / login / me (app/api/auth.py) ---------------------------


def test_register_creates_account_and_returns_usable_token(client):
    resp = client.post("/auth/register", json={"email": "alice@example.com", "password": "hunter2pass"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["token_type"] == "bearer"
    token = body["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"
    assert "hashed_password" not in me.json()  # never leaks the hash


def test_register_duplicate_email_is_rejected(client):
    client.post("/auth/register", json={"email": "bob@example.com", "password": "hunter2pass"})
    resp = client.post("/auth/register", json={"email": "bob@example.com", "password": "different-pass"})
    assert resp.status_code == 409


def test_register_duplicate_email_case_insensitive(client):
    client.post("/auth/register", json={"email": "Carol@Example.com", "password": "hunter2pass"})
    resp = client.post("/auth/register", json={"email": "carol@example.com", "password": "different-pass"})
    assert resp.status_code == 409


def test_register_rejects_short_password(client):
    resp = client.post("/auth/register", json={"email": "dave@example.com", "password": "short"})
    assert resp.status_code == 422


def test_login_with_correct_credentials_returns_token(client):
    client.post("/auth/register", json={"email": "erin@example.com", "password": "hunter2pass"})
    resp = client.post("/auth/login", json={"email": "erin@example.com", "password": "hunter2pass"})
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_wrong_password_is_rejected_with_generic_message(client):
    client.post("/auth/register", json={"email": "frank@example.com", "password": "hunter2pass"})
    resp = client.post("/auth/login", json={"email": "frank@example.com", "password": "wrong-password"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password."


def test_login_nonexistent_email_gives_the_same_generic_message(client):
    """Login errors must not distinguish 'no such account' from 'wrong
    password' -- that distinction lets an attacker enumerate registered
    emails (see app/api/auth.py's module docstring)."""
    resp = client.post("/auth/login", json={"email": "nobody@example.com", "password": "whatever12"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password."


def test_me_without_token_is_401(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_with_garbage_token_is_401(client):
    resp = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


# --- token issuance/verification (app.security.auth) --------------------


def test_create_and_decode_access_token_round_trips():
    settings = Settings(_env_file=None)

    class _FakeUser:
        id = 42
        email = "grace@example.com"

    token = create_access_token(_FakeUser(), settings)
    payload = decode_access_token(token, settings)
    assert payload["sub"] == "42"
    assert payload["email"] == "grace@example.com"


def test_decode_access_token_rejects_expired_token():
    settings = Settings(_env_file=None).model_copy(update={"jwt_expire_minutes": 1})

    class _FakeUser:
        id = 1
        email = "henry@example.com"

    # Build an already-expired token directly (not waiting 60s in a test).
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    payload = {"sub": "1", "email": "henry@example.com", "iat": now - dt.timedelta(hours=2), "exp": now - dt.timedelta(hours=1)}
    token = jwt.encode(payload, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token, settings)


def test_decode_access_token_rejects_token_signed_with_different_secret():
    from pydantic import SecretStr

    settings_a = Settings(_env_file=None).model_copy(update={"jwt_secret_key": SecretStr("secret-a")})
    settings_b = Settings(_env_file=None).model_copy(update={"jwt_secret_key": SecretStr("secret-b")})

    class _FakeUser:
        id = 1
        email = "iris@example.com"

    token = create_access_token(_FakeUser(), settings_a)
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(token, settings_b)
