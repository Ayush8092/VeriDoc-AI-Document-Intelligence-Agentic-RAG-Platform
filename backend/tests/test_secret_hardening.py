"""Tests for item 2 of the Phase 5 completion pass: refuse to start (raise)
when ENVIRONMENT=production and JWT_SECRET_KEY is still the insecure
default or shorter than 32 bytes. A no-op in development/test.
"""

from __future__ import annotations

import pytest

from app.security.auth import InsecureSecretError, assert_secret_is_safe_for_environment


def _settings(**overrides):
    from app.core.config import Settings

    return Settings(**overrides)


def test_default_environment_is_development_and_permissive():
    settings = _settings()
    assert settings.environment == "development"
    assert_secret_is_safe_for_environment(settings)  # must not raise


def test_production_with_default_secret_raises():
    settings = _settings(environment="production")
    with pytest.raises(InsecureSecretError):
        assert_secret_is_safe_for_environment(settings)


def test_production_with_short_secret_raises():
    settings = _settings(environment="production", jwt_secret_key="short-key")
    with pytest.raises(InsecureSecretError):
        assert_secret_is_safe_for_environment(settings)


def test_production_with_strong_secret_does_not_raise():
    settings = _settings(environment="production", jwt_secret_key="x" * 40)
    assert_secret_is_safe_for_environment(settings)  # must not raise


def test_test_environment_is_permissive_like_development():
    settings = _settings(environment="test")
    assert_secret_is_safe_for_environment(settings)  # must not raise


def test_default_secret_constant_is_never_hand_duplicated_out_of_sync():
    """Regression guard for a real bug caught in review: `Settings.
    jwt_secret_key`'s default was once lengthened (to also stop it
    tripping PyJWT's InsecureKeyLengthWarning) without updating a
    hand-copied `_DEFAULT_JWT_SECRET` string constant in this module,
    which would have silently defeated `assert_secret_is_safe_for_environment`
    for a production deployment that never set JWT_SECRET_KEY at all.
    Fixed by deriving `_DEFAULT_JWT_SECRET` directly from `Settings`'
    own field default instead of a separate literal — this test pins
    that invariant so it can never silently regress again.
    """
    from app.core.config import Settings
    from app.security.auth import _DEFAULT_JWT_SECRET

    assert _DEFAULT_JWT_SECRET == Settings.model_fields["jwt_secret_key"].default.get_secret_value()
    # And the default itself must be long enough that its insecurity is
    # caught by the EXACT-MATCH check, not accidentally by the length
    # check instead (which would mask the exact-match check silently
    # breaking, exactly as it did before this fix).
    assert len(_DEFAULT_JWT_SECRET.encode("utf-8")) >= 32


def test_app_startup_raises_in_production_with_default_secret(monkeypatch, tmp_path):
    """End-to-end: app startup (lifespan) actually refuses to start."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    from app.core.config import get_settings
    from app.db.session import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()

    from fastapi.testclient import TestClient

    from app.main import app

    with pytest.raises(InsecureSecretError):
        with TestClient(app):
            pass

    get_settings.cache_clear()
    reset_engine_for_tests()
