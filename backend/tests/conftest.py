"""Shared pytest fixtures.

`_isolate_settings_from_real_dotenv_file` is autouse and runs BEFORE
every other fixture (autouse fixtures with no explicit dependency run in
definition order within a module, and this is the first one defined) —
see its own docstring for the real bug this exists to prevent.

`_disable_rate_limiting_by_default` is autouse: the Phase 5 completion
pass added `app.core.rate_limit.RateLimitMiddleware` (section 6, applied
to `/auth/*`, `/documents/upload`, `/documents*`, `/ask`). Enabled by
default in the *application*, it would otherwise start rejecting
requests with 429s partway through any existing test file that posts to
one of those routes more than a handful of times (e.g.
`tests/test_auth.py`'s many register/login calls, `tests/test_api.py`'s
repeated `/ask` calls) — this fixture keeps every pre-existing test
running exactly as before. Rate limiting itself is exercised directly
and explicitly in `tests/test_rate_limiting.py`, which re-enables it
(and lowers the limits) for just those tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings_from_real_dotenv_file(monkeypatch):
    """Prevent a real `.env` file on the machine running the tests from
    leaking real credentials into the test process.

    **The bug this fixes** (see docs/architecture.md, "Test isolation
    from .env" for the full writeup): `app.core.config.Settings` sets
    `model_config = SettingsConfigDict(env_file=".env", ...)`, so
    `Settings()` — and therefore the `lru_cache`d `get_settings()` every
    fixture/route ultimately calls — automatically reads a `.env` file
    from the current working directory (i.e. wherever `pytest` is
    invoked from — normally the `backend/` project root, exactly where a
    real developer's own `.env` with real API keys normally lives).

    `monkeypatch.setenv`/`delenv` only ever touch `os.environ`. pydantic-
    settings' precedence is constructor kwargs > OS environment variables
    > `.env` file > field defaults — so `monkeypatch.delenv("PINECONE_API_KEY")`
    does NOT guarantee `settings.pinecone_api_key` is `None`; if a real
    `.env` file also defines `PINECONE_API_KEY`, removing the OS env var
    just stops shadowing that `.env` value, and `Settings()` falls
    through to it. Any test that asserts a setting is ABSENT (`is None`,
    `is False`, "no provider keys configured", etc.) rather than
    explicitly setting it to a known value is vulnerable to this —
    confirmed via `tests/test_api.py::test_ready_reports_not_ready_without_provider_keys`
    and `tests/test_free_mode_e2e.py::test_settings_have_no_pinecone_s3_or_redis_configured`,
    both of which fail exactly this way on a machine with a populated
    `backend/.env` (reproduced directly: `settings.pinecone_api_key`
    came back as a real `SecretStr` instead of `None`, and `/ready`
    reported `True` instead of `False`), while passing in any CI/sandbox
    environment with no `.env` file — which is why this was never caught
    there.

    **The fix**: patch `Settings.model_config`'s `env_file` to `None` for
    the whole test session, so `Settings()`/`get_settings()` NEVER reads
    a real `.env` file during tests, regardless of who runs them or what
    their local `backend/.env` contains — every test's environment
    becomes exactly what its own `monkeypatch.setenv` calls (plus
    whatever's already in `os.environ`, e.g. real CI secrets
    intentionally exported for a specific job) put there, nothing more.
    This is the SAME idiom `tests/test_config.py` already uses per-call
    (`Settings(_env_file=None)`) — applied once, globally, here, instead
    of requiring every fixture that asserts an absence to remember to
    opt in individually. Production is completely unaffected: this only
    ever patches the class attribute for the duration of a test (via
    `monkeypatch`, auto-reverted), never touches the real `env_file=".env"`
    a running application (or `alembic`, which loads `Settings`
    separately — see `alembic/env.py`) uses.
    """
    from app.core.config import Settings, get_settings

    patched_config = dict(Settings.model_config)
    patched_config["env_file"] = None
    monkeypatch.setattr(Settings, "model_config", patched_config)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _disable_rate_limiting_by_default(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    from app.core.config import get_settings
    from app.core.rate_limit import reset_rate_limiter_for_tests

    get_settings.cache_clear()
    reset_rate_limiter_for_tests()
    yield
    reset_rate_limiter_for_tests()
