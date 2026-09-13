"""Authentication foundation (Phase 5).

Password hashing (bcrypt), access-token issuance/verification (JWT,
HS256), and two FastAPI dependencies:

- `get_current_user_optional` — returns the authenticated `User`, or
  `None` if no/invalid `Authorization` header was sent. This is what
  every pre-existing endpoint (`/ask`, `/ask/stream`, `/documents`,
  `/documents/upload`, `/source/*`) is wired to use, so an anonymous
  request behaves EXACTLY as it did before Phase 5 (see
  app.core.config.Settings' authentication docstring) while an
  authenticated request gets tenant-scoped behavior.
- `get_current_user_required` — the same, but raises 401 if there's no
  valid token. Used only by genuinely auth-required endpoints
  (`GET /auth/me`).

Design choices, and why:

- bcrypt over a hand-rolled hash: per-password salt is automatic, the
  work factor is tunable, and it's the same choice virtually every
  production Python auth system makes — no reason to reinvent this.
- JWT (PyJWT, HS256) over server-side sessions: no new stateful store is
  needed (the existing SQLite/PostgreSQL `users` table is enough — the
  token itself carries the claims), which matches the spec's "do not
  over-engineer" guidance for a project this size. A token's claims are
  `sub` (user id, string) and `email`; `exp` is enforced by `jwt.decode`.
- Token invalidation/refresh/rotation is NOT implemented — logout is
  client-side (discard the token) per the spec's "logout/session
  handling" being a FRONTEND concern for this phase, not a server-side
  revocation list. Documented as a known limitation in docs/architecture.md
  rather than silently pretended-away.
"""

from __future__ import annotations

import datetime as dt

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_session_factory

# `auto_error=False`: a missing/malformed Authorization header must NOT
# itself raise — `get_current_user_optional` needs to distinguish "no
# token sent" (fall back to anonymous/public behavior) from "token sent
# but invalid" (still just falls back to anonymous here; only
# `get_current_user_required` turns either case into a 401).
_bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed/corrupt stored hash must fail closed (deny), never
        # raise into a 500 that could leak implementation details.
        return False


def create_access_token(user: User, settings: Settings) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "iat": now,
        "exp": now + dt.timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> dict:
    """Raises `jwt.PyJWTError` (or a subclass — ExpiredSignatureError,
    InvalidTokenError, ...) on any invalid/expired/tampered token. Callers
    in this module catch it; callers elsewhere generally shouldn't need
    to call this directly (use the dependencies below instead)."""
    return jwt.decode(token, settings.jwt_secret_key.get_secret_value(), algorithms=[settings.jwt_algorithm])


def _user_from_token(token: str, settings: Settings) -> User | None:
    try:
        payload = decode_access_token(token, settings)
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        return None

    factory = get_session_factory(settings)
    session: Session = factory()
    try:
        user = session.get(User, user_id)
        if user is None or not user.is_active:
            return None
        # Detach from the session before returning — the session closes
        # at the end of this function, and callers (request-scoped
        # FastAPI dependencies) must not hold a live session across a
        # request. `to_dict()`-safe scalar attributes remain readable on
        # a detached instance because they're already loaded.
        session.expunge(user)
        return user
    finally:
        session.close()


def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> User | None:
    if credentials is None or not credentials.credentials:
        return None
    return _user_from_token(credentials.credentials, settings)


def get_current_user_required(
    current_user: User | None = Depends(get_current_user_optional),
) -> User:
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def allowed_owner_ids(current_user: User | None) -> frozenset[str]:
    """The set of `owner_id` values (as strings, matching the Pinecone/
    lexical-index metadata convention — see app/vectorstore.py's
    `upsert_chunks`) a request is permitted to see: always the shared
    public corpus (`""`, i.e. `owner_id IS NULL`), plus the current
    user's own id when authenticated. Anonymous requests therefore see
    exactly what every pre-Phase-5 request saw — see this module's
    docstring.
    """
    ids = {""}
    if current_user is not None:
        ids.add(str(current_user.id))
    return frozenset(ids)


# ==========================================================
# Secret hardening (Phase 5 completion pass, item 2)
# ==========================================================

#: The obviously-insecure default from `Settings.jwt_secret_key` —
#: intentionally never mistaken for a real secret (see that field's
#: description). Read directly from `Settings`' own field default
#: (rather than a hand-copied literal) specifically to make it
#: IMPOSSIBLE for this check to silently drift out of sync with the
#: real default the way a duplicated string constant did once already
#: (a caught, real bug: an earlier pass lengthened the `Settings`
#: default to stop it tripping PyJWT's own InsecureKeyLengthWarning,
#: without updating what was then a separate hardcoded copy here — which
#: would have silently defeated BOTH checks below at once for a
#: production deployment that never set JWT_SECRET_KEY at all, since the
#: exact-string check no longer matched AND the now-longer string
#: happened to also clear the length check). See
#: tests/test_secret_hardening.py's dedicated regression test.
_DEFAULT_JWT_SECRET = Settings.model_fields["jwt_secret_key"].default.get_secret_value()
_MIN_PRODUCTION_SECRET_BYTES = 32


class InsecureSecretError(RuntimeError):
    """Raised when `Settings.environment == "production"` and
    `JWT_SECRET_KEY` is still the default or too short to be a real
    HMAC secret. Meant to abort application startup — see
    `app/main.py`'s `lifespan` — not to be caught and ignored.
    """


class InsecureDatabaseConfigError(RuntimeError):
    """Raised when `Settings.environment == "production"` and
    `DATABASE_URL` still points at SQLite (Phase 5 completion pass,
    round 4, item 8 — "configuration audit: production does not use
    SQLite"). A SEPARATE exception from `InsecureSecretError` (not a
    broadened version of it) so existing callers/tests that catch
    `InsecureSecretError` specifically are completely unaffected by
    this addition — see `assert_database_is_safe_for_environment`.

    SQLite is genuinely fine for local dev and the test suite (every
    test fixture already uses it — see `tests/test_api.py` etc.) but is
    the wrong choice for a real deployment: no real concurrent-writer
    story (a single file lock serializes every write across the API
    process AND the standalone ingestion worker —
    `app.services.ingestion_jobs`'s own docstring already documents that
    its safe concurrent job-claiming via `SELECT ... FOR UPDATE SKIP
    LOCKED` requires PostgreSQL specifically), and no answer at all for
    multi-host/multi-replica deployment (a local file isn't shared
    across machines).
    """


def assert_secret_is_safe_for_environment(settings: Settings) -> None:
    """Refuse (raise `InsecureSecretError`) if this looks like a
    production deployment still running with a development-grade JWT
    secret. A no-op for every `environment` value other than
    `"production"` (development/test/anything else stay permissive —
    see `Settings.environment`'s docstring), so this never affects local
    dev or the test suite.
    """
    if settings.environment != "production":
        return

    secret = settings.jwt_secret_key.get_secret_value()
    if secret == _DEFAULT_JWT_SECRET:
        raise InsecureSecretError(
            "ENVIRONMENT=production but JWT_SECRET_KEY is still the insecure default. "
            "Set a real, random JWT_SECRET_KEY (at least 32 bytes) via the environment "
            "before starting in production."
        )
    if len(secret.encode("utf-8")) < _MIN_PRODUCTION_SECRET_BYTES:
        raise InsecureSecretError(
            f"ENVIRONMENT=production but JWT_SECRET_KEY is only {len(secret.encode('utf-8'))} "
            f"bytes long (minimum {_MIN_PRODUCTION_SECRET_BYTES} bytes recommended for HMAC-"
            "SHA256). Set a longer, random JWT_SECRET_KEY via the environment before starting "
            "in production."
        )
