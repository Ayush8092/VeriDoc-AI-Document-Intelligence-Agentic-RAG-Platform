"""API rate limiting (Phase 5 completion pass, item 6 / section 12 of the
Phase 5 spec: "protect expensive operations ... prevent accidental or
malicious request floods ... do not over-engineer this into a full
enterprise gateway").

**Two window backends, chosen by `Settings.rate_limit_redis_url`:**

- **In-memory** (default, `rate_limit_redis_url` unset) — a simple
  in-process, fixed-client-window counter. State lives in this process's
  memory only — NOT shared across multiple backend replicas/workers
  behind a load balancer; each replica enforces its own independent
  limit, so the *effective* aggregate limit scales with replica count.
  Fine for a single-process (or few-process, sharing no state)
  deployment.
- **Redis-backed** (`rate_limit_redis_url` set, Phase 5 completion pass,
  round 2, item 6) — a Redis sorted set sliding window
  (`ZREMRANGEBYSCORE` + `ZCARD` + conditionally `ZADD`), shared by every
  process that points at the same Redis instance — the correct choice
  for a real multi-replica deployment. `redis` is imported LAZILY
  (inside `_RedisWindow.__init__`, matching `app/storage/s3.py`'s
  `boto3` pattern) so this module still imports cleanly with `redis` not
  installed; only actually configuring `rate_limit_redis_url` requires
  it.

  **Runtime-verified against a real local Redis** (Phase 5 completion
  pass, round 4, item 7 — `apt-get install redis-server`, run locally,
  `tests/test_rate_limit_redis_integration.py` run with
  `VERIDOC_TEST_REDIS_URL` pointed at it). That run caught a genuine
  bug in the original two-`pipeline()` version of `hit()`: reading the
  count in one pipeline, branching on it in Python, then `ZADD`-ing in a
  SECOND pipeline leaves a window between the read and the write where
  multiple concurrent requests can all observe "count is under the
  limit" and all add themselves — oversubscribing the limit (measured:
  13 requests allowed through a limit of 10 under concurrent load, not
  the required exactly-10).
  `test_concurrent_hits_never_allow_more_than_the_limit` failed exactly
  this way before the fix below. The fix: the whole
  evict-count-conditionally-add sequence now runs as ONE Lua script via
  `EVAL`, which Redis executes atomically — there is no window for
  another client to interleave. Same sliding-window semantics as
  before (a request older than 60 seconds is evicted; `ZCARD` after
  eviction is the count compared against the limit) — the fix changes
  how atomicity is achieved, not the algorithm.

Limits are enforced per (route group, client IP) using a rolling
60-second window. Four independent route groups, each with its own
configured limit (`app.core.config.Settings`):

- `auth`       — `/auth/*` (register/login — brute-force protection)
- `upload`     — `/documents/upload` (expensive: parsing/OCR/embedding)
- `documents`  — `/documents*` other than upload (listing/source-viewer reads)
- `ask`        — `/ask`, `/ask/stream` (expensive: retrieval + LLM calls)

Any request to a path outside these four groups (e.g. `/health`, `/ready`)
is never rate limited, regardless of configuration.
"""

from __future__ import annotations

import abc
import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings

_log = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0


class _Window(abc.ABC):
    """One sliding-window counter backend. `hit(key, limit)` records one
    request for `key` and returns `(allowed, retry_after_seconds)` —
    `retry_after_seconds` is only meaningful when `allowed` is False.
    """

    @abc.abstractmethod
    def hit(self, key: str, limit: int) -> tuple[bool, int]: ...

    @abc.abstractmethod
    def reset(self) -> None: ...


class _InMemoryWindow(_Window):
    """(group, client_key) -> list of monotonic request timestamps within
    the current rolling window. Process-wide — see module docstring.
    """

    def __init__(self) -> None:
        self._state: dict[str, list[float]] = {}

    def hit(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.monotonic()
        bucket = self._state.setdefault(key, [])
        cutoff = now - _WINDOW_SECONDS
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)

        if len(bucket) >= limit:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - bucket[0])) + 1)
            return False, retry_after

        bucket.append(now)
        return True, 0

    def reset(self) -> None:
        self._state.clear()


class _RedisWindow(_Window):
    """Redis sorted-set sliding window — see module docstring for the
    algorithm and why it was chosen. One sorted set per `key`, member =
    a unique per-request token (monotonic-ish: wall-clock time in
    microseconds + a request-local counter, to avoid two same-millisecond
    requests colliding as the same ZSET member), score = wall-clock
    request time in seconds (Redis TIME, not this process's clock — see
    `hit`'s docstring for why that matters for correctness across
    processes/hosts with clock drift).
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RATE_LIMIT_REDIS_URL is set but the 'redis' package is not installed. "
                "Install it with `pip install redis` (listed in requirements.txt) or unset "
                "RATE_LIMIT_REDIS_URL to use the in-memory limiter instead."
            ) from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=False)
        self._counter = 0

    # The whole "evict expired -> count -> maybe add" sequence has to be
    # ATOMIC from Redis's point of view, not just each individual command
    # — two pipelines (a read pipeline, then a Python-side `if`, then a
    # write pipeline) leaves a window between the count-read and the
    # ZADD where multiple concurrent clients can all observe
    # `count_before < limit` and all add themselves, oversubscribing the
    # limit (confirmed by `tests/test_rate_limit_redis_integration.py::
    # test_concurrent_hits_never_allow_more_than_the_limit` against a
    # real local Redis: a naive two-pipeline version measurably let 13
    # requests through a limit of 10 under concurrent load). A Lua
    # script is the standard fix — Redis executes the whole script as a
    # single atomic operation, so there is no window for another client
    # to interleave.
    _HIT_SCRIPT = """
local redis_key = KEYS[1]
local cutoff = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local member = ARGV[4]
local ttl = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', redis_key, '-inf', cutoff)
local count_before = redis.call('ZCARD', redis_key)

if count_before >= limit then
    local oldest = redis.call('ZRANGE', redis_key, 0, 0, 'WITHSCORES')
    if oldest[2] then
        return {0, oldest[2]}
    end
    return {0, false}
end

redis.call('ZADD', redis_key, now, member)
redis.call('EXPIRE', redis_key, ttl)
return {1, false}
"""

    def hit(self, key: str, limit: int) -> tuple[bool, int]:
        # Redis's own TIME (seconds, microseconds) is used as the score
        # rather than this process's `time.time()` — the whole point of
        # a shared backend is correctness across MULTIPLE processes/hosts,
        # which may have clock drift relative to each other; every
        # process asking Redis for the current time removes that drift
        # as a source of an incorrect window boundary.
        secs, micros = self._client.time()
        now = float(secs) + float(micros) / 1_000_000.0
        self._counter = (self._counter + 1) % 1_000_000
        member = f"{now:.6f}:{self._counter}".encode()

        redis_key = f"veridoc:ratelimit:{key}"
        cutoff = now - _WINDOW_SECONDS

        allowed_flag, oldest_score = self._client.eval(
            self._HIT_SCRIPT,
            1,
            redis_key,
            f"{cutoff:.6f}",
            str(limit),
            f"{now:.6f}",
            member,
            str(int(_WINDOW_SECONDS) + 5),
        )

        if not allowed_flag:
            if oldest_score:
                retry_after = max(1, int(_WINDOW_SECONDS - (now - float(oldest_score))) + 1)
            else:  # pragma: no cover - defensive: allowed_flag==0 implies oldest exists
                retry_after = int(_WINDOW_SECONDS)
            return False, retry_after

        return True, 0

    def reset(self) -> None:
        for k in self._client.scan_iter(match="veridoc:ratelimit:*"):
            self._client.delete(k)


_window_backend: _Window | None = None
_window_backend_url: str | None = "__unset__"  # sentinel distinct from None (explicitly-unset URL)


def _get_window(settings) -> _Window:
    """Lazily construct (and cache) the configured window backend,
    rebuilding it if `rate_limit_redis_url` changes (tests toggle
    settings via `get_settings.cache_clear()` — this must notice).
    Falls back to the in-memory window (logged once, not raised) if
    Redis is configured but `redis`/the connection isn't actually
    available — a rate limiter that fails open due to a Redis outage is
    safer than one that takes the whole API down with it.
    """
    global _window_backend, _window_backend_url

    url = settings.rate_limit_redis_url
    if _window_backend is not None and _window_backend_url == url:
        return _window_backend

    if url:
        try:
            _window_backend = _RedisWindow(url)
            _window_backend_url = url
            return _window_backend
        except Exception as exc:  # noqa: BLE001 - fail open to in-memory, never take the API down
            _log.warning(
                "RATE_LIMIT_REDIS_URL is set but the Redis backend could not be constructed "
                "(%s: %s) — falling back to the in-memory rate limiter for this process.",
                type(exc).__name__,
                exc,
            )

    _window_backend = _InMemoryWindow()
    _window_backend_url = url
    return _window_backend


# Longest-prefix-first: "/documents/upload" must be checked before the
# more general "/documents" so an upload request is grouped as "upload",
# not "documents".
_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/auth", "auth"),
    ("/documents/upload", "upload"),
    ("/documents", "documents"),
    ("/ask", "ask"),
)


def reset_rate_limiter_for_tests() -> None:
    """Clear all rate-limit state (whichever backend is currently
    active) and force the next request to re-resolve which backend to
    use. Called by tests/conftest.py's autouse fixture (before AND after
    every test) and by tests/test_rate_limiting.py directly, so no
    test's request history leaks into another test.
    """
    global _window_backend, _window_backend_url
    if _window_backend is not None:
        try:
            _window_backend.reset()
        except Exception:  # noqa: BLE001 - best-effort cleanup only
            _log.debug("rate limiter reset() failed (non-fatal, likely no Redis reachable)", exc_info=True)
    _window_backend = None
    _window_backend_url = "__unset__"


def _match_group(path: str) -> str | None:
    for prefix, group in _GROUP_PREFIXES:
        if path.startswith(prefix):
            return group
    return None


def _limit_for_group(settings, group: str) -> int | None:
    return {
        "auth": settings.rate_limit_auth_per_minute,
        "upload": settings.rate_limit_upload_per_minute,
        "documents": settings.rate_limit_documents_per_minute,
        "ask": settings.rate_limit_ask_per_minute,
    }.get(group)


def _client_key(request: Request) -> str:
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware: 429 + `Retry-After` once a route group's
    per-client limit is exceeded within the rolling window. A no-op
    entirely when `Settings.rate_limit_enabled` is False (the default in
    every test — see tests/conftest.py) or a route isn't in any group.
    """

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return await call_next(request)

        group = _match_group(request.url.path)
        if group is None:
            return await call_next(request)

        limit = _limit_for_group(settings, group)
        if not limit or limit <= 0:
            return await call_next(request)

        window = _get_window(settings)
        key = f"{group}:{_client_key(request)}"
        allowed, retry_after = window.hit(key, limit)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded for '{group}' requests "
                        f"({limit}/minute). Try again later."
                    )
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)