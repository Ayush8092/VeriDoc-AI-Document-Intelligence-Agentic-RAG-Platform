"""Real Redis integration tests for `app.core.rate_limit._RedisWindow` —
NOT a fake/mocked client. `tests/test_rate_limiting.py` covers the
middleware's behavior end-to-end but only ever exercises the in-memory
window (`RATE_LIMIT_ENABLED`/Redis are both off by default in tests —
see `tests/conftest.py`); this file is what actually verifies the Redis
-backed sliding-window algorithm itself against a real Redis.

Phase 5 completion pass (final round), item 6: "Test Redis with an
actual Redis-compatible service if practical... Do not claim Redis was
fully verified if it was only mocked."

**Every test in this file is skipped, not failed,** when its
prerequisites aren't available: the `redis` package not installed, or no
reachable Redis configured via `VERIDOC_TEST_REDIS_URL`. This sandbox has
neither — these tests were written but NOT executed here (see the Phase
5 completion report for the honest "implemented, not verified in this
environment" distinction).

To actually run these:

    docker run -p 6379:6379 redis:7-alpine

    pip install redis
    VERIDOC_TEST_REDIS_URL=redis://localhost:6379/0 \\
        pytest tests/test_rate_limit_redis_integration.py -v

`docker-compose.yml`'s `redis` service (see that file) starts exactly
this.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

redis = pytest.importorskip("redis", reason="redis is an optional dependency for RATE_LIMIT_REDIS_URL")

from app.core.rate_limit import _RedisWindow  # noqa: E402

_REDIS_URL = os.environ.get("VERIDOC_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    not _REDIS_URL,
    reason=(
        "VERIDOC_TEST_REDIS_URL not set — no Redis configured for integration testing. "
        "See this module's docstring to run a local Redis."
    ),
)


@pytest.fixture
def window():
    w = _RedisWindow(_REDIS_URL)
    yield w
    w.reset()  # clean up every key this test file's fixture-scoped window ever wrote


def _unique_key() -> str:
    # Every test gets its own key namespace so parallel test runs never
    # interfere with each other, without needing to flush the whole
    # Redis instance (which might be shared with something else).
    return f"integration-test:{uuid.uuid4().hex}"


def test_first_n_requests_under_limit_are_allowed(window):
    key = _unique_key()
    for _ in range(5):
        allowed, retry_after = window.hit(key, limit=5)
        assert allowed is True
        assert retry_after == 0


def test_request_over_limit_is_rejected_with_positive_retry_after(window):
    key = _unique_key()
    for _ in range(3):
        assert window.hit(key, limit=3)[0] is True
    allowed, retry_after = window.hit(key, limit=3)
    assert allowed is False
    assert retry_after > 0


def test_limit_is_scoped_per_key_not_global(window):
    key_a, key_b = _unique_key(), _unique_key()
    for _ in range(3):
        assert window.hit(key_a, limit=3)[0] is True
    # key_a is now exhausted; key_b must be completely unaffected —
    # this is exactly the property that makes the shared Redis window
    # correct for scoping by (route group, client IP) in production.
    assert window.hit(key_a, limit=3)[0] is False
    assert window.hit(key_b, limit=3)[0] is True


def test_concurrent_hits_never_allow_more_than_the_limit(window):
    """The one property a fake/mocked client cannot meaningfully prove:
    that the real `ZADD`/`ZREMRANGEBYSCORE`/`ZCARD` pipeline is actually
    atomic enough under real concurrent access from multiple threads
    (simulating multiple backend replicas hitting the same Redis) that
    the limit is never oversold.
    """
    import concurrent.futures

    key = _unique_key()
    limit = 10
    n_requests = 30

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as pool:
        # Each thread needs its own _RedisWindow instance sharing the
        # same underlying Redis — a single `redis.Redis` client is
        # thread-safe for this usage, but constructing independently
        # mirrors "multiple backend replica processes" more faithfully
        # than sharing one Python object across threads would.
        windows = [_RedisWindow(_REDIS_URL) for _ in range(n_requests)]
        futures = [pool.submit(w.hit, key, limit) for w in windows]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    allowed_count = sum(1 for allowed, _ in results if allowed)
    assert allowed_count == limit, (
        f"expected exactly {limit} of {n_requests} concurrent requests to be allowed, "
        f"got {allowed_count} — the sliding-window pipeline is not correctly atomic"
    )


def test_window_expires_allowing_new_requests_after_the_rolling_period(window, monkeypatch):
    """Verifies eviction actually happens (`ZREMRANGEBYSCORE`) rather
    than just trusting the algorithm's logic in the abstract. Uses a
    real (short) sleep against real Redis TIME rather than mocking time,
    since the whole point of this file is to test against the real
    service — kept short (a few seconds) to stay a fast test, by
    monkeypatching the module's window-length constant down for the
    duration of this one test rather than waiting out the real 60s
    production window.
    """
    import app.core.rate_limit as rate_limit_module

    monkeypatch.setattr(rate_limit_module, "_WINDOW_SECONDS", 2.0)
    key = _unique_key()

    assert window.hit(key, limit=1)[0] is True
    assert window.hit(key, limit=1)[0] is False  # immediately over limit

    time.sleep(2.5)  # let the 2-second window fully roll over

    allowed, _ = window.hit(key, limit=1)
    assert allowed is True, "a new request after the window rolled over must be allowed again"
