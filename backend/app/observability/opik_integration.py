"""Optional Opik tracing around the RAG pipeline.

Off by default (`Settings.opik_enabled`). When enabled, `opik.track`
wraps `DocumentQAService.ask` as a single traced call — Opik's own
`@opik.track` decorator, applied at call time rather than import time, so
enabling/disabling it is a pure config change with no code path
difference beyond "wrapped or not".

Opik's SDK is intentionally resilient to a missing/invalid API key: an
unconfigured client logs a warning and drops the trace rather than
raising (verified in this session — see MIGRATION_PLAN.md). This module
adds one more layer of safety on top of that: if `opik` itself can't be
imported (not installed) or anything about wrapping goes wrong, the
unwrapped function still runs. Tracing must never be able to break the
one thing it's observing.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from app.core.config import Settings

_log = logging.getLogger(__name__)

T = TypeVar("T")

_configured_for: tuple[str, str] | None = None  # (api_key_repr, project_name) already opik.configure()'d


def _ensure_configured(settings: Settings) -> None:
    global _configured_for
    key = settings.opik_api_key.get_secret_value() if settings.opik_api_key else ""
    marker = (key, settings.opik_project_name)
    if _configured_for == marker:
        return
    try:
        import opik

        if key:
            opik.configure(api_key=key, workspace=settings.opik_workspace, force=True)
    except Exception as exc:  # noqa: BLE001 - never let observability setup break the app
        _log.warning("Opik configuration failed, continuing untraced: %s", exc)
    _configured_for = marker


def opik_traced_pipeline(settings: Settings, fn: Callable[..., T], *args, **kwargs) -> T:
    """Call `fn(*args, **kwargs)`, wrapped in an Opik trace iff enabled.

    This is a function (not a decorator applied at import time) so it can
    read `settings.opik_enabled` fresh on every call — tests toggle the
    setting per-test via `get_settings.cache_clear()`, and a decorator
    baked in at import time wouldn't see that.
    """
    if not settings.opik_enabled:
        return fn(*args, **kwargs)

    try:
        import opik

        _ensure_configured(settings)
        traced = opik.track(
            name="veridoc.ask",
            type="general",
            project_name=settings.opik_project_name,
        )(fn)
        return traced(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Opik tracing failed, falling back to untraced call: %s", exc)
        return fn(*args, **kwargs)
