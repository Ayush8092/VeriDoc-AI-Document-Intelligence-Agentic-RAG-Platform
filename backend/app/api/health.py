"""Liveness/readiness endpoints.

`/health` never touches Pinecone/Gemini/Groq/the DB — pure liveness.
`/ready` actually tries to build the RAG service, so it can distinguish
"process is up" from "can actually answer a question right now".
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import ValidationError
from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.schemas.documents import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])
_log = logging.getLogger(__name__)


def _short(exc: Exception, limit: int = 300) -> str:
    text_ = str(exc)
    return text_ if len(text_) <= limit else text_[:limit].rstrip() + "…"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _check_database(settings: Settings) -> tuple[bool, str]:
    """A real connectivity check (`SELECT 1`) against `DATABASE_URL`,
    independent of whether Pinecone/Gemini/Groq are configured — item 1
    of the Phase 5 completion pass. Opens and closes its own short-lived
    connection rather than reusing `session_scope` so a broken engine
    (e.g. an unreachable PostgreSQL host) surfaces as a normal `False`
    result here instead of raising through unrelated code paths.
    """
    try:
        from app.db.session import get_engine

        engine = get_engine(settings)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        _log.warning("readiness check: database unreachable: %s: %s", type(exc).__name__, exc)
        return False, f"{type(exc).__name__}: {_short(exc)}"


def _check_vector_store(settings: Settings) -> tuple[bool, str]:
    """Whether Pinecone is reachable/usable — reuses the existing
    `get_service()` singleton-builder (already does this work for /ask)
    rather than duplicating client construction here. Surfaced as its
    own named check instead of the previous all-or-nothing failure.
    """
    from app.api.ask import get_service  # local import avoids a circular import at module load

    try:
        get_service()
        return True, "ok"
    except ValidationError as exc:
        return False, f"Configuration error: {_short(exc)}"
    except RuntimeError as exc:
        return False, _short(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {_short(exc)}"


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Itemized readiness: each dependency is checked independently and
    reported in `checks`, so a caller can tell WHICH dependency is down
    instead of just "something is wrong". `ready` is the AND of every
    check — unchanged contract for any existing caller that only reads
    `ready`/`detail`.
    """
    settings = get_settings()

    db_ok, db_detail = _check_database(settings)
    vector_ok, vector_detail = _check_vector_store(settings)

    checks = {"database": db_ok, "vector_store": vector_ok}
    overall_ready = db_ok and vector_ok

    details = []
    if not db_ok:
        details.append(f"database: {db_detail}")
    if not vector_ok:
        details.append(f"vector_store: {vector_detail}")
    if overall_ready:
        detail = (
            f"index={settings.pinecone_index_name} namespace={settings.pinecone_namespace} "
            f"model={settings.answer_model}"
        )
    else:
        detail = "; ".join(details)

    return ReadyResponse(ready=overall_ready, detail=detail, checks=checks)
