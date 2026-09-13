"""FastAPI app: assembles the health/ask/documents routers, serves a
minimal static fallback page, and eagerly warms up the RAG service on
startup without blocking it.

The primary UI is the Next.js app in `../frontend` (chat at `/ask`,
document library at `/library`, upload at `/upload`) — see
`../frontend/README.md`. The page served at `GET /` below is a minimal,
dependency-free fallback for running the backend standalone (e.g. a quick
`curl`-free sanity check, or an environment without Node.js), not the
product's frontend.

Endpoints:
GET  /                    - minimal static fallback page (see above; the
                             real UI is ../frontend)
GET  /health               - liveness
GET  /ready                 - readiness (RAG service + Pinecone usable)
POST /ask                    - {"question": "..."} -> {"answer", "found", "citations", "trace"?}
POST /documents/upload         - multipart upload -> ingested into data/uploads/
GET  /documents                - list ingested documents + version metadata
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import ask as ask_router_module
from app.api.ask import router as ask_router
from app.api.ask_stream import router as ask_stream_router
from app.api.auth import router as auth_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.ingestion_jobs import router as ingestion_jobs_router
from app.api.reasoning import router as reasoning_router
from app.api.source import router as source_router
from app.core.config import get_settings
from app.core.rate_limit import RateLimitMiddleware
from app.db.session import init_db
from app.rag.graph import DocumentQAService
from app.security.auth import assert_secret_is_safe_for_environment
from app.storage.base import validate_storage_config

_log = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Phase 5 completion pass, item 3.1: refuse to start rather than
    # silently serve a production deployment on a development-grade JWT
    # secret. Runs FIRST, before init_db/anything else — a fast, clear
    # failure at startup beats a security hole discovered later. A no-op
    # for every `environment` other than "production" (see
    # `assert_secret_is_safe_for_environment`'s docstring); raising here
    # propagates out of `TestClient(app)`/`uvicorn`'s startup exactly as
    # `tests/test_secret_hardening.py::test_app_startup_raises_in_production_with_default_secret`
    # expects.
    assert_secret_is_safe_for_environment(settings)

    # Phase 5 completion pass, item 3.4: refuse to start with a storage
    # configuration that would silently misbehave — see
    # `validate_storage_config`'s docstring for exactly what this checks
    # and why (STORAGE_BACKEND=s3 rejected outright; a STORAGE_LOCAL_ROOT
    # / UPLOAD_DIR / CORPUS_DIR mismatch caught here rather than as a
    # confusing "my upload disappeared" bug later). Runs right after the
    # secret check — both are fast, pre-DB, fail-loud startup guards.
    validate_storage_config(settings)

    init_db(settings)

    async def _warm_up() -> None:
        try:
            loop = asyncio.get_running_loop()
            service = await loop.run_in_executor(None, DocumentQAService)
            ask_router_module.set_service(service)
            _log.info("DocumentQAService initialized at startup.")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "DocumentQAService could not be initialized at startup (%s: %s). "
                "The server is still up; /health stays up, and /ask will retry "
                "initialization on first use.",
                type(exc).__name__,
                exc,
            )

    asyncio.create_task(_warm_up())
    yield


app = FastAPI(
    title="Veridoc Document Intelligence API",
    description=(
        "Document-grounded Q&A over an ingested corpus (PDF, DOCX, TXT/MD, "
        "and scanned/photographed documents via OCR). Answers are restricted "
        "to retrieved chunks; every answer carries citations with page/table "
        "provenance, and unsupported questions are explicitly refused."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Phase 4.1(5): the Source Viewer's page-image endpoint
    # (app/api/source.py::get_page_image) reports rendered/native pixel
    # dimensions via custom response headers — CORS hides non-simple
    # response headers from `fetch()` by default unless explicitly
    # exposed, so without this the frontend's bbox projection
    # (frontend/lib/bbox.ts) would silently see `NaN` for every
    # dimension on a cross-origin deployment (dev already works because
    # NEXT_PUBLIC_API_BASE_URL commonly points at the same origin, which
    # masked this until a real cross-origin deployment would have hit it).
    expose_headers=["X-Rendered-Width", "X-Rendered-Height", "X-Native-Width", "X-Native-Height"],
)
# Middleware registration order matters here: Starlette wraps middleware
# "onion"-style, and the LAST one registered via add_middleware becomes
# the OUTERMOST layer (runs first on the way in, last on the way out).
# CORSMiddleware is registered SECOND (after RateLimitMiddleware) so it
# ends up outermost — meaning it still adds CORS headers to a 429
# response that RateLimitMiddleware short-circuits before reaching any
# route. Registering it the other way around would make a 429 look like
# a CORS failure to a cross-origin browser fetch() (no
# Access-Control-Allow-Origin header on the error response), which would
# be strictly worse than not rate-limiting at all from the frontend's
# point of view.

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(ask_router)
app.include_router(ask_stream_router)
app.include_router(documents_router)
# Phase 5 completion pass, item 3.3: ingestion job status/retry
# endpoints (GET /documents/{id}/ingestion, POST /documents/{id}/reingest)
# — see app/api/ingestion_jobs.py's docstring for why retry genuinely
# backgrounds execution while the upload endpoint (in documents_router,
# above) deliberately does not.
app.include_router(ingestion_jobs_router)
# Phase 4.1(5): Source Viewer read endpoints (document lookup, page image
# rendering, original source serving) — see app/api/source.py's
# docstring. A separate router (not added to app/api/documents.py) so
# the existing upload/list-documents module is untouched, per this
# pass's "do not rewrite existing architecture" constraint.
app.include_router(source_router)
# Phase 6: multi-document reasoning (compare/extract/summarize/report —
# spec items 2-6, 14) and conversation memory (spec items 7, 14). Both
# separate routers, same reasoning as source_router above — neither
# touches app/api/ask.py or app/api/documents.py.
app.include_router(reasoning_router)
app.include_router(conversations_router)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not built (app/static/index.html missing).")
    return FileResponse(index_file)