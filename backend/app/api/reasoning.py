"""POST /compare, /extract, /summarize, /report — Phase 6 multi-document
reasoning features (spec items 2, 3, 4, 5, 6, 14).

Every endpoint here follows the exact same shape as `app/api/ask.py`'s
`ask()`: resolve the caller's tenant scope, reuse the shared
`DocumentQAService` singleton (`app.api.ask.get_service()` — not a
second client-construction path) for `index`/`embeddings`/`chat`
/`settings`, resolve+tenant-check the requested `document_ids`
(`app.rag.multi_doc.resolve_documents`), call the corresponding
`app.rag.*` module function, and translate the same upstream-provider
exception types to the same HTTP status codes `/ask` already uses — a
Phase 6 feature hitting a Gemini/Groq/Pinecone outage should fail the
same way `/ask` does, not invent a different error contract.
"""

from __future__ import annotations

import logging

import groq
from fastapi import APIRouter, Depends, HTTPException
from google.genai import errors as genai_errors
from pinecone.exceptions import PineconeException
from pydantic import ValidationError

from app.api.ask import get_service
from app.core.config import get_settings
from app.db.models import User
from app.db.session import session_scope
from app.rag.comparison import ComparisonError, compare_documents, compare_multiple
from app.rag.extraction import SchemaValidationError, extract_structured
from app.rag.llm import LLMProviderError
from app.rag.multi_doc import resolve_documents
from app.rag.report import generate_report
from app.rag.summarization import SummarizationError, SummaryMode, summarize_documents
from app.schemas.reasoning import (
    CompareMultiDocResponse,
    CompareRequest,
    CompareTwoDocResponse,
    ExtractRequest,
    ExtractResponse,
    ReportRequest,
    ReportResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.security.auth import allowed_owner_ids, get_current_user_optional

_log = logging.getLogger(__name__)
router = APIRouter(tags=["reasoning"])

_UPSTREAM_ERRORS = (LLMProviderError, groq.APIError, genai_errors.APIError, PineconeException)


def _handle_upstream_errors(exc: Exception, endpoint: str):
    """Shared translation of provider/config exceptions into the same
    HTTP status codes `/ask` uses — see module docstring. Called from
    each endpoint's `except` block; raises the appropriate
    `HTTPException` (never returns).
    """
    if isinstance(exc, ValidationError):
        raise HTTPException(status_code=500, detail=f"Configuration error: {exc}") from exc
    if isinstance(exc, _UPSTREAM_ERRORS):
        _log.warning("upstream provider error on %s: %s: %s", endpoint, type(exc).__name__, exc)
        raise HTTPException(status_code=503, detail=f"Upstream service unavailable ({type(exc).__name__}).") from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _log.exception("unexpected error on %s", endpoint)
    raise HTTPException(status_code=500, detail="Internal application error.") from exc


@router.post("/compare", response_model=CompareTwoDocResponse | CompareMultiDocResponse)
def compare(
    request: CompareRequest,
    current_user: User | None = Depends(get_current_user_optional),
):
    settings = get_settings()
    owner_scope = allowed_owner_ids(current_user)
    with session_scope(settings) as session:
        resolved = resolve_documents(session, request.document_ids, current_user)
        try:
            service = get_service()
            if len(resolved) == 2:
                return compare_documents(
                    service.index, service.embeddings, service.chat, service.settings, request.topic, resolved, owner_scope
                )
            return compare_multiple(
                service.index, service.embeddings, service.chat, service.settings, request.topic, resolved, owner_scope
            )
        except ComparisonError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            _handle_upstream_errors(exc, "/compare")


@router.post("/extract", response_model=ExtractResponse)
def extract(
    request: ExtractRequest,
    current_user: User | None = Depends(get_current_user_optional),
) -> ExtractResponse:
    settings = get_settings()
    owner_scope = allowed_owner_ids(current_user)
    with session_scope(settings) as session:
        resolved = resolve_documents(session, request.document_ids, current_user)
        try:
            service = get_service()
            result = extract_structured(
                service.index, service.embeddings, service.chat, service.settings, request.schema_, resolved, owner_scope
            )
            return ExtractResponse(**result)
        except SchemaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            _handle_upstream_errors(exc, "/extract")


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(
    request: SummarizeRequest,
    current_user: User | None = Depends(get_current_user_optional),
) -> SummarizeResponse:
    settings = get_settings()
    owner_scope = allowed_owner_ids(current_user)
    with session_scope(settings) as session:
        resolved = resolve_documents(session, request.document_ids, current_user)
        try:
            mode = SummaryMode(request.mode)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mode {request.mode!r}. Supported: {[m.value for m in SummaryMode]}.",
            ) from exc
        try:
            service = get_service()
            result = summarize_documents(
                service.index, service.embeddings, service.chat, service.settings, resolved, mode, owner_scope
            )
            return SummarizeResponse(**result)
        except SummarizationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            _handle_upstream_errors(exc, "/summarize")


@router.post("/report", response_model=ReportResponse)
def report(
    request: ReportRequest,
    current_user: User | None = Depends(get_current_user_optional),
) -> ReportResponse:
    settings = get_settings()
    owner_scope = allowed_owner_ids(current_user)
    with session_scope(settings) as session:
        resolved = resolve_documents(session, request.document_ids, current_user)
        try:
            service = get_service()
            result = generate_report(
                service.index, service.embeddings, service.chat, service.settings, request.title, resolved, owner_scope
            )
            return ReportResponse(**result)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            _handle_upstream_errors(exc, "/report")