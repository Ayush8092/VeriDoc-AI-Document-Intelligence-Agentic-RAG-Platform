"""POST /ask — document-grounded Q&A over the ingested corpus."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import groq
from fastapi import APIRouter, Depends, HTTPException, Query
from google.genai import errors as genai_errors
from langgraph.errors import GraphRecursionError
from pinecone.exceptions import PineconeException
from pydantic import ValidationError

from app.core.config import get_settings
from app.db.models import User
from app.observability.opik_integration import opik_traced_pipeline
from app.observability.trace import start_trace
from app.rag.graph import DocumentQAService
from app.rag.llm import LLMProviderError, REFUSAL_TEXT
from app.schemas.documents import AskRequest, AskResponse
from app.security.auth import allowed_owner_ids, get_current_user_optional

_log = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

_UPSTREAM_ERRORS = (LLMProviderError, groq.APIError, genai_errors.APIError, PineconeException)

_service: DocumentQAService | None = None
_init_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qaservice-init")


def get_service() -> DocumentQAService:
    """Lazy singleton fallback, bounded by SERVICE_INIT_TIMEOUT_SECONDS."""
    global _service
    if _service is None:
        settings = get_settings()
        future = _init_executor.submit(DocumentQAService)
        try:
            _service = future.result(timeout=settings.service_init_timeout_seconds)
        except FutureTimeoutError as exc:
            raise RuntimeError(
                f"Timed out after {settings.service_init_timeout_seconds}s initializing "
                "Gemini/Groq/Pinecone clients. Check network connectivity and API keys."
            ) from exc
    return _service


def set_service(service: DocumentQAService | None) -> None:
    """Test hook: inject/reset the module-level singleton."""
    global _service
    _service = service


@router.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(
    request: AskRequest,
    trace: bool = Query(default=False, description="Include the LangGraph execution trace in the response."),
    current_user: User | None = Depends(get_current_user_optional),
) -> AskResponse:
    settings = get_settings()
    req_trace = start_trace()
    owner_scope = allowed_owner_ids(current_user)

    try:
        service = get_service()
        with req_trace.span("ask", question_len=len(request.question)):
            result = opik_traced_pipeline(settings, service.ask, request.question, allowed_owner_ids=owner_scope)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Configuration error: {exc}") from exc
    except GraphRecursionError as exc:
        _log.warning("graph recursion limit hit: %s", exc)
        return AskResponse(answer=REFUSAL_TEXT, found=False, citations=[], trace_id=req_trace.trace_id)
    except _UPSTREAM_ERRORS as exc:
        _log.warning("upstream provider error on /ask: %s: %s", type(exc).__name__, exc)
        raise HTTPException(status_code=503, detail=f"Upstream service unavailable ({type(exc).__name__}).") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _log.exception("unexpected error on /ask")
        raise HTTPException(status_code=500, detail="Internal application error.") from exc
    finally:
        req_trace.finish()

    include_trace = trace or settings.include_trace
    return AskResponse(
        answer=result["answer"],
        found=result["found"],
        citations=result["citations"],
        query_type=result.get("query_type"),
        claims=result.get("claims") or [],
        grounded_claim_rate=result.get("grounded_claim_rate"),
        citation_precision=result.get("citation_precision"),
        citation_recall=result.get("citation_recall"),
        trace=result["trace"] if include_trace else None,
        trace_id=req_trace.trace_id,
        usage=req_trace.to_dict() if include_trace else None,
        stage_latencies=result.get("stage_latencies") if include_trace else None,
    )