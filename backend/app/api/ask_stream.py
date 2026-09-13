"""POST /ask/stream — Server-Sent Events streaming of a grounded answer.

**What actually streams, and why.** Citation validation
(`validate_citations` in `app/rag/llm.py`) has to run against the FULL
set of retrieved chunks and the model's FULL structured JSON output
(`{"found": ..., "answer": ..., "evidence_refs": [...]}`) — there's no
safe way to validate a citation against an answer that isn't finished
yet. So this endpoint runs the full retrieve -> grade -> generate ->
validate -> ground pipeline exactly like `POST /ask` does, synchronously,
and only THEN streams the resulting, already-citation-validated answer
text back to the client word-by-word over SSE.

That means TTFT here measures "time until the client starts receiving
the answer" (a real, meaningful UX metric — the user sees text appear
sooner than waiting for one giant response body), not "time to the
model's first generated token" (which would require streaming the
grading/generation JSON call itself and validating citations against a
partial response, substantially riskier and out of scope for this
pass — noted honestly rather than implemented as something it isn't).

Event format (one JSON object per `data:` line):
    {"type": "meta", "trace_id": "...", "query_type": "...", "found": true}
    {"type": "token", "text": "The "}
    {"type": "token", "text": "trial "}
    ...
    {"type": "done", "citations": [...], "usage": {...}, "stage_latencies": [...] | null}

A citations-bearing UI should render nothing until it has the "done"
event's `citations` array — the streamed tokens are for perceived-latency
UX only, not a substitute for the validated citation list.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api import ask as ask_module
from app.core.config import get_settings
from app.db.models import User
from app.observability.trace import start_trace
from app.rag.llm import REFUSAL_TEXT
from app.schemas.documents import AskRequest
from app.security.auth import allowed_owner_ids, get_current_user_optional

_log = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _word_chunks(text: str) -> list[str]:
    """Split into "word + trailing whitespace" chunks so re-joining every
    chunk reproduces the original string exactly (no lost/added spaces).
    """
    chunks: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch.isspace():
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


async def _event_stream(question: str, owner_scope: frozenset[str]):
    settings = get_settings()
    trace = start_trace()

    try:
        service = ask_module.get_service()
        with trace.span("ask_stream"):
            # The pipeline itself is synchronous (LangGraph .invoke); run it
            # off the event loop so this endpoint doesn't block other
            # concurrent requests while retrieval/generation is in flight.
            result = await asyncio.to_thread(service.ask, question, allowed_owner_ids=owner_scope)
    except ask_module._UPSTREAM_ERRORS as exc:
        _log.warning("upstream provider error on /ask/stream: %s: %s", type(exc).__name__, exc)
        yield _sse({"type": "error", "detail": f"Upstream service unavailable ({type(exc).__name__})."})
        return
    except RuntimeError as exc:
        yield _sse({"type": "error", "detail": str(exc)})
        return
    except Exception:  # noqa: BLE001
        _log.exception("unexpected error on /ask/stream")
        yield _sse({"type": "error", "detail": "Internal application error."})
        return

    yield _sse(
        {
            "type": "meta",
            "trace_id": trace.trace_id,
            "query_type": result.get("query_type"),
            "found": result["found"],
        }
    )

    answer = result["answer"] or REFUSAL_TEXT
    first = True
    for chunk in _word_chunks(answer):
        if first:
            trace.mark_first_token()
            first = False
        yield _sse({"type": "token", "text": chunk})
        # A deliberate, small delay — purely so a real client actually
        # perceives incremental delivery instead of the whole answer
        # arriving in one scheduler tick (the SSE framing is real; the
        # per-chunk pacing is a UX simulation layered on top of an
        # already-fully-computed answer, per this module's docstring).
        await asyncio.sleep(0.015)

    trace.finish()
    yield _sse(
        {
            "type": "done",
            "citations": result["citations"],
            "claims": result.get("claims") or [],
            "grounded_claim_rate": result.get("grounded_claim_rate"),
            "citation_precision": result.get("citation_precision"),
            "usage": trace.to_dict() if (settings.include_trace) else {"trace_id": trace.trace_id},
            # Phase 7 trace viewer (item 18) — see AskResponse.stage_latencies'
            # docstring (app/schemas/documents.py) for why this exists.
            "stage_latencies": result.get("stage_latencies") if settings.include_trace else None,
        }
    )


@router.post("/ask/stream")
async def ask_stream(
    request: AskRequest, current_user: User | None = Depends(get_current_user_optional)
) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request.question, allowed_owner_ids(current_user)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )