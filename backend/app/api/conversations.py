"""POST /conversations, GET /conversations, GET /conversations/{id},
DELETE /conversations/{id}, POST /conversations/{id}/messages — Phase 6
conversation memory (spec items 7 & 14).

Follows `app/api/reasoning.py`'s exact established pattern: reuse the
shared `DocumentQAService` singleton (`app.api.ask.get_service()`),
translate upstream-provider exceptions to the same HTTP status codes
`/ask` uses, and never invent a parallel error contract.

**Identity resolution** (spec: "tenant-safe... anonymous sessions should
remain clearly separated"): every endpoint resolves identity the same
way — `current_user` from the Bearer token when present, else an
`anonymous_session_id` supplied by the client (echoed back on creation
for a brand-new anonymous session — see
`ConversationTurnResponse.anonymous_session_id`'s docstring). This
mirrors `app.services.conversation_memory.resolve_identity` exactly;
this module never re-derives identity a second, different way.
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
from app.observability.opik_integration import opik_traced_pipeline
from app.rag.llm import LLMProviderError
from app.schemas.conversations import (
    ConversationCreate,
    ConversationListResponse,
    ConversationMessageOut,
    ConversationOut,
    ConversationTurnRequest,
    ConversationTurnResponse,
)
from app.security.auth import allowed_owner_ids, get_current_user_optional
from app.services.conversation_memory import (
    add_message,
    build_turn_context,
    compose_question_with_context,
    create_conversation,
    delete_conversation,
    get_conversation_or_404,
    list_conversations,
    list_messages,
    maybe_refresh_rolling_summary,
)

_log = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])

_UPSTREAM_ERRORS = (LLMProviderError, groq.APIError, genai_errors.APIError, PineconeException)


def _handle_upstream_errors(exc: Exception, endpoint: str):
    """Identical translation to `app/api/reasoning.py`'s helper — kept
    as its own copy rather than a shared import specifically so neither
    module's error handling can be changed for the other's endpoints by
    accident; both are small enough that duplication here is cheaper
    than the coupling would be.
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


@router.post("", response_model=ConversationOut, status_code=201)
def create(
    request: ConversationCreate,
    current_user: User | None = Depends(get_current_user_optional),
) -> ConversationOut:
    settings = get_settings()
    with session_scope(settings) as session:
        convo = create_conversation(
            session,
            current_user=current_user,
            anonymous_session_id=request.anonymous_session_id,
            title=request.title,
            document_ids=request.document_ids,
        )
        session.flush()
        return ConversationOut(**convo.to_dict())


@router.get("", response_model=ConversationListResponse)
def list_all(
    anonymous_session_id: str | None = None,
    current_user: User | None = Depends(get_current_user_optional),
) -> ConversationListResponse:
    """Lists the caller's own conversations only — `list_conversations`
    (see `app.services.conversation_memory`) scopes strictly by
    `owner_id`/`anonymous_session_id`; an anonymous request with no
    `anonymous_session_id` at all (never had one, or lost it) simply
    sees an empty list, never another session's conversations.
    """
    settings = get_settings()
    with session_scope(settings) as session:
        convos = list_conversations(session, current_user, anonymous_session_id)
        return ConversationListResponse(conversations=[ConversationOut(**c.to_dict()) for c in convos])


@router.get("/{conversation_id}", response_model=ConversationOut)
def get(
    conversation_id: int,
    anonymous_session_id: str | None = None,
    current_user: User | None = Depends(get_current_user_optional),
) -> ConversationOut:
    settings = get_settings()
    with session_scope(settings) as session:
        convo = get_conversation_or_404(session, conversation_id, current_user, anonymous_session_id)
        return ConversationOut(**convo.to_dict())


@router.get("/{conversation_id}/messages", response_model=list[ConversationMessageOut])
def get_messages(
    conversation_id: int,
    anonymous_session_id: str | None = None,
    current_user: User | None = Depends(get_current_user_optional),
) -> list[ConversationMessageOut]:
    settings = get_settings()
    with session_scope(settings) as session:
        convo = get_conversation_or_404(session, conversation_id, current_user, anonymous_session_id)
        return [ConversationMessageOut(**m.to_dict()) for m in list_messages(session, convo)]


@router.delete("/{conversation_id}", status_code=204)
def delete(
    conversation_id: int,
    anonymous_session_id: str | None = None,
    current_user: User | None = Depends(get_current_user_optional),
) -> None:
    settings = get_settings()
    with session_scope(settings) as session:
        convo = get_conversation_or_404(session, conversation_id, current_user, anonymous_session_id)
        delete_conversation(session, convo)


@router.post("/{conversation_id}/messages", response_model=ConversationTurnResponse)
def send_message(
    conversation_id: int,
    request: ConversationTurnRequest,
    current_user: User | None = Depends(get_current_user_optional),
) -> ConversationTurnResponse:
    """One conversation turn: append the user's message, run it through
    the real `DocumentQAService.ask()` pipeline WITH conversation context
    prepended (`build_turn_context`/`compose_question_with_context` — see
    `app.services.conversation_memory`'s module docstring for exactly
    what "context" means here and why it's a formatted string prefix
    rather than a real multi-turn chat structure), append the assistant's
    grounded answer (with its real citations — never stored/returned
    ungrounded), and refresh the rolling summary if the conversation has
    grown past the verbatim window.

    Tenant safety: `get_conversation_or_404` (called first, before
    anything else) is the ONLY way this endpoint ever loads a
    conversation — a conversation belonging to a different identity 404s
    immediately, before any retrieval/LLM call happens.
    """
    settings = get_settings()
    with session_scope(settings) as session:
        convo = get_conversation_or_404(
            session,
            conversation_id,
            current_user,
            request.anonymous_session_id,
        )

        context_prefix, source_files = build_turn_context(session, convo)
        composed_question = compose_question_with_context(context_prefix, request.question)

        user_msg = add_message(session, convo, role="user", content=request.question)

        try:
            service = get_service()
            owner_scope = allowed_owner_ids(current_user)
            result = opik_traced_pipeline(
                settings,
                service.ask,
                composed_question,
                allowed_owner_ids=owner_scope,
                source_files=source_files,
            )
        except Exception as exc:  # noqa: BLE001
            _handle_upstream_errors(exc, "/conversations/{id}/messages")

        assistant_msg = add_message(
            session,
            convo,
            role="assistant",
            content=result["answer"],
            citations=result.get("citations") or [],
            retrieved_chunk_ids=[c["chunk_id"] for c in (result.get("citations") or [])],
            query_type=result.get("query_type"),
        )

        maybe_refresh_rolling_summary(session, convo, service.chat, settings.answer_model)
        session.flush()

        return ConversationTurnResponse(
            conversation_id=convo.id,
            user_message=ConversationMessageOut(**user_msg.to_dict()),
            assistant_message=ConversationMessageOut(**assistant_msg.to_dict()),
            # An existing conversation being continued already has a
            # resolved identity (owner_id XOR anonymous_session_id, set
            # at creation — see create_conversation); the caller's own
            # anonymous_session_id (already validated above by
            # get_conversation_or_404) is simply echoed back, never
            # regenerated. None for an authenticated caller, which never
            # uses this identity path.
            anonymous_session_id=request.anonymous_session_id if current_user is None else None,
        )
