"""Request/response models for conversation memory (Phase 6, spec items
7 & 14). Mirrors `app.db.models.Conversation`/`ConversationMessage`'s
own `to_dict()` shape — see `app.services.conversation_memory`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.documents import Citation


class ConversationCreate(BaseModel):
    title: str = Field(default="", max_length=256)
    document_ids: list[int] = Field(
        default_factory=list,
        description="Documents this conversation is focused on. Empty = whole-corpus search, same as a plain /ask call.",
    )
    # Present a client's existing anonymous session token to continue an
    # anonymous session (rather than starting a new, empty one) —
    # ignored for an authenticated request (see
    # app.services.conversation_memory.resolve_identity's docstring on
    # why an authenticated identity always wins).
    anonymous_session_id: str | None = None


class ConversationOut(BaseModel):
    id: int
    owner_id: int | None
    anonymous_session_id: str | None
    title: str
    document_ids: list[int]
    created_at: str | None
    updated_at: str | None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]


class ConversationMessageOut(BaseModel):
    id: int
    conversation_id: int
    role: str  # "user" | "assistant"
    content: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    query_type: str | None
    created_at: str | None


class ConversationTurnRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Only needed for an ANONYMOUS caller continuing a session across
    # requests (an authenticated caller's identity comes from the
    # Bearer token, same as every other endpoint) — see
    # `app.services.conversation_memory.get_conversation_or_404`.
    anonymous_session_id: str | None = None


class ConversationTurnResponse(BaseModel):
    conversation_id: int
    user_message: ConversationMessageOut
    assistant_message: ConversationMessageOut
    # Echoed back so an anonymous caller that didn't already have one
    # learns the session id assigned on first use (see
    # `ConversationCreate.anonymous_session_id`'s docstring) — None for
    # an authenticated caller, which never uses this identity path.
    anonymous_session_id: str | None = None