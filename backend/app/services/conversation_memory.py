"""Conversation memory (Phase 6, spec item 7).

Built on `app.db.models.Conversation`/`ConversationMessage` (already
defined — this module is the service layer those models' docstrings
reference but that didn't exist yet).

**Tenant safety** (spec: "one user's memory can never be accessed by
another user" / "anonymous sessions should remain clearly separated"):
every conversation belongs to EITHER an authenticated `owner_id` OR an
`anonymous_session_id` — never both, never neither (enforced by the
model's own `CheckConstraint`, not just this module's convention).
`get_conversation_or_404` is the single choke point every API endpoint
uses to load a conversation, and it 404s (never 403 — same reasoning as
`app.api.source._get_document_or_404`) for a conversation that doesn't
belong to the caller's identity.

**Bounded, not "dump unlimited chat history into the prompt"** (spec's
explicit instruction): only the last `RECENT_MESSAGE_WINDOW` messages
are ever sent to the LLM verbatim. Anything older is folded into
`Conversation.rolling_summary` — a single bounded-size running summary,
regenerated (not endlessly appended-to) each time the window is
exceeded, via `maybe_refresh_rolling_summary`. The FULL message history
is still kept in the `conversation_messages` table (spec: "auditable")
— summarization only affects what's SENT to the model on the next
turn, never what's stored.

**Retrieval-aware**: `build_turn_context` also resolves
`Conversation.document_ids` into the `source_files` set
`app.rag.graph.DocumentQAService.ask()`'s (Phase 6-added, see that
module) `source_files` parameter expects, so a conversation "focused"
on specific documents scopes retrieval to them on every turn without
the caller re-specifying it each time (spec: "document-aware").
"""

from __future__ import annotations

import json
import logging
import secrets

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Conversation, ConversationMessage, Document, User
from app.rag.llm import chat_json

_log = logging.getLogger(__name__)

#: Verbatim messages sent to the LLM per turn — see module docstring.
#: 6 (3 user/assistant pairs) is enough for genuine short-term
#: follow-up context ("what about the second one?") without the prompt
#: growing with conversation length.
RECENT_MESSAGE_WINDOW = 6

#: Conversations a single owner/session can have open at once — bounded
#: for the same "keep the workflow bounded" reason as
#: `app.rag.multi_doc.MAX_DOCUMENTS`; prevents unbounded storage growth
#: from one identity.
MAX_CONVERSATIONS_PER_OWNER = 200


class ConversationAccessError(HTTPException):
    def __init__(self, conversation_id: int):
        super().__init__(status_code=404, detail=f"No conversation with id={conversation_id}")


def new_anonymous_session_id() -> str:
    """A client-generated-STYLE opaque token — generated server-side on
    first use so a caller never has to invent their own (and can't
    accidentally collide with another session's). 32 bytes of
    `secrets.token_urlsafe` — cryptographically unguessable, matching
    the model docstring's "never a guessable sequential id" requirement.
    """
    return secrets.token_urlsafe(32)


def resolve_identity(current_user: User | None, anonymous_session_id: str | None) -> tuple[int | None, str | None]:
    """Returns `(owner_id, anonymous_session_id)` — exactly one non-None,
    matching `Conversation`'s CheckConstraint. An authenticated user
    always wins (an authenticated request that also happens to carry a
    leftover anonymous session id from before login is NOT merged into
    that anonymous session — that would let a logged-out browser's
    conversations silently become a specific logged-in user's without
    an explicit action, which is exactly the kind of identity confusion
    the model's CheckConstraint exists to prevent).
    """
    if current_user is not None:
        return current_user.id, None
    if anonymous_session_id:
        return None, anonymous_session_id
    return None, new_anonymous_session_id()


def get_conversation_or_404(
    session: Session, conversation_id: int, current_user: User | None, anonymous_session_id: str | None
) -> Conversation:
    convo = session.get(Conversation, conversation_id)
    if convo is None:
        raise ConversationAccessError(conversation_id)
    if current_user is not None:
        if convo.owner_id != current_user.id:
            raise ConversationAccessError(conversation_id)
    else:
        if not anonymous_session_id or convo.anonymous_session_id != anonymous_session_id:
            raise ConversationAccessError(conversation_id)
    return convo


def list_conversations(session: Session, current_user: User | None, anonymous_session_id: str | None) -> list[Conversation]:
    if current_user is not None:
        stmt = select(Conversation).where(Conversation.owner_id == current_user.id)
    elif anonymous_session_id:
        stmt = select(Conversation).where(Conversation.anonymous_session_id == anonymous_session_id)
    else:
        return []
    stmt = stmt.order_by(Conversation.updated_at.desc())
    return list(session.execute(stmt).scalars().all())


def create_conversation(
    session: Session,
    *,
    current_user: User | None,
    anonymous_session_id: str | None,
    title: str,
    document_ids: list[int],
) -> Conversation:
    owner_id, resolved_anon_id = resolve_identity(current_user, anonymous_session_id)

    existing_count = len(list_conversations(session, current_user, resolved_anon_id))
    if existing_count >= MAX_CONVERSATIONS_PER_OWNER:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum of {MAX_CONVERSATIONS_PER_OWNER} conversations reached. Delete an old one first.",
        )

    # document_ids are validated for tenant-visibility the same way
    # every other multi-document feature does (app.rag.multi_doc.
    # resolve_documents) — but only if non-empty; an unfocused
    # conversation (document_ids=[]) is valid and searches the whole
    # corpus, same as a plain /ask call.
    if document_ids:
        from app.rag.multi_doc import resolve_documents

        resolve_documents(session, document_ids, current_user)

    convo = Conversation(
        owner_id=owner_id,
        anonymous_session_id=resolved_anon_id,
        title=title[:256],
        document_ids=json.dumps(document_ids),
    )
    session.add(convo)
    session.flush()
    return convo


def delete_conversation(session: Session, convo: Conversation) -> None:
    session.query(ConversationMessage).filter(ConversationMessage.conversation_id == convo.id).delete(
        synchronize_session=False
    )
    session.delete(convo)


def list_messages(session: Session, convo: Conversation) -> list[ConversationMessage]:
    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == convo.id)
        .order_by(ConversationMessage.created_at.asc(), ConversationMessage.id.asc())
    )
    return list(session.execute(stmt).scalars().all())


def add_message(
    session: Session,
    convo: Conversation,
    *,
    role: str,
    content: str,
    citations: list[dict] | None = None,
    retrieved_chunk_ids: list[str] | None = None,
    query_type: str | None = None,
) -> ConversationMessage:
    msg = ConversationMessage(
        conversation_id=convo.id,
        role=role,
        content=content,
        citations=json.dumps(citations or []),
        retrieved_chunk_ids=json.dumps(retrieved_chunk_ids or []),
        query_type=query_type,
    )
    session.add(msg)
    session.flush()
    return msg


_SUMMARY_SYSTEM = (
    "You maintain a rolling summary of an ongoing conversation between a user and a document-QA "
    "assistant. You are given the existing summary (may be empty) and the older messages that need "
    "to be folded into it. Produce an updated summary that preserves what topics/documents were "
    "discussed and any conclusions reached, concisely — a few sentences, not a transcript. Do not "
    "invent anything not present in the summary or messages.\n\n"
    'Reply with a single JSON object: {"summary": "<updated summary>"}'
)


def maybe_refresh_rolling_summary(session: Session, convo: Conversation, chat, model: str) -> None:
    """If the conversation has grown past `RECENT_MESSAGE_WINDOW`, fold
    everything older than the window into `rolling_summary` via ONE
    bounded LLM call. A no-op (no LLM call) when the conversation still
    fits entirely within the window — most conversations, most of the
    time. Never raises: `chat_json` already degrades to `{}` on any
    provider failure (see `app.rag.llm._chat_json`'s docstring), and a
    failed summary refresh just means the OLD rolling_summary is kept
    for this turn — never a hard failure of the conversation turn
    itself, since the summary is a memory optimization, not
    correctness-critical.
    """
    messages = list_messages(session, convo)
    if len(messages) <= RECENT_MESSAGE_WINDOW:
        return
    older = messages[: len(messages) - RECENT_MESSAGE_WINDOW]
    if not older:
        return

    transcript = "\n".join(f"{m.role}: {m.content}" for m in older)
    user_text = f"Existing summary:\n{convo.rolling_summary or '(none yet)'}\n\nOlder messages to fold in:\n{transcript}"
    try:
        data = chat_json(chat, model, _SUMMARY_SYSTEM, user_text, purpose="conversation_summary")
        new_summary = str(data.get("summary", "") or "").strip()
        if new_summary:
            convo.rolling_summary = new_summary[:4000]  # bounded — see module docstring
            session.flush()
    except Exception:  # noqa: BLE001
        _log.warning("conversation_memory: rolling summary refresh failed for conversation id=%s", convo.id)


def build_turn_context(session: Session, convo: Conversation) -> tuple[str, frozenset[str] | None]:
    """Returns `(context_prefix, source_files)` for the NEXT question in
    this conversation:

    - `context_prefix`: rolling_summary (if any) + the last
      `RECENT_MESSAGE_WINDOW` messages verbatim, formatted as plain text
      to prepend before the caller's actual new question — see this
      module's docstring for why a formatted-string prefix rather than
      a real multi-turn chat structure (keeps `DocumentQAService.ask()`'s
      existing single-turn pipeline completely unchanged).
    - `source_files`: the conversation's focused document filenames
      (resolved from `Conversation.document_ids`), or `None` if the
      conversation isn't focused on specific documents (whole-corpus
      search, same as a plain `/ask` call).
    """
    parts = []
    if convo.rolling_summary:
        parts.append(f"Earlier in this conversation: {convo.rolling_summary}")

    recent = list_messages(session, convo)[-RECENT_MESSAGE_WINDOW:]
    if recent:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in recent)
        parts.append(f"Recent conversation:\n{transcript}")

    context_prefix = "\n\n".join(parts)

    document_ids = json.loads(convo.document_ids or "[]")
    source_files: frozenset[str] | None = None
    if document_ids:
        docs = session.execute(select(Document).where(Document.id.in_(document_ids))).scalars().all()
        filenames = {d.filename for d in docs}
        source_files = frozenset(filenames) if filenames else None

    return context_prefix, source_files


def compose_question_with_context(context_prefix: str, question: str) -> str:
    """Combines `build_turn_context`'s prefix with the new question into
    the single string `DocumentQAService.ask()` takes — kept as its own
    tiny function (rather than inlined at every call site) so there is
    exactly ONE place that defines what "a question with conversation
    context" looks like.
    """
    if not context_prefix:
        return question
    return f"{context_prefix}\n\nCurrent question: {question}"