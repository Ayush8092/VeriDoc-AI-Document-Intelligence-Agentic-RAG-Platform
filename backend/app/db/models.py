"""Document/ingestion metadata persistence.

Vectors and chunk text live in Pinecone (see app/vectorstore.py); this
module is the system of record for *document-level* metadata: what files
exist, what version they're on, and what each ingestion run did. This is
what lets the API answer "list my documents" / "open the source behind
this citation" without a Pinecone scan, and what makes document
versioning (section 57 of the product spec) possible.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, Boolean, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    """A registered account (Phase 5 — see app/security/auth.py for
    password hashing/token issuance). `hashed_password` is a bcrypt hash,
    never the plaintext password — see `app.security.auth.hash_password`.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def to_dict(self) -> dict:
        # Deliberately excludes `hashed_password` — this is what every
        # endpoint that returns user info (e.g. GET /auth/me) serializes,
        # so a hash can never leak through a response by accident.
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Document(Base):
    """A logical document (identified by source_root + filename).

    A `Document` row is stable across re-uploads of the same filename;
    each content change creates a new `DocumentVersion` rather than
    mutating history away, so "which version was this citation extracted
    from" always has an answer.

    `owner_id` (Phase 5): `None` means this document belongs to the
    shared/public corpus (everything ingested from `data/corpus/`, and
    every upload made without authentication — see
    app.core.config.Settings' authentication docstring). A non-null
    `owner_id` scopes the document to exactly that user; every retrieval
    /listing/source-viewer path must treat `owner_id NOT IN
    (None, current_user.id)` as invisible — see
    app/retrieval.py, app/lexical_index.py, app/api/documents.py,
    app/api/source.py.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_root: Mapped[str] = mapped_column(String(32))   # "corpus" | "uploads"
    filename: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(512), default="")
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    current_version_id: Mapped[int | None] = mapped_column(
        # Phase 7 completion pass: `use_alter=True` breaks the genuine FK
        # cycle between `documents` and `document_versions`
        # (documents.current_version_id -> document_versions.id,
        # document_versions.document_id -> documents.id) at the
        # SQLAlchemy metadata level -- without it, `alembic check`/
        # `alembic revision --autogenerate` emits "Cannot correctly sort
        # tables; there are unresolvable cycles" (SAWarning, and per
        # SQLAlchemy's own message, "may raise an error in a future
        # release"). `use_alter=True` tells SQLAlchemy this specific FK
        # is safe to create via a deferred ALTER TABLE (after both
        # tables already exist) rather than inline in documents'
        # CREATE TABLE -- which is exactly what the existing baseline
        # migration (alembic/versions/89dd0fc94357_baseline_schema.py)
        # already does by hand (`batch_op.create_foreign_key(
        # 'fk_documents_current_version_id', ...)`, added only after
        # both tables exist). This change makes the live SQLAlchemy
        # models agree with what the migration already does -- no new
        # migration needed (verified: `alembic check` reports "No new
        # upgrade operations detected" both before and after this
        # change; only the SAWarning itself is what's newly gone).
        # `name=` must match the migration's constraint name exactly, or
        # `alembic check`/downgrade would silently target two
        # differently-named constraints for the same relationship.
        ForeignKey("document_versions.id", use_alter=True, name="fk_documents_current_version_id"),
        nullable=True,
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        "DocumentVersion", back_populates="document", foreign_keys="DocumentVersion.document_id"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_root": self.source_root,
            "filename": self.filename,
            "file_type": self.file_type,
            "title": self.title,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "current_version_id": self.current_version_id,
        }


class DocumentVersion(Base):
    """One ingested content version of a `Document`.

    `file_hash` is the hash of the whole source file's bytes (distinct
    from each chunk's own `content_hash` in Pinecone metadata) — it is
    what ingestion checks first to decide "unchanged, skip re-parsing"
    vs. "changed, create a new version".
    """

    __tablename__ = "document_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    version_number: Mapped[int] = mapped_column(Integer)
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    has_scanned_pages: Mapped[bool] = mapped_column(Boolean, default=False)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    mean_ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    warnings: Mapped[str] = mapped_column(Text, default="")  # newline-joined
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped["Document"] = relationship("Document", back_populates="versions", foreign_keys=[document_id])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "version_number": self.version_number,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "has_scanned_pages": self.has_scanned_pages,
            "table_count": self.table_count,
            "chunk_count": self.chunk_count,
            "mean_ocr_confidence": self.mean_ocr_confidence,
            "warnings": [w for w in self.warnings.split("\n") if w],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class IngestionJob(Base):
    """Per-document ingestion job (Phase 5 completion pass, items 4/5).

    Deliberately a NEW table rather than an extension of `IngestionRun`:
    `IngestionRun` tracks whole-corpus BATCH runs (one row per
    `ingest_corpus()` invocation, covering every file in `corpus_dir` +
    `upload_dir` together) — it has no per-document identity and is
    unchanged by this pass. `IngestionJob` tracks a single document's
    ingestion lifecycle (state machine: QUEUED -> PROCESSING ->
    COMPLETED | FAILED | RETRYING | CANCELLED), which is a different
    shape of fact. Extending `IngestionRun` with a nullable
    `document_id` was considered but would conflate two different
    granularities under one model; a dedicated table matching the
    requested state machine directly was judged clearer. See
    `app/services/ingestion_jobs.py`.

    `document_id`/`version_id` are nullable: a job is created the moment
    an upload is accepted (before the `Document`/`DocumentVersion` rows
    necessarily exist yet, and — for a batch whose ingestion fails
    entirely — they may never exist at all, per the atomic-upload
    rollback guarantee in `app/api/documents.py`). `owner_id` is
    denormalized here (copied at job-creation time) so a job's ownership
    can be checked without a join, matching every other tenant-isolation
    check in this codebase (`app.security.auth.allowed_owner_ids`).
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("document_versions.id"), nullable=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(512), default="")
    source_root: Mapped[str] = mapped_column(String(32), default="uploads")

    # QUEUED | PROCESSING | COMPLETED | FAILED | RETRYING | CANCELLED
    # (see app.services.ingestion_jobs.JobStatus)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[str] = mapped_column(Text, default="")

    # Phase 5 completion pass (round 3), item 1: durable replacement-file
    # backup, so a REPLACEMENT upload's "restore the old bytes if this
    # job terminally fails" guarantee (see app/api/documents.py's module
    # docstring, "duplicate-filename replacement bug") survives a
    # process boundary — the standalone worker
    # (app.services.ingestion_jobs.worker) that actually processes this
    # job may be a completely different OS process (even a different
    # machine) than the one that accepted the upload, so this can no
    # longer be an in-memory Python object captured in a BackgroundTasks
    # closure (the pre-round-3 design). `backup_key` is the storage key
    # (see app.storage.base.StorageBackend) holding the PRE-UPLOAD bytes
    # of `restore_to_key`, or NULL if this job is for a brand-new
    # filename (nothing to restore). Both NULL for the common case.
    # See app.services.ingestion_jobs.run_job's terminal-status handling.
    backup_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    restore_to_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Progress / observability (item 7) — real measurements only, filled
    # in once a run actually reports them; 0/None until then.
    chunks_total: Mapped[int] = mapped_column(Integer, default=0)
    chunks_changed: Mapped[int] = mapped_column(Integer, default=0)
    vectors_upserted: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "owner_id": self.owner_id,
            "filename": self.filename,
            "source_root": self.source_root,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "backup_key": self.backup_key,
            "restore_to_key": self.restore_to_key,
            "chunks_total": self.chunks_total,
            "chunks_changed": self.chunks_changed,
            "vectors_upserted": self.vectors_upserted,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class IngestionRun(Base):
    """Audit record of one ingestion pipeline execution."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reset: Mapped[bool] = mapped_column(Boolean, default=False)
    chunks_total: Mapped[int] = mapped_column(Integer, default=0)
    chunks_changed: Mapped[int] = mapped_column(Integer, default=0)
    vectors_upserted: Mapped[int] = mapped_column(Integer, default=0)
    stale_vectors_deleted: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="success")  # success | failed
    detail: Mapped[str] = mapped_column(Text, default="")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "reset": self.reset,
            "chunks_total": self.chunks_total,
            "chunks_changed": self.chunks_changed,
            "vectors_upserted": self.vectors_upserted,
            "stale_vectors_deleted": self.stale_vectors_deleted,
            "status": self.status,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Conversation memory (Phase 6, item 7)
# ---------------------------------------------------------------------------


class Conversation(Base):
    """A conversation thread — user-scoped (or anonymous-session-scoped),
    document-aware, and bounded (see `ConversationMessage` and
    `app.services.conversation_memory` for how "bounded" is enforced —
    this table only stores what already happened, it doesn't cap
    anything on its own).

    Exactly one of `owner_id`/`anonymous_session_id` is set, never both
    and never neither — enforced by the CheckConstraint below, not just
    convention, so a conversation can never end up ownerless-but-not-
    anonymous (which would make `allowed_owner_ids`-style tenant checks
    silently pass it through to everyone — see
    `app/security/auth.py`'s docstring on why that specific failure mode
    is the one this project is most careful never to reintroduce).
    `anonymous_session_id` is a client-generated opaque token (never a
    guessable sequential id), scoped ONLY to conversations — it is NOT
    the same namespace as an authenticated `User.id` and can never be
    confused for one (see `app.services.conversation_memory.
    resolve_identity`).
    """

    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "(owner_id IS NULL) != (anonymous_session_id IS NULL)",
            name="ck_conversations_exactly_one_owner",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    anonymous_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    # JSON list of Document.id the conversation is currently "focused" on
    # (see app/schemas/conversations.py's ConversationCreate.document_ids)
    # — used to scope retrieval for follow-up turns without the caller
    # having to re-specify it on every message.
    document_ids: Mapped[str] = mapped_column(Text, default="[]")
    # A bounded, LLM-maintained running summary of every message OLDER
    # than the most recent N (app.services.conversation_memory's
    # RECENT_MESSAGE_WINDOW) — see that module's docstring
    # for exactly how/when this is regenerated. Empty string, never
    # None, so "no summary yet" and "summary is empty" aren't two
    # different representations of the same thing.
    rolling_summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "anonymous_session_id": self.anonymous_session_id,
            "title": self.title,
            "document_ids": json.loads(self.document_ids or "[]"),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ConversationMessage(Base):
    """One turn of a conversation. `retrieved_chunk_ids` (auditable
    memory — spec item 7) records exactly which evidence chunk_ids were
    actually retrieved/cited for THIS turn, independent of
    `citations` (the validated, user-facing citation objects — see
    `app.rag.llm.validate_citations`) — auditability needs the raw
    retrieval trace, not just what made it into the final answer.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[str] = mapped_column(Text, default="[]")  # JSON — validated Citation objects, assistant turns only
    retrieved_chunk_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON list[str]
    query_type: Mapped[str | None] = mapped_column(String(32), nullable=True)  # app.rag.query_classifier.QueryType value
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "citations": json.loads(self.citations or "[]"),
            "retrieved_chunk_ids": json.loads(self.retrieved_chunk_ids or "[]"),
            "query_type": self.query_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }