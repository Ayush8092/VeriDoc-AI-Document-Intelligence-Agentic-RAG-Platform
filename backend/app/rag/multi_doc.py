"""Multi-document resolution — shared by every Phase 6 feature that
operates over a caller-selected SET of documents rather than the whole
corpus: comparison (`app/rag/comparison.py`), structured extraction
(`app/rag/extraction.py`), summarization (`app/rag/summarization.py`),
and report generation (`app/rag/report.py`).

`resolve_documents` is the ONE place that turns a list of `document_id`s
into (a) the `Document` rows themselves (for display — title, filename,
version) and (b) the `source_files` set `app.retrieval.hybrid_retrieve_*`
needs to scope retrieval. It reuses the exact same tenant-isolation rule
`app/api/source.py::_get_document_or_404` already enforces for every
other document-resolving endpoint (a document belonging to another user
is treated as not found, never a 403 — see that function's docstring for
why: a 403 would itself leak which ids are in use) — this is the same
security property, not a re-derivation of it, applied to a LIST instead
of a single id.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import Document, User
from app.security.prompt_injection import scan_chunks, wrap_untrusted

#: Bounded fan-out — see `resolve_documents`'s docstring. 8 documents is
#: comfortably enough for the spec's own worked examples ("compare these
#: three policy documents") while keeping worst-case retrieval/LLM cost
#: predictable (each document gets its own scoped retrieval pass in
#: comparison/report generation — see those modules).
MAX_DOCUMENTS = 8

#: Evidence chunks retrieved PER DOCUMENT for a multi-document feature —
#: deliberately smaller than `Settings.top_k` (the single-document /ask
#: default): with up to MAX_DOCUMENTS documents each contributing this
#: many chunks, the combined evidence context still stays a bounded,
#: predictable size instead of growing unboundedly with document count.
PER_DOCUMENT_TOP_K = 6


@dataclass(frozen=True)
class ResolvedDocument:
    id: int
    filename: str
    title: str
    source_root: str
    file_type: str


def resolve_documents(
    session: Session, document_ids: list[int], current_user: User | None
) -> list[ResolvedDocument]:
    """Resolve every id in `document_ids` to a `ResolvedDocument`,
    enforcing tenant isolation exactly like `app.api.source._get_document_or_404`.

    Raises `HTTPException(404)` naming the FIRST inaccessible id if any
    document doesn't exist or isn't visible to `current_user` — a
    multi-document feature (compare/extract/summarize/report) that
    silently dropped an inaccessible id from the set would produce a
    result that LOOKS complete but quietly omitted a document the caller
    explicitly asked for; failing loudly is the safer default here (spec:
    "never combine evidence without identifying its source" extends to
    never silently combining a SUBSET of the requested sources either).

    Raises `HTTPException(422)` if `document_ids` is empty or has more
    than `MAX_DOCUMENTS` entries (bounded, per the spec's "keep the
    workflow bounded" instruction — an unbounded multi-document fan-out
    is exactly the kind of uncontrolled cost/latency blowup that
    instruction exists to prevent).
    """
    if not document_ids:
        raise HTTPException(status_code=422, detail="document_ids must not be empty.")
    if len(document_ids) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"document_ids must not contain more than {MAX_DOCUMENTS} documents (got {len(document_ids)}).",
        )

    resolved: list[ResolvedDocument] = []
    for document_id in document_ids:
        doc = session.get(Document, document_id)
        if doc is None or (doc.owner_id is not None and (current_user is None or doc.owner_id != current_user.id)):
            raise HTTPException(status_code=404, detail=f"No document with id={document_id}")
        resolved.append(
            ResolvedDocument(
                id=doc.id,
                filename=doc.filename,
                title=doc.title or doc.filename,
                source_root=doc.source_root,
                file_type=doc.file_type,
            )
        )
    return resolved


def per_document_evidence_context(
    chunks_by_document: dict[int, list[dict]], resolved: list[ResolvedDocument]
) -> tuple[str, dict[str, str]]:
    """Build ONE combined, label-indirected evidence context spanning
    MULTIPLE documents, with each block explicitly tagged with which
    document it came from — the label-indirection pattern
    `app.rag.llm._evidence_context`/`app.rag.grounding._evidence_context`
    already use (never let the model see/return a real chunk_id — see
    those functions' docstrings), extended for Phase 6's multi-document
    features which additionally need the model to know WHICH document
    each block belongs to (spec: "never combine evidence without
    identifying its source").

    Labels are `EVIDENCE_<doc_id>_<n>` (not a bare running counter) so
    the document boundary is legible directly in the label itself, both
    to the model and in `label_to_chunk_id` for debugging/tracing —
    `doc_id` here is `ResolvedDocument.id` (the citation-safe internal
    id), never a raw filename, keeping the same "never let the model
    echo something that could be confused with real user-controlled
    text" property `_candidate_context`'s docstring describes.

    Injection scanning/wrapping (`app.security.prompt_injection`) is
    applied per-chunk exactly like the single-document path — a
    multi-document feature reads MORE untrusted document text per call,
    not less, so this defense matters at least as much here.

    Returns `(context_text, label_to_chunk_id)` — the label map is
    keyed by the SAME `EVIDENCE_<doc_id>_<n>` labels, so a caller
    resolving the model's cited labels back to real `chunk_id`s (for
    citation objects) uses this dict exactly like every single-document
    caller already does.
    """
    titles = {r.id: r.title for r in resolved}
    label_to_chunk_id: dict[str, str] = {}
    blocks: list[str] = []
    for doc_id, chunks in chunks_by_document.items():
        scans = scan_chunks(chunks)
        for i, c in enumerate(chunks, start=1):
            label = f"EVIDENCE_{doc_id}_{i}"
            label_to_chunk_id[label] = c["chunk_id"]
            wrapped = wrap_untrusted(c["text"], label, scan=scans.get(c["chunk_id"]))
            blocks.append(
                f"{label}\nDocument: {titles.get(doc_id, doc_id)} (document_id={doc_id})\n"
                f"Source file: {c['source_file']}\nContent:\n{wrapped}"
            )
    context = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    return context, label_to_chunk_id