"""Request/response models for the API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer.")


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    section: str
    snippet: str
    score: float
    block_type: str = "text"
    page_start: int | None = None
    page_end: int | None = None
    source: str = "native"
    confidence: float = 1.0
    table_rows: list[list[str]] | None = None
    # Phase 4.1(4) fix: these were already computed by
    # `app.rag.llm.validate_citations` and silently dropped here — a
    # Pydantic model only serializes fields it declares, so every
    # visual citation's object_id/bbox/etc. was discarded at the exact
    # last step before reaching the API response, despite surviving
    # every step before it. See docs/architecture.md, "Phase 4.1(4)
    # metadata propagation fix".
    object_id: str | None = None
    bbox: dict | None = None
    coordinate_space: str | None = None
    caption: str | None = None
    visual_type: str | None = None
    chart_type: str | None = None
    related_text: str | None = None


class ClaimSupport(BaseModel):
    """One piece of evidence backing a claim (`Claim.support`) — see
    `app.rag.grounding.ground_answer`'s docstring. Every field here is
    read straight off an already-retrieved chunk, never model output —
    same anti-fabrication guarantee as `Citation`.
    """

    chunk_id: str
    page: int | None = None
    block_type: str = "text"
    object_id: str | None = None
    visual_type: str | None = None
    chart_type: str | None = None
    caption: str | None = None
    bbox: dict | None = None
    coordinate_space: str | None = None


class Claim(BaseModel):
    claim: str
    label: str  # "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED"
    reason: str = ""
    # Phase 4.1(4) fix: `app.rag.grounding.ground_answer` already computes
    # per-claim supporting evidence (chunk_id/page/object_id/chart_type/
    # bbox/...) — this field was missing here, so it was computed and
    # then discarded before the API response, the same class of bug as
    # `Citation` above.
    support: list[ClaimSupport] = []


class AskResponse(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]
    query_type: str | None = None
    claims: list[Claim] = []
    grounded_claim_rate: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    trace: list[str] | None = None
    # Phase 7 trace viewer (item 18): per-node timing breakdown from
    # DocumentQAService.ask()'s `stage_latencies` (app/rag/graph.py's
    # `_timed_trace`) — e.g. [{"stage": "retrieve", "elapsed_ms": 121.4,
    # "message": "12 chunks"}, ...]. Previously computed internally on
    # every request but silently discarded before reaching this response
    # (found during the Phase 7 audit) — a trace viewer had no real data
    # to render. Opt-in alongside `trace`/`usage` (same `include_trace`
    # gate in app/api/ask.py), since it's the same kind of
    # debug/observability payload a UI wouldn't render by default.
    stage_latencies: list[dict] | None = None
    # Phase 3B observability (app/observability/trace.py) — trace_id is
    # always present (cheap, and useful to correlate with server logs even
    # without ?trace=true); usage/cost/latency detail is opt-in like the
    # LangGraph step trace above, since it's the kind of payload a UI
    # wouldn't render by default.
    trace_id: str | None = None
    usage: dict | None = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    detail: str
    # Phase 5 section 11: itemized per-dependency status ("database",
    # "vector_store", ...) -> True/False, so a caller (or a human staring
    # at /ready) can tell WHICH dependency is the problem instead of just
    # "something is wrong". Optional/additive — `detail` alone remains a
    # valid, backward-compatible response shape for any existing caller
    # that only reads `ready`/`detail`.
    checks: dict[str, bool] = {}


class UploadRejection(BaseModel):
    filename: str
    error: str


class UploadJobRef(BaseModel):
    """One accepted file's queued job (Phase 5 completion pass, item 2:
    true asynchronous ingestion). Poll `GET /documents/jobs/{job_id}`
    (before `document_id` is known — ingestion hasn't created the
    `Document` row yet) or, once known, `GET /documents/{id}/ingestion`.
    """

    filename: str
    job_id: int


class UploadResponse(BaseModel):
    # "error" (nothing accepted — same synchronous meaning as before this
    # pass) | "queued" (every accepted file has been durably persisted —
    # DB job row committed, bytes written to storage — and handed to the
    # background worker; ingestion itself has NOT run yet, so no chunk/
    # vector counts are known at response time — see UploadJobRef & this
    # module's docstring, "Phase 5 completion pass, item 2").
    #
    # The previous synchronous contract ("success" / "partial_success",
    # with chunks_created/vectors_upserted/etc. populated in THIS
    # response) no longer applies — poll the job endpoints for the
    # actual outcome. Old field names are kept (rather than removed) so
    # existing response consumers don't hard-crash on a missing key, but
    # they are now always 0/unset until the job completes; they do not
    # reflect this specific request. See CHANGELOG entry in
    # docs/architecture.md, "Phase 5 completion pass — async upload".
    status: str
    files: list[str]
    rejected: list[UploadRejection] = []
    jobs: list[UploadJobRef] = []
    chunks_created: int = 0
    vectors_upserted: int = 0
    unchanged_chunks: int = 0
    stale_vectors_deleted: int = 0
    total_chunks: int = 0
    namespace_vector_count: int = 0


class DocumentVersionOut(BaseModel):
    id: int
    version_number: int
    file_hash: str
    page_count: int
    has_scanned_pages: bool
    table_count: int
    chunk_count: int
    mean_ocr_confidence: float | None
    warnings: list[str]
    created_at: str | None


class DocumentOut(BaseModel):
    id: int
    source_root: str
    filename: str
    file_type: str
    title: str
    owner_id: int | None = None
    created_at: str | None
    updated_at: str | None
    current_version: DocumentVersionOut | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]


class DeleteDocumentResponse(BaseModel):
    """`DELETE /documents/{id}` response (Phase 5 completion pass, round
    2, item 19 — document lifecycle: delete). Reports what was actually
    removed from each system, not just "ok" — deletion touches the
    metadata DB, object storage, the Pinecone vector index, and the BM25
    lexical index snapshot, and a caller auditing "is this document
    really gone everywhere" deserves to see all four confirmed, not
    trust a single boolean.
    """

    status: str  # "deleted"
    document_id: int
    filename: str
    jobs_deleted: int
    vectors_deleted: int
    lexical_index_chunks_remaining: int


class PageObjectOut(BaseModel):
    """One chunk (text/table/figure/chart/visual) whose page range covers
    a requested page — Phase 4.1(7) source viewer's "everything on this
    page" listing (`GET /documents/{id}/pages/{n}/objects`).

    Deliberately its OWN model rather than reusing `Citation`: a page
    listing has no retrieval relevance score (every object on the page
    is returned, not a ranked top-k), and rendering it with a `Citation`
    -shaped "relevance X%" would misrepresent that. Every other field
    mirrors `Citation` exactly (same provenance guarantees, same
    Phase 4.1(4) fix) so the frontend's bbox-projection/metadata-display
    logic works identically for both.
    """

    chunk_id: str
    object_id: str | None = None
    block_type: str = "text"
    section: str = ""
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    source: str = "native"
    confidence: float = 1.0
    table_rows: list[list[str]] | None = None
    bbox: dict | None = None
    coordinate_space: str | None = None
    caption: str | None = None
    visual_type: str | None = None
    chart_type: str | None = None
    related_text: str | None = None


class PageObjectsResponse(BaseModel):
    document_id: int
    page_number: int
    objects: list[PageObjectOut]