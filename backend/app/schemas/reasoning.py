"""Request/response models for Phase 6's multi-document reasoning
features: comparison, structured extraction, summarization, and report
generation (spec item 14: "Use Pydantic schemas. Do not return
unstructured LLM output when structured output is expected.").

Every response model here mirrors the return shape its corresponding
`app.rag.*` module already produces field-for-field — these schemas
don't reshape or rename anything the underlying module computed, so
there's exactly one place (the `app.rag.*` module) that defines what
the actual data looks like.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentRef(BaseModel):
    """A document identified in a Phase 6 response — the same small,
    display-oriented subset `app.rag.multi_doc.ResolvedDocument`
    already carries."""

    id: int
    title: str
    filename: str


class EvidenceCitation(BaseModel):
    """A citation attached to a specific field/point/claim in a Phase 6
    response — deliberately leaner than `app.schemas.documents.Citation`
    (no score/section/snippet): these features cite specific evidence
    chunks resolved via `app.rag.multi_doc.per_document_evidence_context`'s
    label map or `app.vectorstore.query_by_filter`, not a ranked
    retrieval result — those extra fields don't exist for this access
    pattern and citing empty scores/sections here would misrepresent
    what a Phase 6 citation actually is.
    """

    chunk_id: str
    source_file: str
    page_start: int | None = None
    page_end: int | None = None
    document_id: int | None = None


# ---------------------------------------------------------------------
# Comparison (spec items 2 & 3)
# ---------------------------------------------------------------------


class CompareRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="What to compare across the selected documents.")
    document_ids: list[int] = Field(..., min_length=2, description="2+ document ids to compare.")


class ComparisonPointTwoDoc(BaseModel):
    aspect: str
    relation: str  # "same" | "different" | "missing_in_a" | "missing_in_b" | "conflicting"
    summary_a: str
    summary_b: str
    evidence_a: list[EvidenceCitation]
    evidence_b: list[EvidenceCitation]


class CompareTwoDocResponse(BaseModel):
    topic: str
    document_a: DocumentRef
    document_b: DocumentRef
    points: list[ComparisonPointTwoDoc]


class ComparisonPointMultiDoc(BaseModel):
    aspect: str
    consensus: str  # "agree" | "disagree" | "partial"
    summary_by_document: dict[str, str]
    evidence_by_document: dict[str, list[EvidenceCitation]]


class CompareMultiDocResponse(BaseModel):
    topic: str
    documents: list[DocumentRef]
    points: list[ComparisonPointMultiDoc]


# ---------------------------------------------------------------------
# Structured extraction (spec item 4)
# ---------------------------------------------------------------------


class ExtractRequest(BaseModel):
    document_ids: list[int] = Field(..., min_length=1)
    schema_: dict[str, str] = Field(
        ...,
        alias="schema",
        description='Field name -> type ("string" | "number" | "date" | "boolean" | "list[string]").',
    )

    model_config = {"populate_by_name": True}


class ExtractedField(BaseModel):
    value: object = None
    status: str  # "found" | "not_found"
    confidence: float
    citations: list[EvidenceCitation]


class ExtractResponse(BaseModel):
    fields: dict[str, ExtractedField]
    documents: list[DocumentRef]


# ---------------------------------------------------------------------
# Summarization (spec item 5)
# ---------------------------------------------------------------------


class SummarizeRequest(BaseModel):
    document_ids: list[int] = Field(..., min_length=1)
    mode: str = Field(
        default="document",
        description="document | executive | key_decisions | key_risks | action_items",
    )


class SummaryPoint(BaseModel):
    text: str
    citations: list[EvidenceCitation]


class PerDocumentSummary(BaseModel):
    summary: str
    points: list[SummaryPoint]
    chunk_count: int


class SummarizeResponse(BaseModel):
    mode: str
    documents: list[DocumentRef]
    per_document: dict[str, PerDocumentSummary]
    combined_summary: str
    combined_points: list[SummaryPoint]


# ---------------------------------------------------------------------
# Report generation (spec item 6)
# ---------------------------------------------------------------------


class ReportRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    document_ids: list[int] = Field(..., min_length=1)


class Recommendation(BaseModel):
    text: str
    basis: str  # always "inference" — see app.rag.report's module docstring
    based_on_findings: list[SummaryPoint]


class ReportAppendix(BaseModel):
    total_documents: int
    total_chunks_considered: int


class ReportResponse(BaseModel):
    title: str
    documents: list[DocumentRef]
    executive_summary: str
    key_findings: list[SummaryPoint]
    detailed_comparison: CompareTwoDocResponse | CompareMultiDocResponse | None = None
    risks: list[SummaryPoint]
    recommendations: list[Recommendation]
    appendix: ReportAppendix