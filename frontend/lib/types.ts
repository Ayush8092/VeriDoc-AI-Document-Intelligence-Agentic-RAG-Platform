// Mirrors app/schemas/documents.py on the backend. Keep these two in sync
// by hand — there are only a handful of shapes, and a generated-client
// step isn't worth it yet for a single frontend consumer.

// Phase 4.1(5): block_type now spans every multimodal block type the
// backend can cite, not just text/table — see
// backend/app/documents/models.py::BlockType. "visual" is the
// unclassified-but-detected fallback (VisualObjectType.UNKNOWN_VISUAL
// on the backend); everything else is a specific, classified kind.
export type BlockType = "text" | "table" | "figure" | "chart" | "visual";
export type ExtractionSource = "native" | "ocr";

// Mirrors backend VisualObjectType (app/documents/models.py). Present on
// a citation only when block_type is "figure" | "chart" | "visual".
export type VisualType =
  | "figure"
  | "photo"
  | "diagram"
  | "illustration"
  | "screenshot"
  | "chart"
  | "unknown_visual";

// Mirrors backend ChartType. Present only when block_type === "chart"
// AND the chart-understanding step classified it with enough confidence
// to commit to a type — otherwise "unknown_chart" (never omitted/null
// for a chart citation; the backend always assigns one of these).
export type ChartType = "bar" | "line" | "pie" | "scatter" | "area" | "unknown_chart";

// Mirrors backend CoordinateSpace (app/documents/models.py). NEVER
// interpret a bbox without checking this — a PDF_POINTS bbox and an
// IMAGE_PIXELS bbox use different origins/scales/units and are not
// interchangeable. See lib/bbox.ts for the one place that's allowed to
// know how to map either of these onto a rendered page image.
export type CoordinateSpace = "pdf_points" | "image_pixels" | "unspecified";

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  coordinate_space: CoordinateSpace;
}

export interface Citation {
  chunk_id: string;
  source_file: string;
  section: string;
  snippet: string;
  score: number;
  block_type: BlockType;
  page_start: number | null;
  page_end: number | null;
  source: ExtractionSource;
  confidence: number;
  table_rows: string[][] | null;
  // Phase 4 multimodal provenance — all optional/nullable: present only
  // for figure/chart/visual citations (and bbox/coordinate_space, when
  // the underlying object's geometry was extracted rather than
  // OCR-only/undetermined). A text/table citation predating Phase 4 (or
  // simply not visual) legitimately has every one of these as null —
  // that is not missing data, it's "not applicable to this block type".
  object_id: string | null;
  bbox: BoundingBox | null;
  coordinate_space: CoordinateSpace | null;
  caption: string | null;
  visual_type: VisualType | null;
  chart_type: ChartType | null;
  related_text: string | null;
}

// "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" — see
// backend app/rag/grounding.py.
export type ClaimLabel = "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED";

// Mirrors backend ClaimSupport (app/schemas/documents.py) — one piece of
// evidence backing a Claim, read straight off an already-retrieved chunk
// (never model output) by app/rag/grounding.py::ground_answer.
export interface ClaimSupport {
  chunk_id: string;
  page: number | null;
  block_type: BlockType;
  object_id: string | null;
  visual_type: VisualType | null;
  chart_type: ChartType | null;
  caption: string | null;
  bbox: BoundingBox | null;
  coordinate_space: CoordinateSpace | null;
}

export interface Claim {
  claim: string;
  label: ClaimLabel;
  reason: string;
  support: ClaimSupport[];
}

export interface StageLatency {
  stage: string;
  elapsed_ms: number;
  message: string;
}

export interface AskResponse {
  answer: string;
  found: boolean;
  citations: Citation[];
  // Phase 3A additions — all optional/nullable so older cached responses
  // (or a backend running without Phase 3A features enabled) still
  // satisfy this type.
  query_type: string | null;
  claims: Claim[];
  grounded_claim_rate: number | null;
  citation_precision: number | null;
  citation_recall: number | null;
  trace: string[] | null;
  trace_id: string | null;
  usage: Record<string, unknown> | null;
  // Phase 7 trace viewer (item 18). Mirrors backend
  // app/schemas/documents.py's AskResponse.stage_latencies — present only
  // when the request was made with trace=true (same gate as `trace`/
  // `usage` above).
  stage_latencies: StageLatency[] | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  question?: string;
  response?: AskResponse;
  pending?: boolean;
  error?: string;
}

export interface HealthResponse {
  status: string;
}

export interface ReadyResponse {
  ready: boolean;
  detail: string;
}

export interface UploadRejection {
  filename: string;
  error: string;
}

// Phase 5 completion pass, item 2 (true asynchronous ingestion): one
// accepted file's queued job — mirrors backend
// app/schemas/documents.py::UploadJobRef. Poll it via getJobStatus()
// (before a document_id exists yet) or, once the job links to a
// document, getIngestionStatus(documentId).
export interface UploadJobRef {
  filename: string;
  job_id: number;
}

export interface UploadResponse {
  // "error" (nothing accepted) | "queued" (every accepted file was
  // durably persisted and handed to the background worker — ingestion
  // itself has NOT run yet). The old "success"/"partial_success"
  // synchronous contract no longer applies — see UploadResponse's
  // backend docstring. chunks_created/vectors_upserted/etc. below are
  // always 0 in THIS response now; poll `jobs[].job_id` for the real
  // outcome.
  status: "queued" | "error";
  files: string[];
  rejected: UploadRejection[];
  jobs: UploadJobRef[];
  chunks_created: number;
  vectors_upserted: number;
  unchanged_chunks: number;
  stale_vectors_deleted: number;
  total_chunks: number;
  namespace_vector_count: number;
}

// Phase 5 completion pass, item 3.3 — mirrors backend
// app/schemas/ingestion.py exactly.
export type IngestionJobStatus =
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "retrying"
  | "cancelled";

export interface IngestionJobOut {
  id: number;
  document_id: number | null;
  version_id: number | null;
  filename: string;
  source_root: string;
  status: IngestionJobStatus;
  attempt_count: number;
  max_attempts: number;
  error: string;
  chunks_total: number;
  chunks_changed: number;
  vectors_upserted: number;
  duration_seconds: number | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface DocumentIngestionStatusOut {
  document_id: number;
  filename: string;
  current_version_id: number | null;
  has_current_version: boolean;
  latest_job: IngestionJobOut | null;
}

export interface ReingestResponse {
  status: string;
  job: IngestionJobOut;
}

export interface DocumentVersionOut {
  id: number;
  version_number: number;
  file_hash: string;
  page_count: number;
  has_scanned_pages: boolean;
  table_count: number;
  chunk_count: number;
  mean_ocr_confidence: number | null;
  warnings: string[];
  created_at: string | null;
}

export interface DocumentOut {
  id: number;
  source_root: "corpus" | "uploads";
  filename: string;
  file_type: string;
  title: string;
  owner_id: number | null;
  created_at: string | null;
  updated_at: string | null;
  current_version: DocumentVersionOut | null;
}

export interface DocumentListResponse {
  documents: DocumentOut[];
}

// Mirrors backend DeleteDocumentResponse (app/schemas/documents.py).
export interface DeleteDocumentResponse {
  status: string; // "deleted"
  document_id: number;
  filename: string;
  jobs_deleted: number;
  vectors_deleted: number;
  lexical_index_chunks_remaining: number;
}

// Phase 5 completion pass, item 2: real backend job states, not a
// frontend-invented progress simulation — see UploadItem below and
// app/upload/page.tsx. "uploading" is the one client-only state (the
// HTTP request itself is still in flight); everything after that
// mirrors IngestionJobStatus verbatim.
export type UploadFileStatus = "queued" | "uploading" | "rejected" | IngestionJobStatus;

export interface UploadItem {
  id: string;
  file: File;
  status: UploadFileStatus;
  error?: string;
  /** Set once POST /documents/upload accepts this file and returns a job id. */
  jobId?: number;
  /** Set once the job links to a real Document (ingestion succeeded at least once). */
  documentId?: number;
}

// ---------------------------------------------------------------------
// Phase 4.1(5): Source Viewer — mirrors backend app/api/source.py.
// ---------------------------------------------------------------------

// GET /documents/by-filename/{filename} and the `document` field on
// SourceLocateResponse resolve a citation's bare `source_file` (all a
// Citation carries — see lib/types.ts's Citation, which deliberately
// wasn't given a document_id to avoid touching the existing citation
// construction path) to the Document row the viewer needs.
export interface SourceDocumentRef {
  id: number;
  source_root: "corpus" | "uploads";
  filename: string;
  file_type: string;
  page_count: number;
  has_scanned_pages: boolean;
}

// One "can this document even be viewed as page images" answer, plus
// (when yes) how to fetch them. A viewer must check `renderable` before
// requesting a page image — a .txt/.md source has no page geometry at
// all, and the citation's `snippet`/`section` is the only "source
// preview" it will ever have.
export interface SourceCapabilities {
  renderable: boolean;      // can /pages/{n}/image be requested at all
  reason?: string;          // present when renderable is false (e.g. "text documents have no page images")
  page_count: number;
}

// Phase 4.1(7): mirrors backend app/schemas/documents.py::PageObjectOut.
// One chunk (text/table/figure/chart/visual) whose page range covers a
// given page — the source viewer's "everything on this page" listing.
// Deliberately NOT a `Citation` (see the backend model's docstring): no
// relevance score, since this is "everything on the page", not a
// ranked search result.
export interface PageObject {
  chunk_id: string;
  object_id: string | null;
  block_type: BlockType;
  section: string;
  snippet: string;
  page_start: number | null;
  page_end: number | null;
  source: ExtractionSource;
  confidence: number;
  table_rows: string[][] | null;
  bbox: BoundingBox | null;
  coordinate_space: CoordinateSpace | null;
  caption: string | null;
  visual_type: VisualType | null;
  chart_type: ChartType | null;
  related_text: string | null;
}

export interface PageObjectsResponse {
  document_id: number;
  page_number: number;
  objects: PageObject[];
}

// Phase 4.1(7): the Source Viewer's single resolved-target shape,
// regardless of how it got there (citation click-through vs. direct
// document/page selection) — see app/source/page.tsx.
export interface SourceLocateResponse {
  document: SourceDocumentRef;
  capabilities: SourceCapabilities;
  page_number: number;
}

// ---------------------------------------------------------------------
// Phase 5 completion pass, item 6: Authentication — mirrors backend
// app/schemas/auth.py.
// ---------------------------------------------------------------------

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in_minutes: number;
}

export interface UserOut {
  id: number;
  email: string;
  is_active: boolean;
  created_at: string | null;
}

// ---------------------------------------------------------------------
// Phase 6 — mirrors backend app/schemas/reasoning.py.
// fix_phase_6, Problem 3: these types were imported by lib/api.ts and
// several pages but never actually defined here, breaking the frontend
// build. Every field name/shape below is copied directly from the
// backend Pydantic models, not guessed.
// ---------------------------------------------------------------------

export interface DocumentRef {
  id: number;
  title: string;
  filename: string;
}

// Deliberately leaner than Citation (see backend app/schemas/reasoning.py's
// EvidenceCitation docstring) — no score/section/snippet, since Phase 6
// features cite specific resolved evidence chunks, not a ranked
// retrieval result.
export interface EvidenceCitation {
  chunk_id: string;
  source_file: string;
  page_start: number | null;
  page_end: number | null;
  document_id: number | null;
}

// ---- Comparison ----

export interface CompareRequest {
  topic: string;
  document_ids: number[];
}

export type ComparisonRelation = "same" | "different" | "missing_in_a" | "missing_in_b" | "conflicting";

export interface ComparisonPointTwoDoc {
  aspect: string;
  relation: ComparisonRelation;
  summary_a: string;
  summary_b: string;
  evidence_a: EvidenceCitation[];
  evidence_b: EvidenceCitation[];
}

export interface TwoDocumentComparison {
  topic: string;
  document_a: DocumentRef;
  document_b: DocumentRef;
  points: ComparisonPointTwoDoc[];
}

export type ComparisonConsensus = "agree" | "disagree" | "partial";

export interface ComparisonPointMultiDoc {
  aspect: string;
  consensus: ComparisonConsensus;
  summary_by_document: Record<string, string>;
  evidence_by_document: Record<string, EvidenceCitation[]>;
}

export interface MultiDocumentComparison {
  topic: string;
  documents: DocumentRef[];
  points: ComparisonPointMultiDoc[];
}

// Mirrors backend POST /compare's actual response_model — a UNION, not
// a single Pydantic class (app/api/reasoning.py:
// `response_model=CompareTwoDocResponse | CompareMultiDocResponse`):
// exactly 2 documents get the two-sided shape (document_a/document_b),
// 3+ get the N-way shape (documents/consensus). Never both at once —
// see isTwoDocComparison below for the one correct way to tell them
// apart.
export type CompareResponse = TwoDocumentComparison | MultiDocumentComparison;

/** The one correct discriminator between CompareResponse's two shapes —
 * `document_a` only exists on TwoDocumentComparison. Every other field
 * that might look distinguishing (`points[].aspect`, `topic`) exists on
 * both shapes and would not safely narrow the type. */
export function isTwoDocComparison(result: CompareResponse): result is TwoDocumentComparison {
  return "document_a" in result;
}

// ---- Extraction ----

export type ExtractionFieldType = "string" | "number" | "date" | "boolean" | "list[string]";

export interface ExtractRequest {
  document_ids: number[];
  schema: Record<string, ExtractionFieldType>;
}

export interface ExtractedField {
  value: string | number | boolean | string[] | null;
  status: "found" | "not_found";
  confidence: number;
  citations: EvidenceCitation[];
}

export interface ExtractResponse {
  fields: Record<string, ExtractedField>;
  documents: DocumentRef[];
}

// ---- Summarization ----

export type SummaryMode = "document" | "executive" | "key_decisions" | "key_risks" | "action_items";
// Alias — app/summarize/page.tsx imports SummaryMode; SummarizeResponse's
// own `mode` field and SummarizeRequest below are named to mirror the
// backend's SummarizeRequest.mode/SummarizeResponse.mode exactly. Same
// type, two names in active use across the codebase; kept both rather
// than picking one and silently breaking the other's import.
export type SummarizeMode = SummaryMode;

export interface SummarizeRequest {
  document_ids: number[];
  mode?: SummaryMode;
}

export interface SummaryPoint {
  text: string;
  citations: EvidenceCitation[];
}

export interface PerDocumentSummary {
  summary: string;
  points: SummaryPoint[];
  chunk_count: number;
}

export interface SummarizeResponse {
  mode: SummaryMode;
  documents: DocumentRef[];
  per_document: Record<string, PerDocumentSummary>;
  combined_summary: string;
  combined_points: SummaryPoint[];
}

// ---- Reports ----

export interface ReportRequest {
  title: string;
  document_ids: number[];
}

export interface Recommendation {
  text: string;
  basis: "inference"; // always "inference" — see backend app/rag/report.py's module docstring
  based_on_findings: SummaryPoint[];
}

export interface ReportAppendix {
  total_documents: number;
  total_chunks_considered: number;
}

export interface ReportResponse {
  title: string;
  documents: DocumentRef[];
  executive_summary: string;
  key_findings: SummaryPoint[];
  detailed_comparison: TwoDocumentComparison | MultiDocumentComparison | null;
  risks: SummaryPoint[];
  recommendations: Recommendation[];
  appendix: ReportAppendix;
}

// ---------------------------------------------------------------------
// Phase 6 — Conversations. Mirrors backend app/schemas/conversations.py.
// ---------------------------------------------------------------------

export interface ConversationCreate {
  title?: string;
  document_ids?: number[];
  // Present an existing anonymous session token to continue that
  // session rather than starting a new, empty one — ignored by the
  // backend for an authenticated request (Bearer token identity always
  // wins; see backend app/services/conversation_memory.py).
  anonymous_session_id?: string | null;
}

export interface ConversationOut {
  id: number;
  owner_id: number | null;
  anonymous_session_id: string | null;
  title: string;
  document_ids: number[];
  created_at: string | null;
  updated_at: string | null;
}

export interface ConversationListResponse {
  conversations: ConversationOut[];
}

export interface ConversationMessageOut {
  id: number;
  conversation_id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  retrieved_chunk_ids: string[];
  query_type: string | null;
  created_at: string | null;
}

export interface ConversationTurnRequest {
  question: string;
  // Only needed for an anonymous caller continuing a session across
  // requests — an authenticated caller's identity comes from the
  // Bearer token, same as every other endpoint.
  anonymous_session_id?: string | null;
}

export interface ConversationTurnResponse {
  conversation_id: number;
  user_message: ConversationMessageOut;
  assistant_message: ConversationMessageOut;
  // Echoed back so an anonymous caller without one yet learns the
  // session id assigned on first use. null for an authenticated caller.
  anonymous_session_id: string | null;
}