import type {
  AskResponse,
  CompareRequest,
  CompareResponse,
  ConversationCreate,
  ConversationListResponse,
  ConversationMessageOut,
  ConversationOut,
  ConversationTurnRequest,
  ConversationTurnResponse,
  DeleteDocumentResponse,
  DocumentIngestionStatusOut,
  DocumentListResponse,
  DocumentOut,
  ExtractRequest,
  ExtractResponse,
  HealthResponse,
  IngestionJobOut,
  PageObjectsResponse,
  ReadyResponse,
  ReingestResponse,
  ReportRequest,
  ReportResponse,
  SourceCapabilities,
  SourceDocumentRef,
  SourceLocateResponse,
  SummarizeRequest,
  SummarizeResponse,
  TokenResponse,
  UploadResponse,
  UserOut,
} from "./types";
import { getStoredToken, notifyUnauthorized } from "./auth-storage";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      // FastAPI/pydantic validation error shape
      return body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ") || res.statusText;
    }
  } catch {
    // response wasn't JSON — fall through to statusText
  }
  return res.statusText || `Request failed (${res.status})`;
}

/** Phase 5 completion pass, item 6: every request automatically carries
 * `Authorization: Bearer <token>` when a session exists — no page/
 * component attaches this header itself. A `401` (token missing/
 * expired/invalid — the backend never distinguishes which, by design;
 * see app/security/auth.py) clears the stored session and notifies
 * `AuthProvider` via `notifyUnauthorized()` before the error still
 * propagates to the caller, so a component mid-request can show its own
 * "couldn't load" state AND the global auth state resets in the same
 * beat, rather than the two drifting out of sync. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getStoredToken();
  const headers = new Headers(init?.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(
      `Could not reach the Veridoc API at ${API_BASE_URL}. Is the backend running?`,
      0,
    );
  }
  if (res.status === 401 && token) {
    notifyUnauthorized();
  }
  if (!res.ok) {
    throw new ApiError(await parseErrorDetail(res), res.status);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getReady(): Promise<ReadyResponse> {
  return request<ReadyResponse>("/ready");
}

export function ask(question: string, trace = false): Promise<AskResponse> {
  return request<AskResponse>(`/ask${trace ? "?trace=true" : ""}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

export function listDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/documents");
}

export function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  return request<UploadResponse>("/documents/upload", {
    method: "POST",
    body: form,
  });
}

// ---------------------------------------------------------------------
// Phase 4.1(5)/(7): Source Viewer — mirrors backend app/api/source.py.
// ---------------------------------------------------------------------

/** `GET /documents/{id}` and `/documents/by-filename/{filename}` return
 * the same `DocumentOut` shape the Library page uses (nested
 * `current_version`) — flattened here into the viewer's own
 * `SourceDocumentRef` shape (page_count/has_scanned_pages pulled up to
 * the top level) so viewer components don't have to null-check
 * `current_version` on every read. A document with no ingested version
 * yet (`current_version: null`) becomes `page_count: 0`,
 * `has_scanned_pages: false` — a real, renderable-false state, not an
 * error. */
function toSourceDocumentRef(doc: DocumentOut): SourceDocumentRef {
  return {
    id: doc.id,
    source_root: doc.source_root,
    filename: doc.filename,
    file_type: doc.file_type,
    page_count: doc.current_version?.page_count ?? 0,
    has_scanned_pages: doc.current_version?.has_scanned_pages ?? false,
  };
}

/** First hop of Citation -> Document: resolve a citation's bare
 * `source_file` to the Document row the viewer needs. A Citation
 * deliberately doesn't carry a document_id (see source.py's docstring
 * for why), so every source-viewer entry point starts here. */
export async function getDocumentByFilename(filename: string): Promise<SourceDocumentRef> {
  const doc = await request<DocumentOut>(`/documents/by-filename/${encodeURIComponent(filename)}`);
  return toSourceDocumentRef(doc);
}

export async function getDocument(documentId: number): Promise<SourceDocumentRef> {
  const doc = await request<DocumentOut>(`/documents/${documentId}`);
  return toSourceDocumentRef(doc);
}

export function getSourceCapabilities(documentId: number): Promise<SourceCapabilities> {
  return request<SourceCapabilities>(`/documents/${documentId}/capabilities`);
}

export function sourceFileUrl(documentId: number): string {
  return `${API_BASE_URL}/documents/${documentId}/source`;
}

/** Every chunk (text/table/figure/chart/visual) whose page range covers
 * `pageNumber` — the source viewer's "what else is on this page" panel
 * and the mechanism behind citation -> visual object navigation. */
export function getPageObjects(documentId: number, pageNumber: number): Promise<PageObjectsResponse> {
  return request<PageObjectsResponse>(`/documents/${documentId}/pages/${pageNumber}/objects`);
}

/** One-call convenience for the viewer's entry point: resolve a
 * filename to its Document, fetch whether/how it can be rendered, and
 * clamp the requested page into range — everything a viewer route needs
 * before its first paint, in one round-trip pair instead of three
 * sequential ones scattered across component effects. */
export async function locateSource(
  filename: string,
  requestedPage = 1,
): Promise<SourceLocateResponse> {
  const document = await getDocumentByFilename(filename);
  const capabilities = await getSourceCapabilities(document.id);
  const clampedPage = capabilities.page_count > 0
    ? Math.min(Math.max(1, requestedPage), capabilities.page_count)
    : requestedPage;
  return { document, capabilities, page_number: clampedPage };
}

/** Same as `locateSource`, but starting from a known `document_id`
 * (the document-picker entry point, or a `?document=`/`?page=` deep
 * link) rather than a citation's filename. */
export async function locateSourceById(
  documentId: number,
  requestedPage = 1,
): Promise<SourceLocateResponse> {
  const document = await getDocument(documentId);
  const capabilities = await getSourceCapabilities(documentId);
  const clampedPage = capabilities.page_count > 0
    ? Math.min(Math.max(1, requestedPage), capabilities.page_count)
    : requestedPage;
  return { document, capabilities, page_number: clampedPage };
}

/** Returns the page image's blob URL (for an <img src>), its ACTUAL
 * rendered pixel dimensions, and — when the backend could determine it —
 * the page/image's NATIVE (unscaled) size, all from response headers
 * (see source.py's docstring on why bbox scaling needs both). Caller is
 * responsible for revoking the returned object URL when done with it
 * (`URL.revokeObjectURL`) to avoid leaking memory across navigations. */
export async function getPageImage(
  documentId: number,
  pageNumber: number,
  dpi = 150,
): Promise<{ url: string; width: number; height: number; nativeWidth: number | null; nativeHeight: number | null }> {
  const res = await fetch(`${API_BASE_URL}/documents/${documentId}/pages/${pageNumber}/image?dpi=${dpi}`);
  if (!res.ok) {
    throw new ApiError(await parseErrorDetail(res), res.status);
  }
  const width = Number(res.headers.get("X-Rendered-Width"));
  const height = Number(res.headers.get("X-Rendered-Height"));
  const nativeWidthHeader = res.headers.get("X-Native-Width");
  const nativeHeightHeader = res.headers.get("X-Native-Height");
  const blob = await res.blob();
  return {
    url: URL.createObjectURL(blob),
    width,
    height,
    nativeWidth: nativeWidthHeader ? Number(nativeWidthHeader) : null,
    nativeHeight: nativeHeightHeader ? Number(nativeHeightHeader) : null,
  };
}

// ---------------------------------------------------------------------
// Phase 5 completion pass, item 6: Authentication — mirrors backend
// app/api/auth.py exactly (paths, status codes, request shapes).
// ---------------------------------------------------------------------

export function registerAccount(email: string, password: string): Promise<TokenResponse> {
  return request<TokenResponse>("/auth/register", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return request<TokenResponse>("/auth/login", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

/** Also doubles as "is my stored token still valid" — `AuthProvider`
 * calls this once on mount. A 401 here is handled the same centralized
 * way as any other request (see `request()` above), so a stale token
 * from a previous session gets cleared automatically on load. */
export function getMe(): Promise<UserOut> {
  return request<UserOut>("/auth/me");
}

// ---------------------------------------------------------------------
// Phase 5 completion pass, item 3.3: ingestion job status/retry —
// mirrors backend app/api/ingestion_jobs.py / app/schemas/ingestion.py.
// ---------------------------------------------------------------------

/** Poll immediately after upload, before a `document_id` exists yet —
 * see `UploadJobRef` / `app/api/ingestion_jobs.py`'s docstring for why
 * two separate polling endpoints exist. */
export function getJobStatus(jobId: number): Promise<IngestionJobOut> {
  return request<IngestionJobOut>(`/documents/jobs/${jobId}`);
}

export function getIngestionStatus(documentId: number): Promise<DocumentIngestionStatusOut> {
  return request<DocumentIngestionStatusOut>(`/documents/${documentId}/ingestion`);
}

/** Queues a fresh ingestion attempt and returns as soon as it's QUEUED
 * — does not wait for the outcome. Poll `getIngestionStatus` afterward,
 * same as after upload. */
export function reingestDocument(documentId: number): Promise<ReingestResponse> {
  return request<ReingestResponse>(`/documents/${documentId}/reingest`, { method: "POST" });
}

/** Permanently deletes a document — see backend
 * `app/api/documents.py::delete_document`'s docstring: only a document
 * the CALLER owns can be deleted (never shared corpus content, never
 * another user's upload — the backend returns 404 for either, which
 * `request()` surfaces as a normal `ApiError`, same as any other 404).
 * Requires an authenticated session (`getStoredToken()` must be set) —
 * calling this while signed out will 401. */
export function deleteDocument(documentId: number): Promise<DeleteDocumentResponse> {
  return request<DeleteDocumentResponse>(`/documents/${documentId}`, { method: "DELETE" });
}
// ---------------------------------------------------------------------
// Phase 6: multi-document reasoning — mirrors backend app/api/reasoning.py.
// Every function here follows the exact same request() pattern as
// ask()/uploadDocuments() above: no page/component builds its own fetch
// call or attaches its own Authorization header.
// ---------------------------------------------------------------------

export function compareDocuments(req: CompareRequest): Promise<CompareResponse> {
  return request<CompareResponse>("/compare", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function extractStructured(req: ExtractRequest): Promise<ExtractResponse> {
  return request<ExtractResponse>("/extract", {
    method: "POST",
    headers: { "content-type": "application/json" },
    // Backend app/schemas/reasoning.py::ExtractRequest uses a `schema`
    // alias for the (Python-reserved-adjacent) `schema_` field —
    // request() sends exactly what's given, so the wire shape here
    // must already use the alias name, not the TS property name.
    body: JSON.stringify({ document_ids: req.document_ids, schema: req.schema }),
  });
}

export function summarizeDocuments(req: SummarizeRequest): Promise<SummarizeResponse> {
  return request<SummarizeResponse>("/summarize", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function generateReport(req: ReportRequest): Promise<ReportResponse> {
  return request<ReportResponse>("/report", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------
// Phase 6: conversation memory — mirrors backend app/api/conversations.py.
// ---------------------------------------------------------------------

function anonQuery(anonymousSessionId?: string | null): string {
  return anonymousSessionId ? `?anonymous_session_id=${encodeURIComponent(anonymousSessionId)}` : "";
}

export function createConversation(req: ConversationCreate): Promise<ConversationOut> {
  return request<ConversationOut>("/conversations", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function listConversations(anonymousSessionId?: string | null): Promise<ConversationListResponse> {
  return request<ConversationListResponse>(`/conversations${anonQuery(anonymousSessionId)}`);
}

export function getConversation(conversationId: number, anonymousSessionId?: string | null): Promise<ConversationOut> {
  return request<ConversationOut>(`/conversations/${conversationId}${anonQuery(anonymousSessionId)}`);
}

export function getConversationMessages(
  conversationId: number,
  anonymousSessionId?: string | null,
): Promise<ConversationMessageOut[]> {
  return request<ConversationMessageOut[]>(`/conversations/${conversationId}/messages${anonQuery(anonymousSessionId)}`);
}

export function deleteConversation(conversationId: number, anonymousSessionId?: string | null): Promise<void> {
  return request<void>(`/conversations/${conversationId}${anonQuery(anonymousSessionId)}`, { method: "DELETE" });
}

export function sendConversationMessage(
  conversationId: number,
  req: ConversationTurnRequest,
): Promise<ConversationTurnResponse> {
  return request<ConversationTurnResponse>(`/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------
// Anonymous conversation session persistence (localStorage) — a
// conversation's anonymous_session_id must survive a page reload for
// "continue this conversation later" to work at all. Mirrors
// lib/auth-storage.ts's token-storage pattern (a plain localStorage
// key, not a cookie — same tradeoff, see that file).
// ---------------------------------------------------------------------

const ANON_SESSION_STORAGE_KEY = "veridoc.conversations.anonymous_session_id";

export function getStoredAnonymousSessionId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(ANON_SESSION_STORAGE_KEY);
}

export function setStoredAnonymousSessionId(id: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(ANON_SESSION_STORAGE_KEY, id);
}