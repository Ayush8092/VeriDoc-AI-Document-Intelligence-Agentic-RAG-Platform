"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import clsx from "clsx";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Download,
  Loader2,
  AlertTriangle,
  FileText,
  Table2,
  Image as ImageIcon,
  BarChart3,
  ScanText,
  Library,
} from "lucide-react";

import {
  getPageImage,
  getPageObjects,
  listDocuments,
  locateSource,
  locateSourceById,
  sourceFileUrl,
  ApiError,
} from "@/lib/api";
import { loadCitationsForViewer } from "@/lib/citation-viewer-link";
import { coordinateSpaceLabel } from "@/lib/bbox";
import { formatConfidence, truncate } from "@/lib/format";
import type { Citation, CoordinateSpace, DocumentOut, PageObject, SourceDocumentRef } from "@/lib/types";
import ConfidenceMeter from "@/components/confidence-meter";
import TablePreview from "@/components/table-preview";
import BBoxOverlay, { type HighlightBox } from "@/components/bbox-overlay";

const BLOCK_ICON: Record<string, typeof FileText> = {
  text: FileText,
  table: Table2,
  figure: ImageIcon,
  visual: ImageIcon,
  chart: BarChart3,
};

/**
 * Phase 4.1(7): the Source Viewer.
 *
 *   Answer -> Citation -> click -> Source Viewer -> Document -> Page
 *   -> exact object -> bounding-box highlight
 *
 * Three entry points, one page:
 *   ?session=TOKEN            citation click-through (EvidenceStamp)
 *   ?document=ID&page=N       direct deep link / document picker
 *   (none)                    document picker
 *
 * Coordinate spaces (PDF_POINTS / IMAGE_PIXELS) are never touched here —
 * see lib/bbox.ts's `projectBBox`, the one place allowed to do that
 * projection, and `lib/bbox.ts`'s docstring on why an absent highlight
 * beats a guessed one.
 */
export default function SourceViewerPage() {
  return (
    <Suspense fallback={<ViewerShell><LoadingState label="Loading viewer…" /></ViewerShell>}>
      <SourceViewerInner />
    </Suspense>
  );
}

function SourceViewerInner() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const sessionToken = searchParams.get("session");
  const documentParam = searchParams.get("document");
  const pageParam = searchParams.get("page");
  const objectParam = searchParams.get("object");

  // Citation-session state (only populated in ?session= mode).
  const [citations, setCitations] = useState<Citation[] | null>(null);
  const [citationIndex, setCitationIndex] = useState(0);

  const [document, setDocument] = useState<SourceDocumentRef | null>(null);
  const [renderable, setRenderable] = useState<boolean | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);

  const [pageImage, setPageImage] = useState<{ url: string; width: number; height: number; nativeWidth: number | null; nativeHeight: number | null } | null>(null);
  const [pageObjects, setPageObjects] = useState<PageObject[] | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(objectParam);

  const [loading, setLoading] = useState(true);
  const [imageLoading, setImageLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeCitation = sessionToken && citations ? citations[citationIndex] : null;

  // ---- resolve the target document/page from the URL ------------------
  useEffect(() => {
    let cancelled = false;

    async function resolve() {
      setLoading(true);
      setError(null);
      try {
        if (sessionToken) {
          const loaded = loadCitationsForViewer(sessionToken);
          if (!loaded || loaded.citations.length === 0) {
            if (!cancelled) {
              setError("This citation link has expired or is invalid. Return to Ask and click the citation again.");
              setLoading(false);
            }
            return;
          }
          if (cancelled) return;
          setCitations(loaded.citations);
          setCitationIndex(Math.min(Math.max(0, loaded.index), loaded.citations.length - 1));
          return; // the citationIndex effect below does the actual document/page fetch
        }

        if (documentParam) {
          const id = Number(documentParam);
          const requestedPage = pageParam ? Number(pageParam) : 1;
          const located = await locateSourceById(id, Number.isFinite(requestedPage) ? requestedPage : 1);
          if (cancelled) return;
          setDocument(located.document);
          setRenderable(located.capabilities.renderable);
          setPageCount(located.capabilities.page_count);
          setPageNumber(located.page_number);
          setLoading(false);
          return;
        }

        // No target specified at all -> document picker.
        setDocument(null);
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Could not load the source viewer.");
        setLoading(false);
      }
    }

    resolve();
    return () => {
      cancelled = true;
    };
  }, [sessionToken, documentParam, pageParam]);

  // ---- citation-session mode: (re)resolve document/page whenever the
  // active citation changes (prev/next evidence can jump documents) -----
  useEffect(() => {
    if (!sessionToken || !activeCitation) return;
    let cancelled = false;

    async function resolveCitation() {
      setLoading(true);
      setError(null);
      try {
        const located = await locateSource(activeCitation!.source_file, activeCitation!.page_start ?? 1);
        if (cancelled) return;
        setDocument(located.document);
        setRenderable(located.capabilities.renderable);
        setPageCount(located.capabilities.page_count);
        setPageNumber(located.page_number);
        setSelectedObjectId(activeCitation!.object_id);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.message
            : `Could not locate "${activeCitation!.source_file}" — it may not be indexed yet.`,
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    resolveCitation();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionToken, citationIndex, activeCitation?.chunk_id]);

  // ---- load the rendered page image + the objects on this page --------
  useEffect(() => {
    let cancelled = false;

    // Inline async function (see components/auth-provider.tsx's fix for
    // why: `react-hooks/set-state-in-effect` accepts a synchronous
    // `setState` reachable from a function defined AND invoked within
    // the same effect — including on an early-exit guard branch — but
    // not a bare top-level `setState` in the effect body itself, which
    // is what the guard clause used to be before this fix.
    async function run() {
      if (!document || renderable !== true) {
        setPageImage(null);
        return;
      }
      setImageLoading(true);
      try {
        const img = await getPageImage(document.id, pageNumber);
        if (cancelled) return;
        setPageImage((prev) => {
          if (prev) URL.revokeObjectURL(prev.url);
          return img;
        });
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Could not render this page.");
      } finally {
        if (!cancelled) setImageLoading(false);
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [document, pageNumber, renderable]);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      if (!document) {
        setPageObjects(null);
        return;
      }
      try {
        const res = await getPageObjects(document.id, pageNumber);
        if (!cancelled) setPageObjects(res.objects);
      } catch {
        if (!cancelled) setPageObjects([]); // best-effort panel; a failure here shouldn't block the page image
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [document, pageNumber]);

  // release the blob URL on unmount
  useEffect(() => {
    return () => {
      setPageImage((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return prev;
      });
    };
  }, []);

  const goToPage = useCallback(
    (next: number) => {
      if (!document || pageCount <= 0) return;
      const clamped = Math.min(Math.max(1, next), pageCount);
      setPageNumber(clamped);
      setSelectedObjectId(null);
      if (documentParam) {
        router.replace(`/source?document=${document.id}&page=${clamped}`);
      }
    },
    [document, pageCount, documentParam, router],
  );

  function goToEvidence(delta: number) {
    if (!citations) return;
    const next = citationIndex + delta;
    if (next < 0 || next >= citations.length) return;
    setCitationIndex(next);
  }

  function pickDocument(doc: DocumentOut) {
    router.push(`/source?document=${doc.id}&page=1`);
  }

  // ---- render -----------------------------------------------------------

  if (!sessionToken && !documentParam) {
    return (
      <ViewerShell>
        <DocumentPicker onPick={pickDocument} />
      </ViewerShell>
    );
  }

  if (loading && !document) {
    return (
      <ViewerShell>
        <LoadingState label="Locating source…" />
      </ViewerShell>
    );
  }

  if (error && !document) {
    return (
      <ViewerShell>
        <ErrorState message={error} />
      </ViewerShell>
    );
  }

  if (!document) {
    return (
      <ViewerShell>
        <ErrorState message="Nothing to show yet." />
      </ViewerShell>
    );
  }

  const activeBbox = activeCitation?.bbox && activeCitation.page_start === pageNumber ? activeCitation.bbox : null;
  const selectedFromPanel = pageObjects?.find((o) => o.object_id === selectedObjectId && o !== null) ?? null;
  const detailObject: DetailTarget | null = activeBbox
    ? citationToDetail(activeCitation!)
    : selectedFromPanel
      ? pageObjectToDetail(selectedFromPanel)
      : null;

  const highlightBoxes: HighlightBox[] = (pageObjects ?? [])
    .filter((o) => o.bbox)
    .map((o) => ({
      key: o.chunk_id,
      bbox: o.bbox!,
      active: o.object_id != null && o.object_id === (detailObject?.object_id ?? selectedObjectId),
      label: o.caption || o.section || o.block_type,
      onClick: () => setSelectedObjectId(o.object_id),
    }));
  // The active citation's own bbox always wins as the highlighted box,
  // even if (e.g. Phase 4 not yet re-run) it doesn't show up in the
  // page-objects listing for some reason — the citation is ground truth
  // for "what was actually cited", the objects panel is supplementary
  // context.
  if (activeBbox && !highlightBoxes.some((b) => b.active)) {
    highlightBoxes.push({
      key: `citation-${activeCitation!.chunk_id}`,
      bbox: activeBbox,
      active: true,
      label: activeCitation!.caption || activeCitation!.section,
    });
  }

  return (
    <ViewerShell>
      <div className="flex h-full flex-col">
        <ViewerHeader
          document={document}
          onBack={() => router.push(sessionToken ? "/ask" : "/source")}
        />

        {error && (
          <div className="mx-6 mt-4 flex items-center gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
            <AlertTriangle size={16} className="shrink-0 text-rust" />
            {error}
          </div>
        )}

        <div className="flex min-h-0 flex-1 gap-0">
          <div className="flex min-w-0 flex-1 flex-col">
            {renderable === false ? (
              <NonRenderableNotice document={document} citation={activeCitation} />
            ) : (
              <>
                <PageNav
                  pageNumber={pageNumber}
                  pageCount={pageCount}
                  onChange={goToPage}
                  onPageOnCitation={
                    activeCitation?.page_start ? () => goToPage(activeCitation.page_start!) : undefined
                  }
                />
                <div className="relative flex-1 overflow-auto bg-ink-950 px-6 py-6">
                  {imageLoading && (
                    <div className="absolute inset-0 z-20 flex items-center justify-center bg-ink-950/60">
                      <Loader2 size={20} className="animate-spin text-muted" />
                    </div>
                  )}
                  {pageImage && (
                    <div className="relative mx-auto" style={{ width: pageImage.width, maxWidth: "100%" }}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={pageImage.url}
                        alt={`${document.filename}, page ${pageNumber}`}
                        width={pageImage.width}
                        height={pageImage.height}
                        className="w-full rounded-md border border-ink-700 shadow-lg"
                      />
                      <BBoxOverlay
                        boxes={highlightBoxes}
                        context={{
                          renderedWidth: pageImage.width,
                          renderedHeight: pageImage.height,
                          nativeWidth: pageImage.nativeWidth,
                          nativeHeight: pageImage.nativeHeight,
                        }}
                      />
                    </div>
                  )}
                </div>
              </>
            )}
          </div>

          <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-ink-700 bg-ink-900">
            {sessionToken && citations && (
              <EvidenceNav
                index={citationIndex}
                total={citations.length}
                onPrev={() => goToEvidence(-1)}
                onNext={() => goToEvidence(1)}
              />
            )}

            {detailObject && <ObjectDetailPanel target={detailObject} />}

            <PageObjectsPanel
              objects={pageObjects}
              selectedObjectId={detailObject?.object_id ?? selectedObjectId}
              onSelect={(o) => setSelectedObjectId(o.object_id)}
            />

            <div className="mt-auto border-t border-ink-700 px-4 py-3">
              <a
                href={sourceFileUrl(document.id)}
                download={document.filename}
                className="flex items-center justify-center gap-1.5 rounded-md border border-ink-600 py-2 text-[12px] text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
              >
                <Download size={13} />
                Download original ({document.file_type})
              </a>
            </div>
          </aside>
        </div>
      </div>
    </ViewerShell>
  );
}

// ---------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------

function ViewerShell({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full flex-col bg-ink-950">{children}</div>;
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex flex-1 items-center justify-center gap-2 text-[13px] text-muted">
      <Loader2 size={16} className="animate-spin" />
      {label}
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
      <AlertTriangle size={22} className="text-rust" />
      <p className="max-w-sm text-[13px] text-paper-dim">{message}</p>
      <Link
        href="/source"
        className="mt-1 flex items-center gap-1.5 rounded-md border border-ink-600 px-3 py-1.5 text-[12px] text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
      >
        <Library size={13} />
        Browse documents
      </Link>
    </div>
  );
}

function ViewerHeader({ document, onBack }: { document: SourceDocumentRef; onBack: () => void }) {
  return (
    <div className="flex items-center gap-3 border-b border-ink-700 px-6 py-4">
      <button
        onClick={onBack}
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-ink-600 text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
        aria-label="Back"
      >
        <ArrowLeft size={15} />
      </button>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <h1 className="truncate text-[15px] font-medium text-paper">{document.filename}</h1>
          <span className="shrink-0 rounded-sm bg-ink-700 px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-paper-dim">
            {document.source_root}
          </span>
          {document.has_scanned_pages && (
            <span className="flex shrink-0 items-center gap-1 rounded-sm bg-teal-dim px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-teal">
              <ScanText size={10} />
              OCR
            </span>
          )}
        </div>
        <p className="mt-0.5 font-mono text-[11px] text-muted">
          {document.file_type.toUpperCase()} · {document.page_count} {document.page_count === 1 ? "page" : "pages"}
        </p>
      </div>
    </div>
  );
}

function PageNav({
  pageNumber,
  pageCount,
  onChange,
  onPageOnCitation,
}: {
  pageNumber: number;
  pageCount: number;
  onChange: (page: number) => void;
  onPageOnCitation?: () => void;
}) {
  return (
    <div className="flex items-center justify-center gap-3 border-b border-ink-700 bg-ink-900 px-6 py-2.5">
      <button
        onClick={() => onChange(pageNumber - 1)}
        disabled={pageNumber <= 1}
        className="flex h-7 w-7 items-center justify-center rounded-md text-paper-dim transition-colors hover:bg-ink-700 hover:text-paper disabled:opacity-30"
        aria-label="Previous page"
      >
        <ChevronLeft size={15} />
      </button>
      <div className="flex items-center gap-1.5 font-mono text-[12px] text-paper-dim">
        <input
          type="number"
          value={pageNumber}
          min={1}
          max={pageCount || 1}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v)) onChange(v);
          }}
          className="w-12 rounded-sm border border-ink-600 bg-ink-800 px-1.5 py-0.5 text-center text-paper focus:border-brass-dim focus:outline-none"
        />
        <span>/ {pageCount || "—"}</span>
      </div>
      <button
        onClick={() => onChange(pageNumber + 1)}
        disabled={pageCount === 0 || pageNumber >= pageCount}
        className="flex h-7 w-7 items-center justify-center rounded-md text-paper-dim transition-colors hover:bg-ink-700 hover:text-paper disabled:opacity-30"
        aria-label="Next page"
      >
        <ChevronRight size={15} />
      </button>
      {onPageOnCitation && (
        <button
          onClick={onPageOnCitation}
          className="ml-2 rounded-sm border border-ink-600 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
        >
          Back to cited page
        </button>
      )}
    </div>
  );
}

function NonRenderableNotice({ document, citation }: { document: SourceDocumentRef; citation: Citation | null }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
      <FileText size={26} className="text-muted" />
      <p className="max-w-sm text-[13px] text-paper-dim">
        {document.filename} is a {document.file_type} document — text was extracted directly, so there is no page
        image to display. The cited passage is shown below instead.
      </p>
      {citation && (
        <div className="mt-2 max-w-md rounded-md border border-ink-600 bg-ink-800 px-4 py-3 text-left">
          <p className="font-mono text-[11px] uppercase tracking-wide text-muted">{citation.section}</p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-paper/90">{truncate(citation.snippet, 600)}</p>
        </div>
      )}
    </div>
  );
}

function EvidenceNav({
  index,
  total,
  onPrev,
  onNext,
}: {
  index: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  return (
    <div className="flex items-center justify-between border-b border-ink-700 px-4 py-3">
      <button
        onClick={onPrev}
        disabled={index <= 0}
        className="flex items-center gap-1 rounded-sm border border-ink-600 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-paper-dim transition-colors hover:border-brass-dim hover:text-paper disabled:opacity-30"
      >
        <ChevronLeft size={11} />
        Prev
      </button>
      <span className="font-mono text-[11px] text-muted">
        Evidence {index + 1} / {total}
      </span>
      <button
        onClick={onNext}
        disabled={index >= total - 1}
        className="flex items-center gap-1 rounded-sm border border-ink-600 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-paper-dim transition-colors hover:border-brass-dim hover:text-paper disabled:opacity-30"
      >
        Next
        <ChevronRight size={11} />
      </button>
    </div>
  );
}

interface DetailTarget {
  object_id: string | null;
  block_type: string;
  caption: string | null;
  visual_type: string | null;
  chart_type: string | null;
  confidence: number;
  source: string;
  coordinate_space: CoordinateSpace | null;
  page: number | null;
  chunk_id: string;
  snippet: string;
  table_rows: string[][] | null;
  related_text: string | null;
}

function citationToDetail(c: Citation): DetailTarget {
  return {
    object_id: c.object_id,
    block_type: c.block_type,
    caption: c.caption,
    visual_type: c.visual_type,
    chart_type: c.chart_type,
    confidence: c.confidence,
    source: c.source,
    coordinate_space: c.coordinate_space,
    page: c.page_start,
    chunk_id: c.chunk_id,
    snippet: c.snippet,
    table_rows: c.table_rows,
    related_text: c.related_text,
  };
}

function pageObjectToDetail(o: PageObject): DetailTarget {
  return {
    object_id: o.object_id,
    block_type: o.block_type,
    caption: o.caption,
    visual_type: o.visual_type,
    chart_type: o.chart_type,
    confidence: o.confidence,
    source: o.source,
    coordinate_space: o.coordinate_space,
    page: o.page_start,
    chunk_id: o.chunk_id,
    snippet: o.snippet,
    table_rows: o.table_rows,
    related_text: o.related_text,
  };
}

function ObjectDetailPanel({ target }: { target: DetailTarget }) {
  const Icon = BLOCK_ICON[target.block_type] ?? FileText;
  return (
    <div className="border-b border-ink-700 px-4 py-3.5">
      <div className="flex items-center gap-2">
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-brass-dim text-brass-soft">
          <Icon size={13} />
        </div>
        <span className="font-mono text-[11px] uppercase tracking-wide text-paper-dim">{target.block_type}</span>
        {target.visual_type && target.visual_type !== "unknown_visual" && (
          <span className="rounded-sm bg-ink-700 px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-paper-dim">
            {target.visual_type}
          </span>
        )}
        {target.chart_type && target.chart_type !== "unknown_chart" && (
          <span className="rounded-sm bg-ink-700 px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-paper-dim">
            {target.chart_type} chart
          </span>
        )}
      </div>

      {target.caption && (
        <p className="mt-2.5 text-[13px] leading-relaxed text-paper/90">
          <span className="text-muted">Caption: </span>
          {target.caption}
        </p>
      )}

      {target.table_rows ? (
        <div className="mt-2.5">
          <TablePreview rows={target.table_rows} />
        </div>
      ) : (
        <p className="mt-2.5 text-[13px] leading-relaxed text-paper/80">{truncate(target.snippet, 400)}</p>
      )}

      {target.related_text && !target.table_rows && (
        <p className="mt-2 text-[12px] leading-relaxed text-paper-dim">
          <span className="text-muted">Nearby text: </span>
          {truncate(target.related_text, 220)}
        </p>
      )}

      <div className="mt-3 flex flex-col gap-1.5 border-t border-ink-700 pt-2.5 font-mono text-[10px] text-muted">
        <div className="flex items-center justify-between">
          <span>Confidence</span>
          <ConfidenceMeter confidence={target.confidence} source={target.source === "ocr" ? "ocr" : "native"} />
        </div>
        <div className="flex items-center justify-between">
          <span>Source</span>
          <span className="text-paper-dim">{target.source === "ocr" ? "OCR" : "Native extraction"}</span>
        </div>
        {target.coordinate_space && (
          <div className="flex items-center justify-between">
            <span>Coordinate space</span>
            <span className="text-paper-dim">{coordinateSpaceLabel(target.coordinate_space)}</span>
          </div>
        )}
        {target.object_id && (
          <div className="flex items-center justify-between gap-2">
            <span>Object ID</span>
            <span className="truncate text-paper-dim" title={target.object_id}>
              {target.object_id}
            </span>
          </div>
        )}
        <div className="flex items-center justify-between gap-2">
          <span>Chunk</span>
          <span className="truncate text-paper-dim" title={target.chunk_id}>
            {target.chunk_id}
          </span>
        </div>
      </div>
    </div>
  );
}

function PageObjectsPanel({
  objects,
  selectedObjectId,
  onSelect,
}: {
  objects: PageObject[] | null;
  selectedObjectId: string | null;
  onSelect: (o: PageObject) => void;
}) {
  return (
    <div className="flex-1 px-4 py-3.5">
      <p className="mb-2 font-mono text-[11px] uppercase tracking-wide text-muted">On this page</p>
      {objects === null ? (
        <div className="flex items-center gap-2 py-4 text-[12px] text-muted">
          <Loader2 size={12} className="animate-spin" />
          Loading objects…
        </div>
      ) : objects.length === 0 ? (
        <p className="py-2 text-[12px] text-muted">Nothing indexed for this page.</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {objects.map((o) => {
            const Icon = BLOCK_ICON[o.block_type] ?? FileText;
            const active = o.object_id != null && o.object_id === selectedObjectId;
            return (
              <button
                key={o.chunk_id}
                onClick={() => onSelect(o)}
                className={clsx(
                  "flex items-start gap-2 rounded-md border px-2.5 py-2 text-left transition-colors",
                  active
                    ? "border-brass-dim bg-brass-dim/15"
                    : "border-ink-700 hover:border-ink-600 hover:bg-ink-800",
                )}
              >
                <Icon size={13} className={clsx("mt-0.5 shrink-0", active ? "text-brass-soft" : "text-muted")} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[12px] text-paper-dim">
                    {o.caption || o.section || o.block_type}
                  </p>
                  <p className="font-mono text-[10px] text-muted">{formatConfidence(o.confidence)} confidence</p>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function DocumentPicker({ onPick }: { onPick: (doc: DocumentOut) => void }) {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then((res) => setDocuments(res.documents))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Could not load documents."));
  }, []);

  const renderableDocs = useMemo(
    () => (documents ?? []).filter((d) => ["pdf", "png", "jpg", "jpeg"].includes(d.file_type)),
    [documents],
  );
  const textDocs = useMemo(
    () => (documents ?? []).filter((d) => !["pdf", "png", "jpg", "jpeg"].includes(d.file_type)),
    [documents],
  );

  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-10">
      <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
        Source Viewer
      </h1>
      <p className="mt-1 text-[13px] text-paper-dim">
        Pick a document to open, or click &ldquo;View source&rdquo; on any citation in Ask.
      </p>

      {error && (
        <div className="mt-6 flex items-center gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <AlertTriangle size={16} className="shrink-0 text-rust" />
          {error}
        </div>
      )}

      {documents === null && !error && <LoadingState label="Loading documents…" />}

      {documents !== null && documents.length === 0 && (
        <div className="mt-6 rounded-lg border border-dashed border-ink-600 px-6 py-10 text-center text-[13px] text-muted">
          No documents ingested yet.
        </div>
      )}

      {renderableDocs.length > 0 && (
        <DocumentGroup title="Page-renderable" documents={renderableDocs} onPick={onPick} />
      )}
      {textDocs.length > 0 && (
        <DocumentGroup title="Text-extracted (no page image)" documents={textDocs} onPick={onPick} />
      )}
    </div>
  );
}

function DocumentGroup({
  title,
  documents,
  onPick,
}: {
  title: string;
  documents: DocumentOut[];
  onPick: (doc: DocumentOut) => void;
}) {
  return (
    <div className="mt-6">
      <p className="mb-2 font-mono text-[11px] uppercase tracking-wide text-muted">{title}</p>
      <div className="flex flex-col gap-2">
        {documents.map((doc) => (
          <button
            key={doc.id}
            onClick={() => onPick(doc)}
            className="flex items-center gap-3 rounded-lg border border-ink-600 bg-ink-800 px-4 py-3 text-left transition-colors hover:border-brass-dim"
          >
            <FileText size={15} className="shrink-0 text-brass-soft" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] text-paper">{doc.title || doc.filename}</p>
              <p className="truncate font-mono text-[11px] text-muted">{doc.filename}</p>
            </div>
            <span className="shrink-0 font-mono text-[11px] text-muted">
              {doc.current_version?.page_count ?? 0} pp
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
