"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import { FileText, Table2, ScanText, ChevronDown, Image as ImageIcon, BarChart3, ExternalLink } from "lucide-react";
import type { Citation } from "@/lib/types";
import { formatPageRange, truncate } from "@/lib/format";
import ConfidenceMeter from "./confidence-meter";
import TablePreview from "./table-preview";
import { storeCitationsForViewer } from "@/lib/citation-viewer-link";

const BLOCK_LABEL: Record<Citation["block_type"], string> = {
  text: "",
  table: "Table",
  figure: "Figure",
  chart: "Chart",
  visual: "Visual",
};

function blockIcon(blockType: Citation["block_type"]) {
  switch (blockType) {
    case "table":
      return <Table2 size={13} />;
    case "figure":
    case "visual":
      return <ImageIcon size={13} />;
    case "chart":
      return <BarChart3 size={13} />;
    default:
      return <FileText size={13} />;
  }
}

/**
 * The product's signature visual: every citation is a small stamped
 * "evidence ticket" rather than a generic pill — a perforated left edge,
 * a page/section stamp, and a confidence meter that reads differently for
 * native extraction (brass, always 100%) vs. OCR (teal, computed). This
 * is deliberate: the whole point of the product is that an answer's
 * provenance is checkable, so the citation itself should look like a
 * ledger entry you'd file, not a chat "source" chip.
 *
 * Phase 4.1(5): extended to cover figure/chart/visual citations (not
 * just text/table) — a distinct icon/badge per `block_type`, a caption
 * line when the backend found one, a chart-type badge, and a
 * "View source" click-through into the Source Viewer
 * (frontend/app/source/[documentId]/page.tsx) that carries this
 * citation's object_id/bbox/coordinate_space so the viewer can
 * highlight the EXACT supporting region, not just open the page.
 */
export default function EvidenceStamp({
  citation,
  index,
  allCitations,
}: {
  citation: Citation;
  index: number;
  // Full citation list for this answer, so "View source" can offer
  // previous/next-evidence navigation across every citation in the
  // same answer, not just this one in isolation.
  allCitations?: Citation[];
}) {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const isTable = citation.block_type === "table";
  const isVisual = citation.block_type === "figure" || citation.block_type === "chart" || citation.block_type === "visual";
  const isOcr = citation.source === "ocr";
  const blockLabel = BLOCK_LABEL[citation.block_type];

  function viewSource() {
    const citations = allCitations && allCitations.length ? allCitations : [citation];
    const idx = citations.findIndex((c) => c.chunk_id === citation.chunk_id);
    const token = storeCitationsForViewer(citations, idx >= 0 ? idx : 0);
    router.push(`/source?session=${token}`);
  }

  return (
    <div
      className="animate-stamp-in relative overflow-hidden rounded-md border border-ink-600 bg-ink-800 text-left"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {/* perforation edge */}
      <div
        className="absolute top-0 bottom-0 left-0 w-[3px]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(to bottom, var(--color-ink-950) 0 4px, transparent 4px 8px)",
        }}
        aria-hidden
      />
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 py-2.5 pr-3 pl-4 text-left transition-colors hover:bg-ink-700/60"
      >
        <div
          className={clsx(
            "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
            isTable || isVisual ? "bg-brass-dim text-brass-soft" : isOcr ? "bg-teal-dim text-teal" : "bg-ink-600 text-paper-dim",
          )}
        >
          {isOcr && !isTable && !isVisual ? <ScanText size={13} /> : blockIcon(citation.block_type)}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="truncate font-mono text-[12px] font-medium text-paper">
              {citation.source_file}
            </span>
            <span className="font-mono text-[11px] text-muted">
              {formatPageRange(citation.page_start, citation.page_end)}
            </span>
            {isOcr && (
              <span className="rounded-sm bg-teal-dim px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-teal">
                OCR
              </span>
            )}
            {blockLabel && (
              <span className="rounded-sm bg-brass-dim px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-brass-soft">
                {blockLabel}
              </span>
            )}
            {citation.visual_type && citation.visual_type !== "unknown_visual" && (
              <span className="rounded-sm bg-ink-600 px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-paper-dim">
                {citation.visual_type}
              </span>
            )}
            {citation.chart_type && citation.chart_type !== "unknown_chart" && (
              <span className="rounded-sm bg-ink-600 px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-paper-dim">
                {citation.chart_type} chart
              </span>
            )}
          </div>
          <p className="mt-0.5 truncate text-[12px] text-paper-dim">
            {citation.caption || citation.section}
          </p>
        </div>

        <ConfidenceMeter confidence={citation.confidence} source={citation.source} className="mt-0.5 shrink-0" />
        <ChevronDown
          size={14}
          className={clsx("mt-1.5 shrink-0 text-muted transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <div className="border-t border-ink-600 bg-ink-900/60 px-4 py-3">
          {isTable && citation.table_rows ? (
            <TablePreview rows={citation.table_rows} />
          ) : isVisual ? (
            <div className="flex flex-col gap-1.5">
              {citation.caption && (
                <p className="text-[13px] leading-relaxed text-paper/90">
                  <span className="text-muted">Caption: </span>
                  {citation.caption}
                </p>
              )}
              {citation.related_text && (
                <p className="text-[13px] leading-relaxed text-paper/80">{truncate(citation.related_text, 500)}</p>
              )}
              {!citation.caption && !citation.related_text && (
                <p className="text-[13px] leading-relaxed text-paper/70 italic">
                  No caption or extracted text was associated with this {citation.block_type}.
                </p>
              )}
            </div>
          ) : (
            <p className="text-[13px] leading-relaxed text-paper/90">{truncate(citation.snippet, 500)}</p>
          )}

          <div className="mt-2 flex items-center justify-between gap-2">
            <p className="font-mono text-[10px] text-muted">
              chunk {citation.chunk_id} · relevance {(citation.score * 100).toFixed(0)}%
              {citation.object_id ? ` · object ${citation.object_id}` : ""}
            </p>
            {citation.page_start != null && (
              <button
                type="button"
                onClick={viewSource}
                className="flex shrink-0 items-center gap-1 rounded-sm border border-ink-600 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
              >
                <ExternalLink size={11} />
                View source
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}