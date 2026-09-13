"use client";

import { FileText } from "lucide-react";
import type { EvidenceCitation } from "@/lib/types";
import { formatPageRange } from "@/lib/format";

/** A small citation chip for Phase 6's `EvidenceCitation` shape
 * (chunk_id/source_file/page_start/page_end/document_id — see
 * backend app/schemas/reasoning.py's docstring on why this is
 * deliberately leaner than the full `Citation` type `EvidenceStamp`
 * renders: no score/section/snippet exist for this access pattern, so
 * this component doesn't fabricate placeholder values for them.
 */
export default function EvidenceCitationChip({ citation }: { citation: EvidenceCitation }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-sm border border-ink-600 bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-muted">
      <FileText size={10} />
      {citation.source_file}
      {citation.page_start != null && ` · ${formatPageRange(citation.page_start, citation.page_end)}`}
    </span>
  );
}