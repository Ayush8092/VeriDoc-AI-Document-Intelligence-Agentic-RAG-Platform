"use client";

import { useState } from "react";
import { FileText, Loader2, XCircle, Lightbulb } from "lucide-react";
import { generateReport, ApiError } from "@/lib/api";
import type { ReportResponse } from "@/lib/types";
import DocumentPicker from "@/components/document-picker";
import EvidenceCitationChip from "@/components/evidence-citation-chip";

export default function ReportPage() {
  const [documentIds, setDocumentIds] = useState<number[]>([]);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReportResponse | null>(null);

  async function submit() {
    if (documentIds.length < 1 || !title.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await generateReport({ title: title.trim(), document_ids: documentIds });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Report generation failed unexpectedly.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-6 overflow-y-auto px-6 py-8">
      <div>
        <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
          Generate a report
        </h1>
        <p className="mt-1 text-[13px] text-paper-dim">
          Executive summary, findings, comparison, and risks — all grounded. Recommendations are labeled as inference, never mistaken for evidence.
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-[12px] uppercase tracking-wide text-muted">Report title</label>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Vendor contract risk review"
          className="rounded-lg border border-ink-600 bg-ink-800 px-3 py-2 text-[14px] text-paper placeholder:text-muted focus:border-brass-dim focus:outline-none"
        />
      </div>

      <div className="flex flex-col gap-3">
        <label className="text-[12px] uppercase tracking-wide text-muted">Documents</label>
        <DocumentPicker selected={documentIds} onChange={setDocumentIds} min={1} />
      </div>

      <button
        onClick={submit}
        disabled={busy || documentIds.length < 1 || !title.trim()}
        className="flex items-center justify-center gap-2 rounded-lg bg-brass px-4 py-2.5 text-[14px] font-medium text-ink-950 transition-opacity disabled:opacity-30"
      >
        {busy ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} />}
        Generate report
      </button>

      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <XCircle size={16} className="mt-0.5 shrink-0 text-rust" />
          {error}
        </div>
      )}

      {result && (
        <article className="flex flex-col gap-6 rounded-lg border border-ink-600 bg-ink-800 p-5">
          <div>
            <h2 className="text-lg text-paper" style={{ fontFamily: "var(--font-display)" }}>
              {result.title}
            </h2>
            <p className="mt-1 text-[12px] text-muted">{result.documents.map((d) => d.title).join(", ")}</p>
          </div>

          {result.executive_summary && (
            <section>
              <h3 className="text-[12px] uppercase tracking-wide text-muted">Executive summary</h3>
              <p className="mt-1.5 text-[14px] leading-relaxed text-paper">{result.executive_summary}</p>
            </section>
          )}

          {result.key_findings.length > 0 && (
            <section>
              <h3 className="text-[12px] uppercase tracking-wide text-muted">Key findings</h3>
              <ul className="mt-1.5 flex flex-col gap-1.5">
                {result.key_findings.map((f, i) => (
                  <li key={i} className="text-[13px] text-paper-dim">
                    <span className="mr-1.5 text-muted">·</span>
                    {f.text}
                    <span className="ml-1.5 inline-flex flex-wrap gap-1">
                      {f.citations.map((c) => (
                        <EvidenceCitationChip key={c.chunk_id} citation={c} />
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {result.risks.length > 0 && (
            <section>
              <h3 className="text-[12px] uppercase tracking-wide text-muted">Risks</h3>
              <ul className="mt-1.5 flex flex-col gap-1.5">
                {result.risks.map((r, i) => (
                  <li key={i} className="text-[13px] text-paper-dim">
                    <span className="mr-1.5 text-rust">·</span>
                    {r.text}
                    <span className="ml-1.5 inline-flex flex-wrap gap-1">
                      {r.citations.map((c) => (
                        <EvidenceCitationChip key={c.chunk_id} citation={c} />
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {result.recommendations.length > 0 && (
            <section>
              <h3 className="text-[12px] uppercase tracking-wide text-muted">Recommendations</h3>
              <ul className="mt-1.5 flex flex-col gap-2">
                {result.recommendations.map((r, i) => (
                  <li key={i} className="rounded-lg border border-brass-dim/40 bg-brass-dim/10 px-3 py-2 text-[13px] text-paper-dim">
                    <div className="flex items-start gap-1.5">
                      <Lightbulb size={13} className="mt-0.5 shrink-0 text-brass" />
                      <div>
                        {r.text}
                        <span className="ml-1.5 inline-block rounded-sm bg-ink-700 px-1.5 py-[1px] font-mono text-[9px] uppercase tracking-wide text-muted">
                          inference — not a document statement
                        </span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <p className="text-[11px] text-muted">
            {result.appendix.total_documents} document(s), {result.appendix.total_chunks_considered} chunks considered.
          </p>
        </article>
      )}
    </div>
  );
}