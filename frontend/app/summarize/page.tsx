"use client";

import { useState } from "react";
import { Files, Loader2, XCircle } from "lucide-react";
import { summarizeDocuments, ApiError } from "@/lib/api";
import type { SummarizeResponse, SummaryMode } from "@/lib/types";
import DocumentPicker from "@/components/document-picker";
import EvidenceCitationChip from "@/components/evidence-citation-chip";

const MODES: { value: SummaryMode; label: string }[] = [
  { value: "document", label: "Full summary" },
  { value: "executive", label: "Executive summary" },
  { value: "key_decisions", label: "Key decisions" },
  { value: "key_risks", label: "Key risks" },
  { value: "action_items", label: "Action items" },
];

export default function SummarizePage() {
  const [documentIds, setDocumentIds] = useState<number[]>([]);
  const [mode, setMode] = useState<SummaryMode>("document");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SummarizeResponse | null>(null);

  async function submit() {
    if (documentIds.length < 1 || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await summarizeDocuments({ document_ids: documentIds, mode });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Summarization failed unexpectedly.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-6 overflow-y-auto px-6 py-8">
      <div>
        <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
          Summarize documents
        </h1>
        <p className="mt-1 text-[13px] text-paper-dim">
          Hierarchical, map-reduce summarization — every point traces back to real evidence.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <label className="text-[12px] uppercase tracking-wide text-muted">Documents</label>
        <DocumentPicker selected={documentIds} onChange={setDocumentIds} min={1} />
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-[12px] uppercase tracking-wide text-muted">Mode</label>
        <div className="flex flex-wrap gap-2">
          {MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => setMode(m.value)}
              className={`rounded-lg border px-3 py-1.5 text-[12px] ${
                mode === m.value
                  ? "border-brass-dim bg-brass-dim/20 text-brass"
                  : "border-ink-600 text-paper-dim hover:border-brass-dim hover:text-paper"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={submit}
        disabled={busy || documentIds.length < 1}
        className="flex items-center justify-center gap-2 rounded-lg bg-brass px-4 py-2.5 text-[14px] font-medium text-ink-950 transition-opacity disabled:opacity-30"
      >
        {busy ? <Loader2 size={16} className="animate-spin" /> : <Files size={16} />}
        Summarize
      </button>

      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <XCircle size={16} className="mt-0.5 shrink-0 text-rust" />
          {error}
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-5">
          {result.combined_summary && (
            <div>
              <p className="text-[12px] uppercase tracking-wide text-muted">Combined summary</p>
              <p className="mt-1 text-[14px] leading-relaxed text-paper">{result.combined_summary}</p>
              <PointList points={result.combined_points} />
            </div>
          )}

          {Object.entries(result.per_document).map(([docId, doc]) => {
            const ref = result.documents.find((d) => String(d.id) === docId);
            return (
              <div key={docId} className="rounded-lg border border-ink-600 bg-ink-800 p-4">
                <p className="text-[13px] font-medium text-paper">{ref?.title ?? `Document ${docId}`}</p>
                <p className="mt-1 text-[13px] leading-relaxed text-paper-dim">{doc.summary || "No summary produced."}</p>
                <PointList points={doc.points} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function PointList({ points }: { points: SummarizeResponse["combined_points"] }) {
  if (points.length === 0) return null;
  return (
    <ul className="mt-2 flex flex-col gap-1.5">
      {points.map((p, i) => (
        <li key={i} className="text-[13px] text-paper-dim">
          <span className="mr-1.5 text-muted">·</span>
          {p.text}
          <span className="ml-1.5 inline-flex flex-wrap gap-1">
            {p.citations.map((c) => (
              <EvidenceCitationChip key={c.chunk_id} citation={c} />
            ))}
          </span>
        </li>
      ))}
    </ul>
  );
}