"use client";

import { useState } from "react";
import { GitCompare, Loader2, XCircle } from "lucide-react";
import { compareDocuments, ApiError } from "@/lib/api";
import { isTwoDocComparison, type CompareResponse } from "@/lib/types";
import DocumentPicker from "@/components/document-picker";
import EvidenceCitationChip from "@/components/evidence-citation-chip";

const RELATION_STYLE: Record<string, string> = {
  same: "text-teal",
  agree: "text-teal",
  different: "text-brass",
  partial: "text-brass",
  missing_in_a: "text-muted",
  missing_in_b: "text-muted",
  conflicting: "text-rust",
  disagree: "text-rust",
};

export default function ComparePage() {
  const [documentIds, setDocumentIds] = useState<number[]>([]);
  const [topic, setTopic] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);

  async function submit() {
    if (documentIds.length < 2 || !topic.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await compareDocuments({ topic: topic.trim(), document_ids: documentIds });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Comparison failed unexpectedly.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-6 overflow-y-auto px-6 py-8">
      <div>
        <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
          Compare documents
        </h1>
        <p className="mt-1 text-[13px] text-paper-dim">
          Pick 2 or more documents and a topic — every point is cited back to each document separately, never blended.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <label className="text-[12px] uppercase tracking-wide text-muted">Documents to compare</label>
        <DocumentPicker selected={documentIds} onChange={setDocumentIds} min={2} />
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-[12px] uppercase tracking-wide text-muted">Topic</label>
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. notice period, termination fees, renewal terms"
          className="rounded-lg border border-ink-600 bg-ink-800 px-3 py-2 text-[14px] text-paper placeholder:text-muted focus:border-brass-dim focus:outline-none"
        />
      </div>

      <button
        onClick={submit}
        disabled={busy || documentIds.length < 2 || !topic.trim()}
        className="flex items-center justify-center gap-2 rounded-lg bg-brass px-4 py-2.5 text-[14px] font-medium text-ink-950 transition-opacity disabled:opacity-30"
      >
        {busy ? <Loader2 size={16} className="animate-spin" /> : <GitCompare size={16} />}
        Compare
      </button>

      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <XCircle size={16} className="mt-0.5 shrink-0 text-rust" />
          {error}
        </div>
      )}

      {result && <ComparisonResult result={result} />}
    </div>
  );
}

function ComparisonResult({ result }: { result: CompareResponse }) {
  if (isTwoDocComparison(result)) {
    return (
      <div className="flex flex-col gap-3">
        <p className="text-[12px] text-muted">
          {result.document_a.title} vs. {result.document_b.title}
        </p>
        {result.points.length === 0 && (
          <p className="text-[13px] text-muted">No comparable points found for this topic.</p>
        )}
        {result.points.map((point, i) => (
          <div key={i} className="rounded-lg border border-ink-600 bg-ink-800 p-4">
            <div className="flex items-center justify-between">
              <span className="text-[13px] font-medium text-paper">{point.aspect}</span>
              <span className={`font-mono text-[10px] uppercase tracking-wide ${RELATION_STYLE[point.relation] ?? "text-muted"}`}>
                {point.relation.replace(/_/g, " ")}
              </span>
            </div>
            <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div>
                <p className="text-[11px] uppercase text-muted">{result.document_a.title}</p>
                <p className="mt-1 text-[13px] text-paper-dim">{point.summary_a || "—"}</p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {point.evidence_a.map((c) => (
                    <EvidenceCitationChip key={c.chunk_id} citation={c} />
                  ))}
                </div>
              </div>
              <div>
                <p className="text-[11px] uppercase text-muted">{result.document_b.title}</p>
                <p className="mt-1 text-[13px] text-paper-dim">{point.summary_b || "—"}</p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {point.evidence_b.map((c) => (
                    <EvidenceCitationChip key={c.chunk_id} citation={c} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-[12px] text-muted">{result.documents.map((d) => d.title).join(", ")}</p>
      {result.points.length === 0 && <p className="text-[13px] text-muted">No comparable points found for this topic.</p>}
      {result.points.map((point, i) => (
        <div key={i} className="rounded-lg border border-ink-600 bg-ink-800 p-4">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-medium text-paper">{point.aspect}</span>
            <span className={`font-mono text-[10px] uppercase tracking-wide ${RELATION_STYLE[point.consensus] ?? "text-muted"}`}>
              {point.consensus}
            </span>
          </div>
          <div className="mt-2 flex flex-col gap-2">
            {Object.entries(point.summary_by_document).map(([docId, summary]) => {
              const doc = result.documents.find((d) => String(d.id) === docId);
              return (
                <div key={docId}>
                  <p className="text-[11px] uppercase text-muted">{doc?.title ?? `Document ${docId}`}</p>
                  <p className="mt-0.5 text-[13px] text-paper-dim">{summary}</p>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {(point.evidence_by_document[docId] ?? []).map((c) => (
                      <EvidenceCitationChip key={c.chunk_id} citation={c} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}