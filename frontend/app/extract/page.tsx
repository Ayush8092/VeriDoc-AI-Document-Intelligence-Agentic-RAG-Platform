"use client";

import { useState } from "react";
import { ListTree, Loader2, Plus, Trash2, XCircle } from "lucide-react";
import { extractStructured, ApiError } from "@/lib/api";
import type { ExtractResponse, ExtractionFieldType } from "@/lib/types";
import DocumentPicker from "@/components/document-picker";
import EvidenceCitationChip from "@/components/evidence-citation-chip";

const FIELD_TYPES: ExtractionFieldType[] = ["string", "number", "date", "boolean", "list[string]"];

interface SchemaFieldRow {
  name: string;
  type: ExtractionFieldType;
}

export default function ExtractPage() {
  const [documentIds, setDocumentIds] = useState<number[]>([]);
  const [fields, setFields] = useState<SchemaFieldRow[]>([{ name: "", type: "string" }]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractResponse | null>(null);

  function updateField(i: number, patch: Partial<SchemaFieldRow>) {
    setFields((prev) => prev.map((f, idx) => (idx === i ? { ...f, ...patch } : f)));
  }

  const validFields = fields.filter((f) => f.name.trim());
  const canSubmit = documentIds.length >= 1 && validFields.length >= 1 && !busy;

  async function submit() {
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const schema: Record<string, ExtractionFieldType> = {};
      for (const f of validFields) schema[f.name.trim()] = f.type;
      const res = await extractStructured({ document_ids: documentIds, schema });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Extraction failed unexpectedly.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-6 overflow-y-auto px-6 py-8">
      <div>
        <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
          Extract structured fields
        </h1>
        <p className="mt-1 text-[13px] text-paper-dim">
          Define a schema of fields to pull out — each result is cited or marked not found, never guessed.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <label className="text-[12px] uppercase tracking-wide text-muted">Documents</label>
        <DocumentPicker selected={documentIds} onChange={setDocumentIds} min={1} />
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-[12px] uppercase tracking-wide text-muted">Fields to extract</label>
        {fields.map((field, i) => (
          <div key={i} className="flex items-center gap-2">
            <input
              value={field.name}
              onChange={(e) => updateField(i, { name: e.target.value })}
              placeholder="field name, e.g. effective_date"
              className="flex-1 rounded-lg border border-ink-600 bg-ink-800 px-3 py-2 text-[13px] text-paper placeholder:text-muted focus:border-brass-dim focus:outline-none"
            />
            <select
              value={field.type}
              onChange={(e) => updateField(i, { type: e.target.value as ExtractionFieldType })}
              className="rounded-lg border border-ink-600 bg-ink-800 px-2 py-2 text-[13px] text-paper focus:border-brass-dim focus:outline-none"
            >
              {FIELD_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <button
              onClick={() => setFields((prev) => prev.filter((_, idx) => idx !== i))}
              disabled={fields.length === 1}
              className="rounded-lg p-2 text-muted transition-colors hover:text-rust disabled:opacity-30"
              aria-label="Remove field"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
        <button
          onClick={() => setFields((prev) => [...prev, { name: "", type: "string" }])}
          className="flex w-fit items-center gap-1.5 rounded-lg border border-ink-600 px-3 py-1.5 text-[12px] text-paper-dim hover:border-brass-dim hover:text-paper"
        >
          <Plus size={13} />
          Add field
        </button>
      </div>

      <button
        onClick={submit}
        disabled={!canSubmit}
        className="flex items-center justify-center gap-2 rounded-lg bg-brass px-4 py-2.5 text-[14px] font-medium text-ink-950 transition-opacity disabled:opacity-30"
      >
        {busy ? <Loader2 size={16} className="animate-spin" /> : <ListTree size={16} />}
        Extract
      </button>

      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <XCircle size={16} className="mt-0.5 shrink-0 text-rust" />
          {error}
        </div>
      )}

      {result && (
        <div className="overflow-hidden rounded-lg border border-ink-600">
          <table className="w-full text-[13px]">
            <tbody>
              {Object.entries(result.fields).map(([name, field]) => (
                <tr key={name} className="border-b border-ink-700 last:border-b-0">
                  <td className="w-1/3 bg-ink-800 px-3 py-2.5 align-top font-mono text-[12px] text-muted">{name}</td>
                  <td className="px-3 py-2.5 align-top">
                    {field.status === "not_found" ? (
                      <span className="text-muted italic">not found</span>
                    ) : (
                      <>
                        <span className="text-paper">{JSON.stringify(field.value)}</span>
                        <span className="ml-2 font-mono text-[10px] text-muted">
                          {Math.round(field.confidence * 100)}% confidence
                        </span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {field.citations.map((c) => (
                            <EvidenceCitationChip key={c.chunk_id} citation={c} />
                          ))}
                        </div>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}