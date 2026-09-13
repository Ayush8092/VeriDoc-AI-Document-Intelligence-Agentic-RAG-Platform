"use client";

import { useEffect, useState } from "react";
import { FileText, Loader2 } from "lucide-react";
import { listDocuments, ApiError } from "@/lib/api";
import type { DocumentOut } from "@/lib/types";

/** Shared multi-document selector for /compare, /extract, /summarize,
 * /report — each of those pages needs "pick N documents from my
 * library" and nothing page-specific about how that picker looks or
 * fetches data, so it lives here once rather than four times.
 */
export default function DocumentPicker({
  selected,
  onChange,
  min,
  max,
}: {
  selected: number[];
  onChange: (ids: number[]) => void;
  min?: number;
  max?: number;
}) {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listDocuments()
      .then((res) => {
        if (!cancelled) setDocuments(res.documents);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load documents.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function toggle(id: number) {
    if (selected.includes(id)) {
      onChange(selected.filter((x) => x !== id));
    } else {
      if (max && selected.length >= max) return;
      onChange([...selected, id]);
    }
  }

  if (error) {
    return <p className="text-[13px] text-rust">{error}</p>;
  }

  if (!documents) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-muted">
        <Loader2 size={14} className="animate-spin" />
        Loading documents…
      </div>
    );
  }

  if (documents.length === 0) {
    return <p className="text-[13px] text-muted">No documents yet — upload one first.</p>;
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="max-h-64 overflow-y-auto rounded-lg border border-ink-600 bg-ink-800">
        {documents.map((doc) => {
          const checked = selected.includes(doc.id);
          const disabled = !checked && !!max && selected.length >= max;
          return (
            <label
              key={doc.id}
              className={`flex cursor-pointer items-center gap-2.5 border-b border-ink-700 px-3 py-2 text-[13px] last:border-b-0 ${
                disabled ? "cursor-not-allowed opacity-40" : "hover:bg-ink-700"
              }`}
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={() => toggle(doc.id)}
                className="accent-brass"
              />
              <FileText size={14} className="shrink-0 text-muted" />
              <span className="truncate text-paper-dim">{doc.title || doc.filename}</span>
            </label>
          );
        })}
      </div>
      {min !== undefined && (
        <p className="text-[11px] text-muted">
          {selected.length} selected{max ? ` (max ${max})` : ""} — {min} required.
        </p>
      )}
    </div>
  );
}