"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import {
  FileText,
  FileType2,
  Image as ImageIcon,
  RefreshCw,
  ScanText,
  Table2,
  AlertTriangle,
  Loader2,
  Trash2,
} from "lucide-react";
import { listDocuments, deleteDocument, ApiError } from "@/lib/api";
import type { DocumentOut } from "@/lib/types";
import { formatDate } from "@/lib/format";
import ConfidenceMeter from "@/components/confidence-meter";
import { useAuth } from "@/components/auth-provider";

const FILE_ICONS: Record<string, typeof FileText> = {
  pdf: FileText,
  docx: FileType2,
  md: FileText,
  txt: FileText,
  png: ImageIcon,
  jpg: ImageIcon,
  jpeg: ImageIcon,
};

export default function LibraryPage() {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // `fetchAndSetDocuments` deliberately does nothing synchronous before
  // its first `await` — shared by the mount effect below and by `load`
  // (the retry-button handler), which additionally resets `loading`/
  // `error` synchronously before calling it — fine there since `load`
  // is only ever invoked from a plain `onClick` event handler, not from
  // inside an effect.
  async function fetchAndSetDocuments(cancelledRef?: { current: boolean }) {
    try {
      const res = await listDocuments();
      if (!cancelledRef?.current) setDocuments(res.documents);
    } catch (err) {
      if (!cancelledRef?.current) {
        setError(err instanceof ApiError ? err.message : "Could not load the document library.");
      }
    } finally {
      if (!cancelledRef?.current) setLoading(false);
    }
  }

  async function load() {
    setLoading(true);
    setError(null);
    await fetchAndSetDocuments();
  }

  useEffect(() => {
    const cancelledRef = { current: false };

    async function run() {
      await fetchAndSetDocuments(cancelledRef);
    }

    run();
    return () => {
      cancelledRef.current = true;
    };
  }, []);

  const corpusDocs = documents?.filter((d) => d.source_root === "corpus") ?? [];
  const uploadDocs = documents?.filter((d) => d.source_root === "uploads") ?? [];

  async function handleDeleted(documentId: number) {
    // Optimistic removal — the DELETE call already succeeded by the
    // time this is called (see DocumentCard's onDelete); re-fetching
    // the whole list would work too but this avoids a visible
    // flash/round-trip for something that's already done.
    setDocuments((prev) => (prev ? prev.filter((d) => d.id !== documentId) : prev));
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-3xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
            Library
          </h1>
          <p className="mt-1 text-[13px] text-paper-dim">
            Every document Veridoc has ingested, with per-version page, table, and OCR stats.
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-md border border-ink-600 px-3 py-1.5 text-[12px] text-paper-dim transition-colors hover:border-brass-dim hover:text-paper disabled:opacity-40"
        >
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="mb-6 flex items-center gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <AlertTriangle size={16} className="shrink-0 text-rust" />
          {error}
        </div>
      )}

      {loading && !documents && (
        <div className="flex items-center gap-2 py-12 text-[13px] text-muted">
          <Loader2 size={14} className="animate-spin" />
          Loading library…
        </div>
      )}

      {documents && documents.length === 0 && !error && (
        <div className="rounded-lg border border-dashed border-ink-600 px-6 py-12 text-center text-[13px] text-muted">
          No documents ingested yet. Run the ingestion CLI, or head to{" "}
          <span className="text-paper-dim">Upload</span> to add some.
        </div>
      )}

      {corpusDocs.length > 0 && <DocumentSection title="Corpus" documents={corpusDocs} onDeleted={handleDeleted} />}
      {uploadDocs.length > 0 && <DocumentSection title="Uploaded" documents={uploadDocs} onDeleted={handleDeleted} />}
    </div>
  );
}

function DocumentSection({
  title,
  documents,
  onDeleted,
}: {
  title: string;
  documents: DocumentOut[];
  onDeleted: (documentId: number) => void;
}) {
  return (
    <div className="mb-8">
      <h2 className="mb-3 font-mono text-[11px] uppercase tracking-wide text-muted">{title}</h2>
      <div className="flex flex-col gap-2.5">
        {documents.map((doc) => (
          <DocumentCard key={doc.id} doc={doc} onDeleted={onDeleted} />
        ))}
      </div>
    </div>
  );
}

function DocumentCard({ doc, onDeleted }: { doc: DocumentOut; onDeleted: (documentId: number) => void }) {
  const Icon = FILE_ICONS[doc.file_type] ?? FileText;
  const v = doc.current_version;
  const { user } = useAuth();
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Matches backend app/api/documents.py::delete_document's own scope
  // check exactly (only the owner can delete; shared corpus content —
  // owner_id === null — is never deletable through this UI, same as the
  // backend never allows it through the API either regardless of what
  // this button does or doesn't show).
  const canDelete = user != null && doc.owner_id === user.id;

  async function handleDelete() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteDocument(doc.id);
      onDeleted(doc.id);
    } catch (err) {
      setDeleteError(err instanceof ApiError ? err.message : "Could not delete this document.");
      setDeleting(false);
      setConfirming(false);
    }
  }

  return (
    <div className="flex items-start gap-4 rounded-lg border border-ink-600 bg-ink-800 px-4 py-3.5">
      <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-ink-700 text-brass-soft">
        <Icon size={16} />
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="truncate text-[14px] font-medium text-paper">{doc.title || doc.filename}</span>
          <span className="font-mono text-[11px] text-muted">{doc.filename}</span>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[12px] text-paper-dim">
          {v && (
            <>
              <span className="flex items-center gap-1">
                <FileText size={12} className="text-muted" />
                {v.page_count} {v.page_count === 1 ? "page" : "pages"}
              </span>
              {v.table_count > 0 && (
                <span className="flex items-center gap-1">
                  <Table2 size={12} className="text-muted" />
                  {v.table_count} {v.table_count === 1 ? "table" : "tables"}
                </span>
              )}
              <span className="font-mono text-muted">{v.chunk_count} chunks</span>
              <span className="text-muted">v{v.version_number}</span>
              <span className="text-muted">{formatDate(v.created_at)}</span>
            </>
          )}
        </div>

        {v?.warnings && v.warnings.length > 0 && (
          <div className="mt-2 flex flex-col gap-1">
            {v.warnings.map((w, i) => (
              <span key={i} className="flex items-start gap-1.5 text-[11px] text-rust">
                <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                {w}
              </span>
            ))}
          </div>
        )}

        {deleteError && (
          <p className="mt-2 flex items-start gap-1.5 text-[11px] text-rust">
            <AlertTriangle size={11} className="mt-0.5 shrink-0" />
            {deleteError}
          </p>
        )}
      </div>

      <div className="flex shrink-0 flex-col items-end gap-1.5">
        {v?.has_scanned_pages && (
          <span
            className={clsx(
              "flex items-center gap-1 rounded-sm bg-teal-dim px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-teal",
            )}
          >
            <ScanText size={10} />
            OCR
          </span>
        )}
        {v?.mean_ocr_confidence != null && (
          <ConfidenceMeter confidence={v.mean_ocr_confidence / 100} source="ocr" />
        )}

        {canDelete && (
          <div className="mt-1">
            {confirming ? (
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="flex items-center gap-1 rounded-sm border border-rust px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-rust transition-colors hover:bg-rust-dim/30 disabled:opacity-50"
                >
                  {deleting ? <Loader2 size={11} className="animate-spin" /> : null}
                  Confirm delete
                </button>
                <button
                  onClick={() => setConfirming(false)}
                  disabled={deleting}
                  className="rounded-sm border border-ink-600 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-paper-dim transition-colors hover:border-brass-dim hover:text-paper disabled:opacity-50"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirming(true)}
                className="flex items-center gap-1 rounded-sm border border-ink-600 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-paper-dim transition-colors hover:border-rust hover:text-rust"
                title="Delete this document"
              >
                <Trash2 size={11} />
                Delete
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}