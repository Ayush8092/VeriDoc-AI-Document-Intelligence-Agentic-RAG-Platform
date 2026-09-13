"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowRight,
  Clock,
  RotateCw,
  UserCircle2,
} from "lucide-react";
import { uploadDocuments, getJobStatus, reingestDocument, ApiError } from "@/lib/api";
import type { IngestionJobOut, UploadItem } from "@/lib/types";
import { formatBytes } from "@/lib/format";
import { useAuth } from "@/components/auth-provider";

const ACCEPTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg"];

// Phase 5 completion pass, item 8: poll the REAL job endpoint — no
// simulated progress percentage anywhere in this file. The backend only
// ever reports one of IngestionJobStatus's six values; that's exactly
// what's rendered.
const POLL_INTERVAL_MS = 1500;
const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

function newId() {
  return Math.random().toString(36).slice(2);
}

export default function UploadPage() {
  const { user } = useAuth();
  const [items, setItems] = useState<UploadItem[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    setItems((prev) => [
      ...prev,
      ...files.map((file) => ({ id: newId(), file, status: "queued" as const })),
    ]);
    setBatchError(null);
  }, []);

  async function submit() {
    const toUpload = items.filter((i) => i.status === "queued" || i.status === "rejected");
    if (toUpload.length === 0 || busy) return;

    setBusy(true);
    setBatchError(null);
    setItems((prev) =>
      prev.map((i) => (toUpload.some((q) => q.id === i.id) ? { ...i, status: "uploading" } : i)),
    );

    try {
      const response = await uploadDocuments(toUpload.map((i) => i.file));
      const rejectedByName = new Map(response.rejected.map((r) => [r.filename, r.error]));
      const jobByName = new Map(response.jobs.map((j) => [j.filename, j.job_id]));

      setItems((prev) =>
        prev.map((i) => {
          if (!toUpload.some((q) => q.id === i.id)) return i;
          const err = rejectedByName.get(i.file.name);
          if (err) return { ...i, status: "rejected", error: err };
          const jobId = jobByName.get(i.file.name);
          // Every accepted file has a job — see UploadResponse's
          // backend contract. If one is somehow missing (shouldn't
          // happen), fall back to an explicit error rather than silently
          // stalling at "uploading" forever.
          return jobId
            ? { ...i, status: "queued", jobId }
            : { ...i, status: "failed", error: "Accepted, but no job id was returned." };
        }),
      );
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Upload failed unexpectedly.";
      setBatchError(message);
      setItems((prev) =>
        prev.map((i) => (toUpload.some((q) => q.id === i.id) ? { ...i, status: "rejected", error: message } : i)),
      );
    } finally {
      setBusy(false);
    }
  }

  function remove(id: string) {
    setItems((prev) => prev.filter((i) => i.id !== id));
  }

  async function retry(item: UploadItem) {
    if (!item.documentId) return; // nothing to retry against yet — see FileRow's guard
    setItems((prev) =>
      prev.map((i) => (i.id === item.id ? { ...i, status: "queued", error: undefined } : i)),
    );
    try {
      const res = await reingestDocument(item.documentId);
      setItems((prev) =>
        prev.map((i) => (i.id === item.id ? { ...i, jobId: res.job.id, status: res.job.status } : i)),
      );
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Retry failed to queue.";
      setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, status: "failed", error: message } : i)));
    }
  }

  // Poll every non-terminal job on an interval. One shared interval
  // (not one per item) so N in-flight uploads don't open N separate
  // timers.
  useEffect(() => {
    const hasPending = items.some((i) => i.jobId && !TERMINAL_STATUSES.has(i.status));
    if (!hasPending) return;

    const interval = setInterval(async () => {
      const pending = items.filter((i) => i.jobId && !TERMINAL_STATUSES.has(i.status));
      const results = await Promise.allSettled(
        pending.map((i) => getJobStatus(i.jobId as number)),
      );
      setItems((prev) => {
        const byId = new Map(pending.map((i, idx) => [i.id, results[idx]]));
        return prev.map((i) => {
          const result = byId.get(i.id);
          if (!result) return i;
          if (result.status === "rejected") return i; // transient poll failure — try again next tick
          const job: IngestionJobOut = result.value;
          return {
            ...i,
            status: job.status,
            documentId: job.document_id ?? i.documentId,
            error: job.status === "failed" ? job.error || "Ingestion failed." : undefined,
          };
        });
      });
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [items]);

  const hasQueued = items.some((i) => i.status === "queued" && !i.jobId);
  const anyDone = items.some((i) => i.status === "completed");

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-3xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
        Upload
      </h1>
      <p className="mt-1 text-[13px] text-paper-dim">
        Add documents to the knowledge base. Supported formats: {ACCEPTED_EXTENSIONS.join(", ")} —
        scanned PDFs and photos are OCR&apos;d automatically.
      </p>
      <p className="mt-2 flex items-center gap-1.5 text-[12px] text-muted">
        <UserCircle2 size={13} />
        {user === undefined
          ? "…"
          : user
            ? `Uploading privately as ${user.email}.`
            : "Uploading to the shared library (anonymous). Sign in to keep uploads private to your account."}
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={clsx(
          "mt-6 flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors",
          dragActive ? "border-brass bg-brass-dim/10" : "border-ink-600 hover:border-ink-500",
        )}
      >
        <UploadCloud size={28} className={dragActive ? "text-brass" : "text-muted"} />
        <div>
          <p className="text-[14px] text-paper">Drop files here, or click to browse</p>
          <p className="mt-1 text-[12px] text-muted">Multiple files at once are fine.</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {items.length > 0 && (
        <div className="mt-6 flex flex-col gap-2">
          {items.map((item) => (
            <FileRow key={item.id} item={item} onRemove={() => remove(item.id)} onRetry={() => retry(item)} />
          ))}
        </div>
      )}

      {batchError && (
        <div className="mt-4 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          {batchError}
        </div>
      )}

      {items.length > 0 && (
        <div className="mt-6 flex items-center justify-between">
          <button
            onClick={submit}
            disabled={!hasQueued || busy}
            className="flex items-center gap-2 rounded-lg bg-brass px-4 py-2 text-[13px] font-medium text-ink-950 transition-opacity disabled:opacity-30"
          >
            {busy && <Loader2 size={14} className="animate-spin" />}
            {busy ? "Uploading…" : "Upload"}
          </button>
          {anyDone && (
            <Link
              href="/library"
              className="flex items-center gap-1 text-[13px] text-brass-soft hover:text-brass"
            >
              View in library <ArrowRight size={13} />
            </Link>
          )}
        </div>
      )}
    </div>
  );
}

function FileRow({
  item,
  onRemove,
  onRetry,
}: {
  item: UploadItem;
  onRemove: () => void;
  onRetry: () => void;
}) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-ink-600 bg-ink-800 px-3.5 py-2.5">
      <FileText size={15} className="shrink-0 text-muted" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] text-paper">{item.file.name}</p>
        <p className="font-mono text-[11px] text-muted">{formatBytes(item.file.size)}</p>
        {item.error && <p className="mt-0.5 text-[11px] text-rust">{item.error}</p>}
      </div>
      <StatusBadge status={item.status} />
      {item.status === "failed" && item.documentId && (
        <button
          onClick={onRetry}
          className="flex shrink-0 items-center gap-1 rounded-sm border border-ink-600 px-2 py-1 text-[11px] text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
        >
          <RotateCw size={11} />
          Retry
        </button>
      )}
      {(item.status === "queued" && !item.jobId) || item.status === "rejected" ? (
        <button onClick={onRemove} className="shrink-0 text-[11px] text-muted hover:text-paper">
          Remove
        </button>
      ) : null}
    </div>
  );
}

const STATUS_META: Record<UploadItem["status"], { label: string; className: string }> = {
  queued: { label: "Queued", className: "text-muted" },
  uploading: { label: "Uploading…", className: "text-brass" },
  processing: { label: "Processing…", className: "text-brass" },
  retrying: { label: "Retrying…", className: "text-brass" },
  completed: { label: "Indexed", className: "text-teal" },
  failed: { label: "Failed", className: "text-rust" },
  cancelled: { label: "Cancelled", className: "text-muted" },
  rejected: { label: "Rejected", className: "text-rust" },
};

function StatusBadge({ status }: { status: UploadItem["status"] }) {
  const meta = STATUS_META[status];
  const icon =
    status === "uploading" || status === "processing" || status === "retrying" ? (
      <Loader2 size={13} className="animate-spin" />
    ) : status === "completed" ? (
      <CheckCircle2 size={13} />
    ) : status === "failed" || status === "rejected" ? (
      <XCircle size={13} />
    ) : (
      <Clock size={13} />
    );
  return (
    <span className={clsx("flex shrink-0 items-center gap-1.5 font-mono text-[11px]", meta.className)}>
      {icon}
      {meta.label}
    </span>
  );
}
