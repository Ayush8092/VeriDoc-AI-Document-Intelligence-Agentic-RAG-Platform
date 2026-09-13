"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { MessagesSquare, Plus, Trash2, AlertTriangle, Loader2, X } from "lucide-react";
import {
  listConversations,
  createConversation,
  deleteConversation,
  listDocuments,
  getStoredAnonymousSessionId,
  setStoredAnonymousSessionId,
  ApiError,
} from "@/lib/api";
import type { ConversationOut, DocumentOut } from "@/lib/types";
import { useAuth } from "@/components/auth-provider";
import { formatDate } from "@/lib/format";

/**
 * fix_phase_6, Problem 4: `/conversations` previously had no real
 * list/create page — the file at this path was actually the dynamic
 * per-conversation chat view (now correctly at
 * `app/conversations/[id]/page.tsx`). This is the other half of the
 * fix: list existing conversations, create a new one (optionally
 * document-focused), and navigate into one by id.
 */
export default function ConversationsListPage() {
  const { user } = useAuth();
  const router = useRouter();

  const [conversations, setConversations] = useState<ConversationOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);

  const anonId = user ? undefined : getStoredAnonymousSessionId() ?? undefined;

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await listConversations(anonId);
      setConversations(res.conversations);
    } catch (err) {
      // An anonymous caller with no session yet (no anonymous_session_id
      // in localStorage) has, definitionally, no conversations —
      // backend-independent empty state, not an error to surface.
      if (!user && !anonId) {
        setConversations([]);
      } else {
        setError(err instanceof ApiError ? err.message : "Could not load conversations.");
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    // react-hooks/set-state-in-effect: `load()` sets `loading`/`error`
    // synchronously before its first `await` — a one-shot data fetch on
    // mount/user-change, not a render loop risk. Matches this
    // component's existing exhaustive-deps suppression immediately
    // below (both are intentional, not oversights).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  async function handleCreated(conversation: ConversationOut, anonymousSessionId?: string | null) {
    if (!user && anonymousSessionId) {
      setStoredAnonymousSessionId(anonymousSessionId);
    }
    router.push(`/conversations/${conversation.id}`);
  }

  async function handleDelete(id: number) {
    try {
      await deleteConversation(id, anonId);
      setConversations((prev) => (prev ? prev.filter((c) => c.id !== id) : prev));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not delete this conversation.");
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <div className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-3xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
            Conversations
          </h1>
          <p className="mt-1 text-[13px] text-paper-dim">
            Multi-turn Q&A with rolling context — optionally focused on specific documents.
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 rounded-md bg-brass px-3 py-1.5 text-[12px] font-medium text-ink-950 transition-opacity hover:opacity-90"
        >
          <Plus size={14} />
          New conversation
        </button>
      </div>

      {error && (
        <div className="mb-6 flex items-center gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
          <AlertTriangle size={16} className="shrink-0 text-rust" />
          {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-2 py-12 text-[13px] text-muted">
          <Loader2 size={14} className="animate-spin" />
          Loading conversations…
        </div>
      )}

      {!loading && conversations && conversations.length === 0 && (
        <div className="rounded-lg border border-dashed border-ink-600 px-6 py-12 text-center text-[13px] text-muted">
          No conversations yet. Start one to ask multi-turn questions with context carried forward.
        </div>
      )}

      {!loading && conversations && conversations.length > 0 && (
        <div className="flex flex-col gap-2.5">
          {conversations.map((c) => (
            <ConversationRow
              key={c.id}
              conversation={c}
              onOpen={() => router.push(`/conversations/${c.id}`)}
              onDelete={() => handleDelete(c.id)}
            />
          ))}
        </div>
      )}

      {showCreate && (
        <CreateConversationDialog onClose={() => setShowCreate(false)} onCreated={handleCreated} anonId={anonId} />
      )}
    </div>
  );
}

function ConversationRow({
  conversation,
  onOpen,
  onDelete,
}: {
  conversation: ConversationOut;
  onOpen: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-ink-600 bg-ink-800 px-4 py-3.5">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-ink-700 text-brass-soft">
        <MessagesSquare size={16} />
      </div>
      <button onClick={onOpen} className="min-w-0 flex-1 text-left">
        <p className="truncate text-[14px] font-medium text-paper">{conversation.title || "Untitled conversation"}</p>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
          {conversation.document_ids.length > 0 && (
            <span>Focused on {conversation.document_ids.length} document(s)</span>
          )}
          {conversation.updated_at && <span>Updated {formatDate(conversation.updated_at)}</span>}
        </div>
      </button>
      <button
        onClick={onDelete}
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted transition-colors hover:text-rust"
        aria-label="Delete conversation"
        title="Delete conversation"
      >
        <Trash2 size={13} />
      </button>
    </div>
  );
}

function CreateConversationDialog({
  onClose,
  onCreated,
  anonId,
}: {
  onClose: () => void;
  onCreated: (conversation: ConversationOut, anonymousSessionId?: string | null) => void;
  anonId?: string;
}) {
  const [title, setTitle] = useState("");
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => setDocuments([])); // document picker is optional — a load failure shouldn't block creating a whole-corpus conversation
  }, []);

  function toggleDocument(id: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const conversation = await createConversation({
        title: title.trim(),
        document_ids: Array.from(selectedIds),
        anonymous_session_id: anonId,
      });
      // A fresh anonymous session's id is already on the created
      // conversation row itself (assigned server-side on first use).
      onCreated(conversation, conversation.anonymous_session_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not create the conversation.");
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/70 px-4">
      <div className="w-full max-w-md rounded-lg border border-ink-600 bg-ink-800 p-5">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-[15px] font-medium text-paper">New conversation</h2>
          <button onClick={onClose} className="text-muted hover:text-paper" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <label className="mb-1 block text-[11px] uppercase tracking-wide text-muted">Title (optional)</label>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Q3 report questions"
          className="mb-4 w-full rounded-md border border-ink-600 bg-ink-900 px-3 py-2 text-[13px] text-paper placeholder:text-muted focus:border-brass-dim focus:outline-none"
        />

        <label className="mb-1 block text-[11px] uppercase tracking-wide text-muted">
          Focus on documents (optional — leave empty to search the whole corpus)
        </label>
        <div className="mb-4 max-h-48 overflow-y-auto rounded-md border border-ink-600">
          {documents === null && (
            <div className="flex items-center gap-2 px-3 py-3 text-[12px] text-muted">
              <Loader2 size={12} className="animate-spin" />
              Loading documents…
            </div>
          )}
          {documents !== null && documents.length === 0 && (
            <p className="px-3 py-3 text-[12px] text-muted">No documents available.</p>
          )}
          {documents?.map((d) => (
            <label
              key={d.id}
              className="flex cursor-pointer items-center gap-2 px-3 py-2 text-[12px] text-paper-dim hover:bg-ink-700"
            >
              <input type="checkbox" checked={selectedIds.has(d.id)} onChange={() => toggleDocument(d.id)} />
              <span className="truncate">{d.title || d.filename}</span>
            </label>
          ))}
        </div>

        {error && (
          <div className="mb-3 flex items-center gap-2 rounded-md border border-rust-dim bg-rust-dim/20 px-3 py-2 text-[12px] text-paper">
            <AlertTriangle size={13} className="shrink-0 text-rust" />
            {error}
          </div>
        )}

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            disabled={busy}
            className="rounded-md border border-ink-600 px-3 py-1.5 text-[12px] text-paper-dim hover:border-brass-dim hover:text-paper disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-md bg-brass px-3 py-1.5 text-[12px] font-medium text-ink-950 transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {busy && <Loader2 size={12} className="animate-spin" />}
            Create
          </button>
        </div>
      </div>
    </div>
  );
}