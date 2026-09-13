"use client";

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowUp, Loader2, XCircle } from "lucide-react";
import {
  getConversation,
  getConversationMessages,
  sendConversationMessage,
  getStoredAnonymousSessionId,
  setStoredAnonymousSessionId,
  ApiError,
} from "@/lib/api";
import type { ConversationMessageOut, ConversationOut } from "@/lib/types";
import { useAuth } from "@/components/auth-provider";
import EvidenceCitationChip from "@/components/evidence-citation-chip";

/** Next.js 15+ passes `params` as a Promise even to client component
 * pages — `use()` (React 19) unwraps it. See lib/api.ts's conversation
 * functions for why an anonymous caller's session id is read from
 * localStorage rather than passed as a route param: it's caller
 * identity, not part of the resource's address.
 *
 * fix_phase_6, Problem 4: this page previously lived at the FLAT
 * `/conversations` route (using a dynamic `params` prop that route
 * shape can't actually provide) — moved to `/conversations/[id]` where
 * a dynamic segment is real, alongside a proper list/create page now at
 * `/conversations` itself. See that page for the other half of the fix.
 */
export default function ConversationDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const conversationId = Number(id);
  const { user } = useAuth();

  const [conversation, setConversation] = useState<ConversationOut | null>(null);
  const [messages, setMessages] = useState<ConversationMessageOut[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const anonId = user ? undefined : getStoredAnonymousSessionId() ?? undefined;

  useEffect(() => {
    let cancelled = false;
    Promise.all([getConversation(conversationId, anonId), getConversationMessages(conversationId, anonId)])
      .then(([convo, msgs]) => {
        if (cancelled) return;
        setConversation(convo);
        setMessages(msgs);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load conversation.");
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function submit() {
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const res = await sendConversationMessage(conversationId, { question, anonymous_session_id: anonId });
      setMessages((prev) => [...prev, res.user_message, res.assistant_message]);
      if (!user && res.anonymous_session_id) {
        setStoredAnonymousSessionId(res.anonymous_session_id);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Message failed to send.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col px-6">
      <div className="border-b border-ink-700 py-4">
        <Link
          href="/conversations"
          className="mb-2 inline-flex items-center gap-1 text-[11px] text-muted transition-colors hover:text-paper"
        >
          <ArrowLeft size={12} />
          All conversations
        </Link>
        <h1 className="truncate text-[15px] text-paper">{conversation?.title || "Conversation"}</h1>
        {conversation && conversation.document_ids.length > 0 && (
          <p className="text-[11px] text-muted">Focused on {conversation.document_ids.length} document(s)</p>
        )}
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto py-6">
        <div className="flex flex-col gap-5">
          {messages.map((m) => (
            <MessageBlock key={m.id} message={m} />
          ))}
          {error && (
            <div className="flex items-start gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
              <XCircle size={16} className="mt-0.5 shrink-0 text-rust" />
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="sticky bottom-0 border-t border-ink-700 bg-ink-950 py-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="flex items-end gap-2 rounded-xl border border-ink-600 bg-ink-800 p-2 pl-4 focus-within:border-brass-dim"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Continue the conversation…"
            rows={1}
            className="max-h-40 flex-1 resize-none bg-transparent py-2 text-[14px] text-paper placeholder:text-muted focus:outline-none"
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brass text-ink-950 transition-opacity disabled:opacity-30"
            aria-label="Send"
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : <ArrowUp size={16} strokeWidth={2.5} />}
          </button>
        </form>
      </div>
    </div>
  );
}

function MessageBlock({ message }: { message: ConversationMessageOut }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-ink-700 px-4 py-2.5 text-[14px] text-paper">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <p className="text-[14px] leading-relaxed text-paper">{message.content}</p>
      {message.citations.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {message.citations.map((c) => (
            <EvidenceCitationChip
              key={c.chunk_id}
              citation={{
                chunk_id: c.chunk_id,
                source_file: c.source_file,
                page_start: c.page_start,
                page_end: c.page_end,
                document_id: null,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}