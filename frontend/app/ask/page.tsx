"use client";

import { useEffect, useRef, useState } from "react";
import { CheckCircle2, XCircle, ArrowUp, Loader2 } from "lucide-react";
import { ask, ApiError } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import EvidenceStamp from "@/components/evidence-stamp";

const SAMPLE_QUESTIONS = [
  "How long is the free trial?",
  "What is the approval threshold for conference fees?",
  "What was Asia Pacific's revenue growth this quarter?",
  "How many paid leave days do employees get?",
];

function newId() {
  return Math.random().toString(36).slice(2);
}

export default function AskPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function submit(question: string) {
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    const userMsg: ChatMessage = { id: newId(), role: "user", question: trimmed };
    const pendingMsg: ChatMessage = { id: newId(), role: "assistant", pending: true };
    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setInput("");
    setBusy(true);

    try {
      const response = await ask(trimmed);
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingMsg.id ? { ...m, pending: false, response } : m)),
      );
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Something went wrong answering that question.";
      setMessages((prev) => prev.map((m) => (m.id === pendingMsg.id ? { ...m, pending: false, error: message } : m)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col px-6">
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-8">
        {messages.length === 0 ? (
          <EmptyState onPick={submit} />
        ) : (
          <div className="flex flex-col gap-6">
            {messages.map((msg) => (
              <MessageBlock key={msg.id} message={msg} />
            ))}
          </div>
        )}
      </div>

      <div className="sticky bottom-0 border-t border-ink-700 bg-ink-950 py-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
          className="flex items-end gap-2 rounded-xl border border-ink-600 bg-ink-800 p-2 pl-4 focus-within:border-brass-dim"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(input);
              }
            }}
            placeholder="Ask a question about your documents…"
            rows={1}
            className="max-h-40 flex-1 resize-none bg-transparent py-2 text-[14px] text-paper placeholder:text-muted focus:outline-none"
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brass text-ink-950 transition-opacity disabled:opacity-30"
            aria-label="Ask"
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : <ArrowUp size={16} strokeWidth={2.5} />}
          </button>
        </form>
        <p className="mt-2 text-center text-[11px] text-muted">
          Answers are restricted to your ingested documents — Veridoc will say so when it can&apos;t find one.
        </p>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 text-center">
      <h1
        className="text-4xl text-paper"
        style={{ fontFamily: "var(--font-display)" }}
      >
        What do your documents say?
      </h1>
      <p className="max-w-md text-[14px] text-paper-dim">
        Every answer is grounded in retrieved evidence — pages, sections, or table cells you can
        check yourself.
      </p>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {SAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-lg border border-ink-600 bg-ink-800 px-4 py-2.5 text-left text-[13px] text-paper-dim transition-colors hover:border-brass-dim hover:text-paper"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBlock({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-ink-700 px-4 py-2.5 text-[14px] text-paper">
          {message.question}
        </div>
      </div>
    );
  }

  if (message.pending) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-muted">
        <Loader2 size={14} className="animate-spin" />
        Retrieving and grading evidence…
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="flex items-start gap-2.5 rounded-lg border border-rust-dim bg-rust-dim/20 px-4 py-3 text-[13px] text-paper">
        <XCircle size={16} className="mt-0.5 shrink-0 text-rust" />
        {message.error}
      </div>
    );
  }

  const response = message.response!;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 shrink-0">
          {response.found ? (
            <CheckCircle2 size={18} className="text-teal" />
          ) : (
            <XCircle size={18} className="text-rust" />
          )}
        </div>
        <div className="flex-1">
          <p className="text-[14px] leading-relaxed text-paper">{response.answer}</p>
          <span
            className={
              response.found
                ? "mt-1.5 inline-block rounded-sm bg-teal-dim px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-teal"
                : "mt-1.5 inline-block rounded-sm bg-rust-dim px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-wide text-rust"
            }
          >
            {response.found ? "Grounded in evidence" : "No supporting evidence found"}
          </span>
        </div>
      </div>

      {response.citations.length > 0 && (
        <div className="ml-[26px] flex flex-col gap-2">
          {response.citations.map((citation, i) => (
            <EvidenceStamp key={citation.chunk_id} citation={citation} index={i} allCitations={response.citations} />
          ))}
        </div>
      )}
    </div>
  );
}