"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import clsx from "clsx";
import { MessageSquareText, Library, UploadCloud, Circle, ScanSearch, LogIn, LogOut, UserCircle2, GitCompare, ListTree, FileText, Files, MessagesSquare } from "lucide-react";
import { getReady } from "@/lib/api";
import { useAuth } from "@/components/auth-provider";

const NAV_ITEMS = [
  { href: "/ask", label: "Ask", icon: MessageSquareText },
  { href: "/conversations", label: "Conversations", icon: MessagesSquare },
  { href: "/library", label: "Library", icon: Library },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/extract", label: "Extract", icon: ListTree },
  { href: "/summarize", label: "Summarize", icon: Files },
  { href: "/report", label: "Report", icon: FileText },
  { href: "/source", label: "Source Viewer", icon: ScanSearch },
  { href: "/upload", label: "Upload", icon: UploadCloud },
];

type BackendStatus = "checking" | "ready" | "not-ready" | "offline";

function useBackendStatus(): { status: BackendStatus; detail: string } {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [detail, setDetail] = useState("Checking connection…");

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const res = await getReady();
        if (cancelled) return;
        setStatus(res.ready ? "ready" : "not-ready");
        setDetail(res.detail);
      } catch (err) {
        if (cancelled) return;
        setStatus("offline");
        setDetail(err instanceof Error ? err.message : "Could not reach the API.");
      }
    }

    check();
    const interval = setInterval(check, 20_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { status, detail };
}

const STATUS_STYLES: Record<BackendStatus, { color: string; label: string }> = {
  checking: { color: "text-muted", label: "Checking…" },
  ready: { color: "text-teal", label: "Ready" },
  "not-ready": { color: "text-brass", label: "Starting up" },
  offline: { color: "text-rust", label: "Offline" },
};

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { status, detail } = useBackendStatus();
  const { user, logout } = useAuth();
  const style = STATUS_STYLES[status];

  return (
    <div className="flex h-screen bg-ink-950">
      <aside className="flex w-60 shrink-0 flex-col border-r border-ink-700 bg-ink-900">
        <div className="px-5 py-6">
          <span
            className="text-2xl tracking-tight text-paper"
            style={{ fontFamily: "var(--font-display)", fontOpticalSizing: "auto" }}
          >
            Veridoc
          </span>
          <p className="mt-1 text-[11px] tracking-wide text-muted uppercase">Document Intelligence</p>
        </div>

        <nav className="flex flex-col gap-1 px-3">
          {NAV_ITEMS.map((item) => {
            const active = pathname?.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                  active
                    ? "bg-ink-700 text-paper"
                    : "text-paper-dim hover:bg-ink-800 hover:text-paper",
                )}
              >
                <Icon size={16} strokeWidth={active ? 2.25 : 1.75} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Phase 5 completion pass, item 6/7: auth state + sign-in/out.
            Deliberately not a hard route gate — see components/auth-provider.tsx's
            docstring: the backend treats anonymous requests as the
            shared public corpus by design (allowed_owner_ids), so
            /library, /upload, /source all still work signed-out. This
            is the one place that surfaces "who you're browsing as". */}
        <div className="border-t border-ink-700 px-3 py-3">
          {user === undefined ? null : user ? (
            <div className="flex items-center gap-2 rounded-md px-2 py-1.5">
              <UserCircle2 size={16} className="shrink-0 text-brass-soft" />
              <span className="min-w-0 flex-1 truncate text-[12px] text-paper-dim" title={user.email}>
                {user.email}
              </span>
              <button
                onClick={logout}
                title="Sign out"
                className="shrink-0 text-paper-dim transition-colors hover:text-paper"
              >
                <LogOut size={14} />
              </button>
            </div>
          ) : (
            <Link
              href={`/login?redirect=${encodeURIComponent(pathname || "/library")}`}
              className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-paper-dim transition-colors hover:bg-ink-800 hover:text-paper"
            >
              <LogIn size={16} strokeWidth={1.75} />
              Sign in
            </Link>
          )}
        </div>

        <div className="mt-auto border-t border-ink-700 px-4 py-4">
          <div className="flex items-center gap-2" title={detail}>
            <Circle size={8} className={clsx(style.color, "fill-current")} />
            <span className="text-[12px] text-paper-dim">{style.label}</span>
          </div>
          <p className="mt-1 truncate font-mono text-[10px] text-muted" title={detail}>
            {detail}
          </p>
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}