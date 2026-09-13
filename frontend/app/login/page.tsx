"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { LogIn, Loader2, AlertTriangle } from "lucide-react";
import { useAuth } from "@/components/auth-provider";
import { ApiError } from "@/lib/api";

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const { login } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get("redirect") || "/library";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      router.push(redirectTo);
    } catch (err) {
      // The backend deliberately returns the same generic message for
      // "no such account" and "wrong password" (see app/api/auth.py's
      // docstring — avoids leaking which registered emails exist), so
      // this just surfaces that message as-is rather than trying to be
      // more specific.
      setError(err instanceof ApiError ? err.message : "Could not sign in. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-sm flex-col justify-center px-6 py-20">
      <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
        Sign in
      </h1>
      <p className="mt-1 text-[13px] text-paper-dim">
        Sign in to upload and manage your own private documents. Browsing the shared library
        doesn&apos;t require an account.
      </p>

      <form onSubmit={submit} className="mt-6 flex flex-col gap-3">
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] text-paper-dim">Email</span>
          <input
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="rounded-md border border-ink-600 bg-ink-800 px-3 py-2 text-[13px] text-paper focus:border-brass-dim focus:outline-none"
          />
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] text-paper-dim">Password</span>
          <input
            type="password"
            required
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="rounded-md border border-ink-600 bg-ink-800 px-3 py-2 text-[13px] text-paper focus:border-brass-dim focus:outline-none"
          />
        </label>

        {error && (
          <div className="flex items-center gap-2 rounded-md border border-rust-dim bg-rust-dim/20 px-3 py-2 text-[12px] text-paper">
            <AlertTriangle size={13} className="shrink-0 text-rust" />
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={busy}
          className="mt-2 flex items-center justify-center gap-2 rounded-lg bg-brass px-4 py-2 text-[13px] font-medium text-ink-950 transition-opacity disabled:opacity-40"
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : <LogIn size={14} />}
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="mt-5 text-center text-[12px] text-muted">
        Don&apos;t have an account?{" "}
        <Link href="/register" className="text-brass-soft hover:text-brass">
          Register
        </Link>
      </p>
    </div>
  );
}
