"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { UserPlus, Loader2, AlertTriangle } from "lucide-react";
import { useAuth } from "@/components/auth-provider";
import { ApiError } from "@/lib/api";

const MIN_PASSWORD_LENGTH = 8; // mirrors backend app/schemas/auth.py::RegisterRequest

export default function RegisterPage() {
  const { register } = useAuth();
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tooShort = password.length > 0 && password.length < MIN_PASSWORD_LENGTH;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy || tooShort) return;
    setBusy(true);
    setError(null);
    try {
      await register(email, password);
      router.push("/library");
    } catch (err) {
      // 409 from the backend when the email is already registered (see
      // app/api/auth.py) surfaces here verbatim — registration is the
      // one case where confirming an email exists is expected/fine.
      setError(err instanceof ApiError ? err.message : "Could not create an account. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-sm flex-col justify-center px-6 py-20">
      <h1 className="text-2xl text-paper" style={{ fontFamily: "var(--font-display)" }}>
        Create an account
      </h1>
      <p className="mt-1 text-[13px] text-paper-dim">
        Documents you upload while signed in are private to your account.
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
            minLength={MIN_PASSWORD_LENGTH}
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="rounded-md border border-ink-600 bg-ink-800 px-3 py-2 text-[13px] text-paper focus:border-brass-dim focus:outline-none"
          />
          {tooShort && (
            <span className="text-[11px] text-rust">At least {MIN_PASSWORD_LENGTH} characters.</span>
          )}
        </label>

        {error && (
          <div className="flex items-center gap-2 rounded-md border border-rust-dim bg-rust-dim/20 px-3 py-2 text-[12px] text-paper">
            <AlertTriangle size={13} className="shrink-0 text-rust" />
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={busy || tooShort}
          className="mt-2 flex items-center justify-center gap-2 rounded-lg bg-brass px-4 py-2 text-[13px] font-medium text-ink-950 transition-opacity disabled:opacity-40"
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : <UserPlus size={14} />}
          {busy ? "Creating account…" : "Create account"}
        </button>
      </form>

      <p className="mt-5 text-center text-[12px] text-muted">
        Already have an account?{" "}
        <Link href="/login" className="text-brass-soft hover:text-brass">
          Sign in
        </Link>
      </p>
    </div>
  );
}
