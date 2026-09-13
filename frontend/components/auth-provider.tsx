"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { ApiError, getMe, login as apiLogin, registerAccount } from "@/lib/api";
import { clearStoredToken, getStoredToken, onUnauthorized, setStoredToken } from "@/lib/auth-storage";
import type { UserOut } from "@/lib/types";

interface AuthContextValue {
  /** `undefined` while the initial `/auth/me` check (on mount, for a
   * previously-stored token) hasn't resolved yet — distinct from `null`
   * ("checked, definitely not signed in") so a page can show a neutral
   * loading state instead of flashing a "log in" prompt for a split
   * second on every reload. */
  user: UserOut | null | undefined;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/**
 * Phase 5 completion pass, item 6. Wraps the app once (see
 * app/layout.tsx) and is the single source of truth for "who is signed
 * in" — every page reads it via `useAuth()` rather than re-checking a
 * token itself.
 *
 * Deliberately does NOT block rendering while the initial check is in
 * flight: this app's pages already work fully signed-out (the backend's
 * tenant model treats anonymous requests as the shared public corpus —
 * see app/security/auth.py's `allowed_owner_ids`), so gating the whole
 * UI behind an auth check would misrepresent that on purpose-built
 * anonymous-friendly product. Pages that care show their own
 * "signed in as…" / "sign in to see your private documents" state from
 * `user` instead.
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserOut | null | undefined>(undefined);

  useEffect(() => {
    // Defined INLINE inside the effect (rather than as a separate
    // `useCallback`-wrapped function referenced from here) so
    // `react-hooks/set-state-in-effect` can see this call is
    // effect-local: the lint rule flags a synchronous `setState` call
    // reachable via a function defined OUTSIDE the effect (it can't
    // prove that function's callers are all safe), but accepts the
    // React-docs-endorsed idiom of an async function declared and
    // invoked within the same effect body — same pattern already used
    // by app/source/page.tsx's `resolveCitation`/`resolve` effects.
    // `refreshFromStoredToken` was previously only ever called from
    // this one effect anyway (not part of `AuthContextValue`), so
    // nothing outside loses access to it by inlining it here.
    let cancelled = false;

    async function refreshFromStoredToken() {
      if (!getStoredToken()) {
        if (!cancelled) setUser(null);
        return;
      }
      try {
        const me = await getMe();
        if (!cancelled) setUser(me);
      } catch (err) {
        // A 401 already cleared the stored token via notifyUnauthorized()
        // (see lib/api.ts's request()) and will also fire the
        // onUnauthorized subscription below, which sets user to null —
        // this catch just prevents an unhandled rejection for that
        // expected case. Any other failure (offline backend, etc.) also
        // falls back to "not signed in" rather than leaving `user`
        // perpetually `undefined`.
        if (!cancelled && (!(err instanceof ApiError) || err.status !== 401)) {
          setUser(null);
        }
      }
    }

    refreshFromStoredToken();
    const unsubscribe = onUnauthorized(() => setUser(null));
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const token = await apiLogin(email, password);
    setStoredToken(token.access_token);
    const me = await getMe();
    setUser(me);
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    const token = await registerAccount(email, password);
    setStoredToken(token.access_token);
    const me = await getMe();
    setUser(me);
  }, []);

  const logout = useCallback(() => {
    clearStoredToken();
    setUser(null);
  }, []);

  return <AuthContext.Provider value={{ user, login, register, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth() must be used within <AuthProvider>.");
  return ctx;
}
