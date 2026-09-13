/**
 * Token persistence + a tiny pub/sub for "the API just told us our
 * session is invalid" — Phase 5 completion pass, item 6 ("finish
 * frontend authentication... centralize [the Authorization header]
 * instead of manually adding it to every page").
 *
 * **Storage choice, stated explicitly (per that item's requirement to
 * document the tradeoff, not just make it work): `localStorage`, not an
 * httpOnly cookie.**
 *
 * Why: the backend (`app/api/auth.py`) returns the JWT as a JSON body
 * field (`TokenResponse.access_token`), not a `Set-Cookie` header — it
 * was never wired to set/read cookies, and adding that is a backend
 * change this pass's "do not modify the backend unnecessarily" rule
 * (docx item 6) argues against for a token scheme that already works.
 * Given a JSON-body token, the realistic frontend-only options are
 * `localStorage`/`sessionStorage` (readable by any JS on the page — an
 * XSS vulnerability anywhere in the app can exfiltrate the token) or
 * keeping it in memory only (safer, but a full page refresh silently
 * logs the user out, which is a worse experience for a document
 * -library app people keep open in a tab for a while).
 *
 * **Tradeoff accepted**: `localStorage`, because this app has no
 * user-generated HTML rendering path (no rich-text/markdown-from-other
 * -users surface that could inject a script), which is the dominant way
 * XSS reaches `localStorage` in practice. If that changes (e.g. a
 * future "shared notes" feature renders arbitrary user HTML), this
 * choice should be revisited — httpOnly cookies would need a
 * corresponding backend change to `app/api/auth.py` to issue/read them.
 */

const TOKEN_KEY = "veridoc.auth.token";

type UnauthorizedListener = () => void;
const unauthorizedListeners = new Set<UnauthorizedListener>();

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    // Storage disabled (private browsing in some browsers, etc.) —
    // treat as "no session" rather than throwing.
    return null;
  }
}

export function setStoredToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Best-effort; a failed write just means the session won't persist
    // across a reload, not a crash.
  }
}

export function clearStoredToken(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore
  }
}

/** Called by `lib/api.ts`'s `request()` whenever a request comes back
 * `401` while a token was attached — the token is stale/expired/invalid
 * server-side, so every subscriber (just `AuthProvider`, in practice)
 * should clear its in-memory user state and prompt re-login. Kept as a
 * plain pub/sub rather than importing the React context directly so
 * this module has zero React dependency and stays trivially testable.
 */
export function onUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => unauthorizedListeners.delete(listener);
}

export function notifyUnauthorized(): void {
  clearStoredToken();
  for (const listener of unauthorizedListeners) listener();
}
