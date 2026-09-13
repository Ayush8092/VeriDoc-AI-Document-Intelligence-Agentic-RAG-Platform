// Phase 4.1(5): passes a citation list + starting index from an answer's
// evidence stamps into the Source Viewer page without cramming a whole
// citation array (snippets, bboxes, ...) into the URL. `sessionStorage`
// rather than a global store/context: the viewer is a separate route the
// user navigates TO (potentially opened in a new tab via the same
// origin), and this data only ever needs to survive that one handoff —
// it's read once on the viewer's mount and not treated as a durable
// cache.
//
// Not used for anything security-sensitive (citations are already
// visible to this same user in the chat UI they came from), so a plain
// session-scoped token is sufficient — no encryption/signing needed.

import type { Citation } from "./types";

const STORAGE_PREFIX = "veridoc:citation-session:";

export function storeCitationsForViewer(citations: Citation[], startIndex: number): string {
  const token = Math.random().toString(36).slice(2, 10);
  try {
    sessionStorage.setItem(
      `${STORAGE_PREFIX}${token}`,
      JSON.stringify({ citations, index: startIndex }),
    );
  } catch {
    // sessionStorage unavailable (SSR, privacy mode, quota) — the viewer
    // falls back to "no citation context", which it handles by showing
    // an empty state rather than throwing.
  }
  return token;
}

export function loadCitationsForViewer(token: string): { citations: Citation[]; index: number } | null {
  try {
    const raw = sessionStorage.getItem(`${STORAGE_PREFIX}${token}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.citations)) return null;
    return { citations: parsed.citations as Citation[], index: Number(parsed.index) || 0 };
  } catch {
    return null;
  }
}