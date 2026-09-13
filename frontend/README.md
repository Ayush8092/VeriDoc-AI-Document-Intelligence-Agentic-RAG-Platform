# Veridoc Frontend

Next.js (App Router, TypeScript, Tailwind v4) frontend for the Veridoc
backend — see [`../backend`](../backend). Three views:

- **Ask** (`/ask`) — chat interface for grounded Q&A. Every answer shows
  whether it was actually grounded (`found`) and renders its citations as
  "evidence stamps" — expandable cards showing the source file, page
  range, section, a confidence meter (brass for native extraction, teal
  for OCR), and — for table citations — the actual table, not just a text
  snippet.
- **Library** (`/library`) — every ingested document (corpus + uploads)
  with its current version's page count, table count, chunk count, and
  (when OCR ran) a confidence meter and any extraction warnings.
- **Upload** (`/upload`) — drag-and-drop or file-picker upload with
  per-file status (queued → uploading → done/error) and an ingestion
  summary (chunks created, vectors upserted, stale vectors removed) once
  the batch completes.

The sidebar shows a live backend status indicator, polling `GET /ready`
every 20s.

## Design

See the "ink & brass ledger" token system in `app/globals.css` — dark
archival surfaces, brass accent for evidence/native extraction, teal for
OCR/verified state, monospace (IBM Plex Mono) for anything precise and
checkable (page numbers, confidence %, chunk IDs), a serif display face
(Fraunces) for headings. Fonts are self-hosted via `@fontsource` packages
(no runtime calls to Google Fonts).

## Setup

```bash
npm install
cp .env.example .env.local   # points at http://localhost:8000 by default
npm run dev
```

Requires the backend (`../backend`) running and reachable at
`NEXT_PUBLIC_API_BASE_URL`. The backend's default CORS config
(`CORS_ALLOW_ORIGINS`) already allows `http://localhost:3000`.

## Build

```bash
npm run build && npm run start
```

## Notes / current limitations

- No auth — matches the backend, which doesn't have any yet either (see
  backend `MIGRATION_PLAN.md` roadmap).
- The chat is not persisted — refreshing `/ask` clears the conversation.
  Nothing in the backend stores conversation history yet, so there's
  nothing to reload.
- Document deletion isn't wired up — the backend doesn't expose a delete
  endpoint yet (only upload + list).
- `GET /ready` is polled, not pushed — no websocket/SSE layer exists yet,
  so the status indicator has up to a 20s lag after the backend actually
  becomes ready.
