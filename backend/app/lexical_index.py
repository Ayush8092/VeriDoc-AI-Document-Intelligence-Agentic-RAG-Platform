"""Corpus-wide BM25 lexical retrieval, independent of Pinecone.

`app/reranker.py`'s BM25 only ever re-scores the small candidate set
Pinecone already returned — it can't find a chunk dense retrieval missed
entirely. Proper hybrid retrieval (see `app/rag/fusion.py`, Phase 3A
architecture in the project spec: "Dense retrieval + BM25 retrieval ->
Fusion") needs BM25 as an independent, first-class *retriever* over the
WHOLE corpus, so it can surface lexically-obvious matches (exact SKU
codes, exact dollar figures, exact proper nouns) that a dense embedding
sometimes ranks low.

Persistence: ingestion (`app/services/ingestion_service.py`) writes every
chunk's `chunk_id` + retrievable text + citation metadata to a flat JSON
snapshot (`LEXICAL_INDEX_PATH`, default `data/lexical_index.json`) after
every run. This is deliberately NOT re-derived from Pinecone at query time
(Pinecone is a vector index, not optimized for "give me every stored
document's text back out") and deliberately NOT a second database table —
for a corpus this size a single JSON file that's rewritten wholesale on
every ingestion run is the simplest correct thing that works, and it
keeps this index trivially inspectable (`cat data/lexical_index.json`).
A larger corpus would swap this for a real inverted-index store
(Elasticsearch/OpenSearch, or `rank_bm25`/Whoosh with disk persistence)
without changing this module's public interface.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from app.rag.bm25 import BM25Scorer, tokenize

_SERIALIZED_FIELDS = (
    "chunk_id",
    "source_root",
    "source_file",
    "document_title",
    "section",
    "chunk_index",
    "text",
    "content_hash",
    "block_type",
    "page_start",
    "page_end",
    "source",
    "confidence",
    "table_rows",
    "bbox",
    # Phase 4.1(4) fix: these were already on `Chunk` and already written
    # into Pinecone metadata (`app/vectorstore.py`'s `upsert_chunks`) but
    # missing here, so a BM25-only hit for a figure/chart/visual chunk
    # silently lost object_id/caption/visual_type/chart_type/
    # coordinate_space/related_text the moment it came from the lexical
    # index instead of Pinecone — breaking downstream code (grounding,
    # citation validation, the API response) that expects the SAME shape
    # regardless of which retriever found the chunk (see
    # `app/rag/fusion.py`'s reciprocal_rank_fusion, which merges both
    # retrievers' results into one list with no per-retriever special
    # -casing). The chunk's searchable TEXT already includes caption/
    # chart-title/axis/legend/series/nearby-text/section context (see
    # `app/chunking.py`'s `_visual_chunk` — `related_text` and the
    # section-context `prefix` are baked into `.text` before this
    # snapshot is even written), so BM25 relevance ranking for a query
    # like "what does the revenue chart show" was already working; only
    # the STRUCTURED metadata fields on the returned record were missing.
    "object_id",
    "caption",
    "visual_type",
    "chart_type",
    "coordinate_space",
    "related_text",
    "owner_id",
)


def chunk_to_record(chunk) -> dict:
    """Serialize a `chunking.Chunk` to a plain JSON-able dict."""
    record = {}
    for field_name in _SERIALIZED_FIELDS:
        value = getattr(chunk, field_name, None)
        if field_name == "table_rows" and value is not None:
            value = [list(row) for row in value]
        record[field_name] = value
    return record


def save_snapshot(chunks: list, path: Path) -> int:
    """Write every chunk's retrievable text + metadata to `path` as JSON.

    Overwrites wholesale (not incremental) — the snapshot always reflects
    exactly the current corpus + uploads, same idempotency guarantee
    ingestion already provides for Pinecone (see ingestion_service.py).

    **Atomic replacement (Phase 5 completion pass, round 4, item 5 — "BM25
    production safety").** Writes to a temp file in the SAME directory as
    `path`, then `os.replace()`s it over the real path. `os.replace` is
    atomic on POSIX and on Windows (since Python 3.3) — a concurrent
    reader (`LexicalIndex.from_path`/`get_lexical_index`, called on every
    `/ask` request) either sees the complete old file or the complete new
    one, NEVER a half-written/truncated one. This is the concrete fix for
    a previously-real risk: `delete_document` (`app/api/documents.py`)
    now calls `ingest_corpus()` — which calls this function —
    SYNCHRONOUSLY and DIRECTLY from an API request, at the same time the
    standalone worker (`app.services.ingestion_jobs.worker`) can
    independently be calling it too from processing a QUEUED job; before
    this fix, a request reading the snapshot in the small window between
    `write_text` truncating the file and finishing writing the new bytes
    could get a `json.JSONDecodeError`, silently degraded to an EMPTY
    index by `LexicalIndex.from_path`'s except clause — a reader would
    see zero BM25 results for that one request, not a crash, but silently
    wrong retrieval quality is still a real bug.

    Same-directory temp file (not a global temp dir) is deliberate:
    `os.replace` is only guaranteed atomic when source and destination
    are on the same filesystem — a cross-filesystem "replace" silently
    degrades to copy+delete on some platforms, reintroducing exactly the
    non-atomic window this fix exists to close.

    **What this does NOT solve** (documented honestly, not silently
    assumed away — see the module docstring's "smallest robust
    solution", not a claim of unlimited horizontal scalability): this is
    still a single JSON file on local disk. `docker-compose.yml` runs
    exactly one `worker` replica and mounts a single shared
    `backend-data` volume for both `backend` and `worker`, which is
    sufficient for that topology, but a Docker named volume is host
    -local — this file is NOT safely shared across multiple HOSTS (a
    genuinely horizontally-scaled, multi-machine deployment). Concurrent
    WRITERS (two full `ingest_corpus()` runs racing, e.g. a delete and a
    reingest at nearly the same moment) are now safe from corruption
    (atomic replace means one complete write simply wins), but not
    coordinated — the loser's write is silently superseded rather than
    merged, which is fine here because each `ingest_corpus()` run
    recomputes the FULL correct snapshot from the same underlying source
    of truth (DB + storage), so "last atomic write wins" converges to a
    consistent state either way. A deployment that needs true multi-host
    horizontal scaling of ingestion should move this to PostgreSQL
    -backed lexical metadata instead (see module docstring, "A larger
    corpus would swap this for...") — not attempted here per this
    phase's own instruction not to introduce infrastructure unless
    genuinely required, and this project's single-worker-by-default
    Compose topology doesn't yet require it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [chunk_to_record(c) for c in chunks]
    payload = json.dumps(records, indent=2, ensure_ascii=False)

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())  # ensure bytes are actually on disk before the rename is visible
        os.replace(tmp_path, path)  # atomic on POSIX and Windows (Python >= 3.3)
    except BaseException:
        # Never leave a stray .tmp file behind on a failed write (e.g.
        # disk full mid-write) -- the real snapshot at `path` is
        # untouched either way since we never wrote to it directly.
        Path(tmp_path).unlink(missing_ok=True)
        raise

    return len(records)


class LexicalIndex:
    """A BM25 index built from the persisted chunk snapshot.

    Construction is cheap enough for this project's corpus size to build
    fresh per request (no server-side caching layer yet — see
    `app/core/config.py`'s `LEXICAL_INDEX_CACHE_ENABLED` for the toggle a
    larger deployment would flip on and back with an mtime check).
    """

    def __init__(self, records: list[dict]):
        self._records = records
        self._scorer = BM25Scorer([tokenize(r.get("text", "")) for r in records])

    @classmethod
    def from_path(cls, path: Path) -> "LexicalIndex":
        path = Path(path)
        if not path.exists():
            return cls([])
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls([])
        return cls(records)

    def __len__(self) -> int:
        return len(self._records)

    def search(
        self,
        query: str,
        top_k: int,
        allowed_owner_ids: frozenset[str] | None = None,
        source_files: frozenset[str] | None = None,
    ) -> list[dict]:
        """Return up to `top_k` chunk dicts, best-first, each carrying a
        BM25 `score` and `_retriever: "bm25"` (consumed by `fusion.py` and
        stripped before the value reaches the API/trace).

        `allowed_owner_ids` (Phase 5 tenant isolation): when given, only
        chunks whose `owner_id` (`""`/missing = shared/public corpus,
        matching `app/vectorstore.py`'s convention) is a member are
        eligible — filtered BEFORE truncating to `top_k`, so a request
        always gets its full `top_k` worth of chunks it's actually
        allowed to see (not `top_k` scored across the whole corpus and
        then filtered down to fewer). `None` (the default) applies no
        filtering — every pre-Phase-5 call site is unaffected.

        `source_files` (Phase 6, multi-document scoping — comparison/
        extraction/summarization/report generation all need to restrict
        retrieval to a caller-selected SET of documents, not the whole
        corpus): same filter-before-truncate treatment, same `None`
        -means-no-restriction default. Combined with `allowed_owner_ids`
        as an AND — a chunk must pass BOTH to be eligible, matching how
        `app.retrieval._combine_filters` combines them for the dense/
        Pinecone side.
        """
        if not self._records:
            return []
        if allowed_owner_ids is None and source_files is None:
            matches = self._scorer.top_k(query, top_k)
        else:
            # Score every record (this corpus is small — see class
            # docstring), filter, THEN take the top_k of what's left, so
            # filtering never silently shrinks below top_k when enough
            # allowed candidates exist.
            ranked = self._scorer.top_k(query, len(self._records))

            def _eligible(idx: int) -> bool:
                record = self._records[idx]
                if allowed_owner_ids is not None and (record.get("owner_id") or "") not in allowed_owner_ids:
                    return False
                if source_files is not None and record.get("source_file") not in source_files:
                    return False
                return True

            matches = [(idx, score) for idx, score in ranked if _eligible(idx)][:top_k]
        out = []
        for idx, score in matches:
            record = dict(self._records[idx])
            record["score"] = score
            record["_retriever"] = "bm25"
            record.setdefault("rerank_score", None)
            out.append(record)
        return out


_cached: tuple[str, float, LexicalIndex] | None = None


def get_lexical_index(path: Path, use_cache: bool = True) -> LexicalIndex:
    """Load (and, when enabled, cache-by-mtime) the lexical index for `path`.

    Caching is keyed on the file's modification time, so a fresh
    ingestion run (which rewrites the snapshot file) is always picked up
    on the next call — there is no stale-cache window a running server
    could serve through.
    """
    global _cached
    path = Path(path)
    if not use_cache or not path.exists():
        return LexicalIndex.from_path(path)

    mtime = path.stat().st_mtime
    if _cached is not None and _cached[0] == str(path) and _cached[1] == mtime:
        return _cached[2]

    index = LexicalIndex.from_path(path)
    _cached = (str(path), mtime, index)
    return index