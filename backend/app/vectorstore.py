"""Pinecone read/write helpers: upsert, query, reset, stats.

`clients.py` owns *connecting*; this module owns *what we do* with the
index once connected. Kept separate so ingestion (writes) and retrieval
(reads) both go through the same, tested upsert/query shape.
"""

from app.chunking import Chunk

UPSERT_BATCH_SIZE = 100


def delete_namespace(pc, settings) -> None:
    """Wipe the configured namespace. Used by `python -m app.ingestion --reset`."""
    index = pc.Index(settings.pinecone_index_name)
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
    except Exception:
        # A brand-new index has no namespace yet — nothing to delete.
        pass


def upsert_chunks(
    index, settings, chunks: list[Chunk], vectors: list[list[float]]
) -> int:
    """Upsert chunks with their embeddings.

    Chunk IDs are deterministic (see chunking.py), so re-running ingestion
    overwrites the same Pinecone records instead of creating duplicates —
    this is what makes ingestion idempotent.
    """
    import json

    records = [
        {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                "chunk_id": chunk.chunk_id,
                "source_root": chunk.source_root,
                "source_file": chunk.source_file,
                "document_title": chunk.document_title,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "text": chunk.text,
                "block_type": chunk.block_type,
                "page_start": chunk.page_start if chunk.page_start is not None else -1,
                "page_end": chunk.page_end if chunk.page_end is not None else -1,
                "source": chunk.source,
                "confidence": chunk.confidence,
                "table_json": json.dumps(chunk.table_rows) if chunk.table_rows else "",
                "bbox_json": json.dumps(chunk.bbox) if chunk.bbox else "",
                "chart_data_json": json.dumps(chunk.chart_data) if chunk.chart_data else "",
                "object_id": chunk.object_id or "",
                "caption": chunk.caption or "",
                "visual_type": chunk.visual_type or "",
                "chart_type": chunk.chart_type or "",
                "coordinate_space": chunk.coordinate_space or "",
                "related_text": chunk.related_text or "",
                # Phase 5 tenant isolation: "" means shared/public corpus
                # (matches every other optional-string convention in this
                # metadata dict) — see Chunk.owner_id's docstring.
                "owner_id": chunk.owner_id or "",
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(
            vectors=records[start : start + UPSERT_BATCH_SIZE],
            namespace=settings.pinecone_namespace,
        )
    return len(records)


def _match_to_result(m, default_score: float = 0.0) -> dict:
    """Shared match -> plain-dict shape for both `query()` (vector search)
    and `query_by_filter()` (metadata-only lookup, Phase 4.1(7) source
    viewer) — factored out so a field added to one (e.g. the Phase 4.1(4)
    object_id/bbox/visual_type/chart_type/coordinate_space/related_text
    fix below) can never silently apply to only one of the two read paths
    again.
    """
    import json

    meta = m["metadata"] if isinstance(m, dict) else m.metadata
    mid = m["id"] if isinstance(m, dict) else m.id
    score = (m["score"] if isinstance(m, dict) else getattr(m, "score", None)) if m is not None else None
    page_start = meta.get("page_start", -1)
    page_end = meta.get("page_end", -1)
    table_json = meta.get("table_json") or ""
    bbox_json = meta.get("bbox_json") or ""
    chart_data_json = meta.get("chart_data_json") or ""
    return {
        "chunk_id": meta.get("chunk_id", mid),
        "score": float(score) if score is not None else default_score,
        "source_root": meta.get("source_root", ""),
        "source_file": meta.get("source_file", ""),
        "document_title": meta.get("document_title", ""),
        "section": meta.get("section", ""),
        "text": meta.get("text", ""),
        "block_type": meta.get("block_type", "text"),
        "page_start": None if page_start in (-1, None) else int(page_start),
        "page_end": None if page_end in (-1, None) else int(page_end),
        "source": meta.get("source", "native"),
        "confidence": float(meta.get("confidence", 1.0)),
        "table_rows": json.loads(table_json) if table_json else None,
        "bbox": json.loads(bbox_json) if bbox_json else None,
        "chart_data": json.loads(chart_data_json) if chart_data_json else None,
        # Phase 4.1(4) fix: these were already written by upsert_chunks
        # above but never read back here, so every visual chunk lost its
        # object_id/caption/visual_type/chart_type/coordinate_space/
        # related_text the moment it round-tripped through Pinecone —
        # reconstructing them from `""` back to `None` (Pinecone metadata
        # can't store a bare null, so upsert_chunks wrote `""` for
        # "absent" — same convention `bbox_json`/`table_json` already use).
        "object_id": meta.get("object_id") or None,
        "caption": meta.get("caption") or None,
        "visual_type": meta.get("visual_type") or None,
        "chart_type": meta.get("chart_type") or None,
        "coordinate_space": meta.get("coordinate_space") or None,
        "related_text": meta.get("related_text") or None,
        "owner_id": meta.get("owner_id") or None,
    }


def query(index, settings, vector: list[float], top_k: int, metadata_filter: dict | None = None) -> list[dict]:
    """Query the index; returns matches as plain dicts with score + metadata.

    `metadata_filter` (Phase 5 tenant isolation): a Pinecone filter dict,
    e.g. `{"owner_id": {"$in": ["", "42"]}}` — restricts results to
    chunks owned by the public corpus or the requesting user. `None`
    (the default) applies NO filter, i.e. every pre-Phase-5 call site is
    unaffected. See `app/retrieval.py`'s `hybrid_retrieve_single`/
    `hybrid_retrieve_pooled`, which are what actually build this filter
    from a request's `allowed_owner_ids`.
    """
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
        **({"filter": metadata_filter} if metadata_filter else {}),
    )
    matches = result.get("matches") if isinstance(result, dict) else result.matches
    return [_match_to_result(m) for m in matches]


def query_by_filter(index, settings, metadata_filter: dict, top_k: int = 1000) -> list[dict]:
    """Metadata-only lookup (Phase 4.1(7), source viewer): every chunk
    matching `metadata_filter`, with no relevance ranking involved.

    Used by `app.api.source`'s "objects on this page" endpoint, which
    needs "every table/figure/chart chunk whose page range covers page
    N" — not a similarity search. Pinecone's query API always requires a
    `vector`; a zero vector is the standard pattern for a metadata
    -filter-only query (every candidate gets the same, uninformative
    cosine similarity, so ranking is meaningless and callers should not
    read `score` from these results — the source viewer doesn't).

    Reuses `_match_to_result` (see its docstring) so this NEVER drifts
    out of sync with `query()`'s field set — the whole point of adding
    this function is to expose the SAME provenance fields `query()`
    already fixed in Phase 4.1(4), for a different access pattern, not a
    parallel, easier-to-forget-fields read path.
    """
    zero_vector = [0.0] * settings.embedding_dimension
    result = index.query(
        vector=zero_vector,
        top_k=top_k,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
        filter=metadata_filter,
    )
    matches = result.get("matches") if isinstance(result, dict) else result.matches
    return [_match_to_result(m) for m in matches]


def fetch_existing_hashes(index, settings, chunk_ids: list[str]) -> dict[str, str]:
    """Look up `content_hash` metadata for chunk_ids already stored in Pinecone.

    Used by ingestion to skip re-embedding chunks whose content hasn't
    changed since the last run (Improvement 5) — the corpus is small enough
    that this isn't required for correctness, but it does cut Gemini
    embedding calls (and free-tier quota) on repeat ingestion runs where
    only one or two documents changed. `index.fetch` is read-only and safe
    to call even on an empty/brand-new namespace.

    Returns {chunk_id: content_hash} for whichever of the requested IDs
    already exist; missing IDs are simply absent from the result (treated
    as "new" by the caller).
    """
    if not chunk_ids:
        return {}
    hashes: dict[str, str] = {}
    for start in range(0, len(chunk_ids), UPSERT_BATCH_SIZE):
        batch = chunk_ids[start : start + UPSERT_BATCH_SIZE]
        try:
            result = index.fetch(ids=batch, namespace=settings.pinecone_namespace)
        except Exception:
            # Brand-new index/namespace, or a transient fetch error — treat
            # every chunk in this batch as new rather than failing ingestion.
            continue
        vectors = result.get("vectors") if isinstance(result, dict) else result.vectors
        for chunk_id, record in (vectors or {}).items():
            meta = record["metadata"] if isinstance(record, dict) else record.metadata
            content_hash = (meta or {}).get("content_hash")
            if content_hash:
                hashes[chunk_id] = content_hash
    return hashes


def vector_count(pc, settings) -> int:
    """Number of vectors currently stored in the configured namespace."""
    stats = pc.Index(settings.pinecone_index_name).describe_index_stats()
    namespaces = stats.get("namespaces") if isinstance(stats, dict) else stats.namespaces
    ns = (namespaces or {}).get(settings.pinecone_namespace)
    if ns is None:
        return 0
    return int(ns["vector_count"] if isinstance(ns, dict) else ns.vector_count)


def list_all_ids(index, settings) -> set[str]:
    """All vector IDs currently stored in the configured namespace.

    Chunk IDs and Pinecone vector IDs are the same string (see
    `upsert_chunks`: `"id": chunk.chunk_id`), so the result can be compared
    directly against the corpus's current chunk IDs to find stale vectors
    left behind by a deleted file or a restructured section — see
    `app/ingestion.py`'s reconciliation step.

    Scoped to `settings.pinecone_namespace` only, via the same
    `index.list(namespace=...)` call `delete_namespace` already uses for
    `--reset` — this project treats its configured namespace as exclusively
    its own (see `delete_namespace`), so every ID this returns is safe to
    reconcile against the current corpus. Nothing here ever touches another
    namespace or another index.

    Pinecone client versions/plans differ in whether `index.list(...)` is
    available (it requires a serverless index). If the installed client
    doesn't support it, this degrades to "nothing known" and prints a
    warning rather than raising — stale-vector cleanup is then skipped for
    this run, but insert/upsert/skip ingestion keeps working exactly as
    before this feature existed.
    """
    if not hasattr(index, "list"):
        print(
            "[ingestion] WARNING: this Pinecone client/index does not "
            "support index.list(); skipping stale-vector cleanup this run."
        )
        return set()

    ids: set[str] = set()
    try:
        for batch in index.list(namespace=settings.pinecone_namespace):
            ids.update(batch)
    except Exception as exc:
        print(
            f"[ingestion] WARNING: could not list existing vector IDs "
            f"({type(exc).__name__}: {exc}); skipping stale-vector cleanup this run."
        )
        return set()
    return ids


def delete_vectors(index, settings, chunk_ids: list[str]) -> int:
    """Delete specific vector IDs from the configured namespace, in batches.

    Always scoped to `settings.pinecone_namespace` and never `delete_all` —
    this can only ever remove vector IDs the caller explicitly names, never
    an entire namespace or index. `app/ingestion.py` is the only caller,
    and only ever passes IDs computed as `existing - current_chunk_ids`
    (see `list_all_ids`), so this never deletes a vector still produced by
    the current corpus.
    """
    if not chunk_ids:
        return 0
    for start in range(0, len(chunk_ids), UPSERT_BATCH_SIZE):
        batch = chunk_ids[start : start + UPSERT_BATCH_SIZE]
        index.delete(ids=batch, namespace=settings.pinecone_namespace)
    return len(chunk_ids)