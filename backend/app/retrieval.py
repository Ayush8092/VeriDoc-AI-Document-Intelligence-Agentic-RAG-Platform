"""Dense retrieval: embed query/queries, search Pinecone, filter, dedupe.

Two entry points:

- `retrieve_single` — one query in, embed once, search once. This is the
  normal path used on the first attempt for every question, so we don't pay
  for multiple embedding + search calls when a single query is enough
  (the corpus is small and free-tier quota matters — see docs/architecture.md).
- `retrieve_pooled` — several queries in (used only on the bounded retry
  after the first attempt is graded insufficient). Each query is searched
  independently and results are pooled and deduplicated by chunk_id, keeping
  the best score a chunk earned across queries.

`SCORE_THRESHOLD` is a retrieval-time recall filter, not proof that a chunk
answers the question — that judgment is left to `llm.grade_chunks`, which is
what actually gates whether we generate an answer or retry (Improvement 4 /
"do not rely exclusively on similarity score for grounding").
"""

from pathlib import Path

from app import lexical_index, vectorstore
from app.rag.fusion import reciprocal_rank_fusion


def _search(index, embeddings, settings, query: str, metadata_filter: dict | None = None) -> list[dict]:
    vector = embeddings.embed_query(query)
    return vectorstore.query(index, settings, vector, settings.top_k, metadata_filter=metadata_filter)


def retrieve_single(index, embeddings, settings, query: str, metadata_filter: dict | None = None) -> list[dict]:
    """Embed and search with a single query; apply the score floor."""
    matches = _search(index, embeddings, settings, query, metadata_filter=metadata_filter)
    return [m for m in matches if m["score"] >= settings.score_threshold]


def retrieve_pooled(
    index, embeddings, settings, queries: list[str], metadata_filter: dict | None = None
) -> list[dict]:
    """Search with several queries, pool, dedupe by chunk_id, filter by score.

    Keeps the highest score seen for a chunk across all queries, then returns
    results sorted best-first.
    """
    pooled: dict[str, dict] = {}
    for query in queries:
        for match in _search(index, embeddings, settings, query, metadata_filter=metadata_filter):
            if match["score"] < settings.score_threshold:
                continue
            existing = pooled.get(match["chunk_id"])
            if existing is None or match["score"] > existing["score"]:
                pooled[match["chunk_id"]] = match
    return sorted(pooled.values(), key=lambda m: m["score"], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid (dense + BM25) retrieval — Phase 3A
# ---------------------------------------------------------------------------
#
# The dense-only functions above (`retrieve_single`/`retrieve_pooled`) are
# kept unchanged and still used directly whenever `HYBRID_RETRIEVAL_ENABLED`
# is false, or in any context (tests, ingestion tooling) that only needs
# plain dense search. The functions below wrap them with an independent
# BM25 pass over the whole corpus (see `app/lexical_index.py`) and fuse the
# two rank-ordered lists with Reciprocal Rank Fusion (`app/rag/fusion.py`)
# — this is what actually implements the architecture the project spec
# calls for: "Dense retrieval + BM25 retrieval -> Fusion -> Candidate set".


def _resolve_lexical_index(settings) -> lexical_index.LexicalIndex:
    path = Path(settings.lexical_index_path)
    if not path.is_absolute():
        # ingestion_service.py resolves relative paths against the backend
        # project root the same way; mirrored here so retrieval finds the
        # same file regardless of the process's current working directory.
        path = Path(__file__).resolve().parent.parent / path
    return lexical_index.get_lexical_index(path, use_cache=settings.lexical_index_cache_enabled)


def _tag_dense(matches: list[dict]) -> list[dict]:
    for m in matches:
        m["_retriever"] = "dense"
    return matches


def _owner_filter(allowed_owner_ids: frozenset[str] | None) -> dict | None:
    """Build the Pinecone metadata filter for `allowed_owner_ids` (Phase 5
    tenant isolation). `None` means "no restriction" (every pre-Phase-5
    caller) — NOT "restrict to nothing"; that distinction matters, so this
    is a dedicated helper rather than inlined at each call site where the
    two could be confused.
    """
    if allowed_owner_ids is None:
        return None
    return {"owner_id": {"$in": sorted(allowed_owner_ids)}}


def _combine_filters(
    allowed_owner_ids: frozenset[str] | None, source_files: frozenset[str] | None
) -> dict | None:
    """Build the combined Pinecone metadata filter for tenant scoping
    (`allowed_owner_ids`) AND document scoping (`source_files`, Phase 6 —
    comparison/extraction/summarization/report generation all need to
    restrict retrieval to a caller-selected set of documents). Both are
    optional and independent; `None`+`None` returns `None` (no filter at
    all), matching every pre-Phase-6 call.
    """
    filter_dict: dict = {}
    if allowed_owner_ids is not None:
        filter_dict["owner_id"] = {"$in": sorted(allowed_owner_ids)}
    if source_files is not None:
        filter_dict["source_file"] = {"$in": sorted(source_files)}
    return filter_dict or None


def hybrid_retrieve_single(
    index,
    embeddings,
    settings,
    query: str,
    allowed_owner_ids: frozenset[str] | None = None,
    source_files: frozenset[str] | None = None,
) -> dict:
    """Dense + BM25 fusion for a single query (the normal first-attempt path).

    Returns the full breakdown — `{"dense": [...], "bm25": [...],
    "fused": [...]}` — not just the fused list, so callers that need to
    report ranking metrics (Recall@K/MRR/nDCG per retriever stage, Phase
    3A requirement #24) have the actual per-stage candidate lists instead
    of only the final merged result. `fused` is what the rest of the RAG
    pipeline (`app/rag/graph.py`) actually retrieves/reranks with; `dense`
    and `bm25` are the two inputs that went into it, each already
    score-sorted best-first.

    `allowed_owner_ids` (Phase 5 tenant isolation): restricts BOTH
    retrievers to chunks owned by the shared corpus or the requesting
    user — see `_combine_filters` and `LexicalIndex.search`. `None` (the
    default) applies no restriction, matching every pre-Phase-5 call.

    `source_files` (Phase 6 multi-document scoping): restricts BOTH
    retrievers to chunks from a caller-selected set of documents — see
    `app.rag.multi_doc` for how comparison/extraction/summarization/
    report generation resolve a list of `document_id`s to this set.
    `None` (the default) applies no restriction.
    """
    metadata_filter = _combine_filters(allowed_owner_ids, source_files)
    dense = _tag_dense(_search(index, embeddings, settings, query, metadata_filter=metadata_filter))
    dense = [m for m in dense if m["score"] >= settings.score_threshold]

    if not settings.hybrid_retrieval_enabled:
        return {"dense": dense, "bm25": [], "fused": dense}

    lex = _resolve_lexical_index(settings)
    bm25_matches = lex.search(
        query, settings.bm25_top_k, allowed_owner_ids=allowed_owner_ids, source_files=source_files
    )
    fused = reciprocal_rank_fusion([dense, bm25_matches], k=settings.rrf_k)[: settings.top_k]
    return {"dense": dense, "bm25": bm25_matches, "fused": fused}


def hybrid_retrieve_pooled(
    index,
    embeddings,
    settings,
    queries: list[str],
    allowed_owner_ids: frozenset[str] | None = None,
    source_files: frozenset[str] | None = None,
) -> dict:
    """Dense + BM25 fusion pooled across several queries (the bounded-retry path).

    Same return shape as `hybrid_retrieve_single` — see its docstring,
    including the `allowed_owner_ids`/`source_files` scoping parameters.
    """
    metadata_filter = _combine_filters(allowed_owner_ids, source_files)
    dense = retrieve_pooled(index, embeddings, settings, queries, metadata_filter=metadata_filter)
    dense = _tag_dense(dense)

    if not settings.hybrid_retrieval_enabled:
        return {"dense": dense, "bm25": [], "fused": dense}

    lex = _resolve_lexical_index(settings)
    bm25_pooled: dict[str, dict] = {}
    for query in queries:
        for match in lex.search(
            query, settings.bm25_top_k, allowed_owner_ids=allowed_owner_ids, source_files=source_files
        ):
            existing = bm25_pooled.get(match["chunk_id"])
            if existing is None or match["score"] > existing["score"]:
                bm25_pooled[match["chunk_id"]] = match
    bm25_matches = sorted(bm25_pooled.values(), key=lambda m: m["score"], reverse=True)

    fused = reciprocal_rank_fusion([dense, bm25_matches], k=settings.rrf_k)[: settings.top_k]
    return {"dense": dense, "bm25": bm25_matches, "fused": fused}