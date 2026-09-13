"""Hybrid retrieval fusion: combine dense (Pinecone) and lexical (BM25)
result lists into one ranked candidate set.

Uses Reciprocal Rank Fusion (RRF) — `score(d) = sum(1 / (k + rank_i(d)))`
over every ranked list `i` the document appears in — rather than trying to
combine dense cosine similarity and BM25 scores directly. The two are on
incompatible scales (cosine in [0,1]-ish, BM25 unbounded and corpus-size
dependent), so naively summing or averaging them is a common but subtly
broken pattern; RRF only uses each list's *rank order*, which sidesteps
the scale mismatch entirely. `k=60` is the standard default from the
original RRF paper (Cormack et al., 2009) and the value most hybrid-search
implementations (Elasticsearch, Weaviate, Pinecone's own hybrid docs) ship
as their default, so it's a reasonable, well-precedented starting point
rather than an untuned magic number.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = DEFAULT_RRF_K,
    id_key: str = "chunk_id",
) -> list[dict]:
    """Fuse several rank-ordered lists of chunk dicts into one, best-first.

    Each input list must already be sorted best-first (rank 1 = index 0).
    A chunk that appears in more than one list accumulates score from
    each; its output dict is a shallow merge of every occurrence (later
    lists' fields win on conflict, except `chunk_id`) plus a `fusion_score`
    and a `retrieval_sources` list naming which retriever(s) it came from,
    so downstream (grading, tracing, evaluation) can see whether a chunk
    survived because dense search, BM25, or both found it.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}

    for list_index, ranked in enumerate(ranked_lists):
        source_name = ranked[0].get("_retriever", f"list_{list_index}") if ranked else f"list_{list_index}"
        for rank, item in enumerate(ranked, start=1):
            cid = item[id_key]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            merged.setdefault(cid, {}).update(item)
            sources.setdefault(cid, set()).add(item.get("_retriever", source_name))

    fused = []
    for cid, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        item = dict(merged[cid])
        item.pop("_retriever", None)
        item["fusion_score"] = round(score, 6)
        item["retrieval_sources"] = sorted(sources[cid])
        fused.append(item)
    return fused
