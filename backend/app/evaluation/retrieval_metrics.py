"""Retrieval ranking metrics: Recall@K, Precision@K, Hit Rate@K, MRR, nDCG.

Pure functions over plain lists — no network, no database, fully
unit-testable, and reused by both `evaluation/run_ablation.py` (the
retrieval ablation study, Phase 3A requirement #25) and
`app/evaluation/metrics.py` (custom metrics, requirement #39).

Every function takes:
    retrieved_ids: list[str]   — chunk_ids in the order retrieval returned
                                  them, best-first (rank 1 = index 0).
    required_ids:  list[str] | set[str]  — the ground-truth chunk_ids that
                                  were actually needed to answer the
                                  question (from a benchmark dataset's
                                  `required_chunks` field).

and returns a float. Graded relevance (for nDCG) defaults to binary
(1.0 if a chunk_id is in `required_ids`, else 0.0) since the benchmark
dataset format (`evaluation/datasets/v1.jsonl`) only records "required or
not" rather than a graded relevance score — see that file for the schema.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], required_ids: list[str] | set[str], k: int) -> float:
    """Fraction of `required_ids` that appear in the top `k` of `retrieved_ids`.

    Returns 1.0 (vacuously satisfied) when `required_ids` is empty — an
    unanswerable/no-ground-truth question can't fail a recall check that
    doesn't apply to it.
    """
    required = set(required_ids)
    if not required:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(required & top_k) / len(required)


def precision_at_k(retrieved_ids: list[str], required_ids: list[str] | set[str], k: int) -> float:
    """Fraction of the top `k` retrieved chunks that are actually
    required (Phase 7 spec item 9 — distinct from recall: recall asks
    "did we find everything we needed", precision asks "how much of
    what we returned was actually needed").

    Returns 1.0 (vacuous) when `required_ids` is empty, same convention
    as `recall_at_k`. Returns 0.0 when `k` results were requested but
    none were retrieved (an empty `retrieved_ids[:k]`) — no wasted slots
    means no numerator, but also no denominator to call it "vacuously
    precise";0.0 is the honest "there was nothing to be precise about,
    and nothing correct was returned either" answer for a metric this
    project reports as an achieved fraction, not a defined-when-applicable
    one like recall/MRR/nDCG are for the "no ground truth" case.
    """
    required = set(required_ids)
    if not required:
        return 1.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in required)
    return hits / len(top_k)


def hit_rate_at_k(retrieved_ids: list[str], required_ids: list[str] | set[str], k: int) -> float:
    """1.0 if AT LEAST ONE required chunk appears in the top `k`, else
    0.0 (a.k.a. Success@K). Distinct from recall@k, which is a fraction
    of required chunks found — hit rate only asks a yes/no question,
    which is what most "did retrieval succeed at all for this question"
    reporting actually wants aggregated across many questions (the mean
    of this over a benchmark IS the classic "Hit Rate" metric).

    Returns 1.0 (vacuous) when `required_ids` is empty, same convention
    as every other function in this module.
    """
    required = set(required_ids)
    if not required:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if required & top_k else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], required_ids: list[str] | set[str]) -> float:
    """Reciprocal rank of the FIRST retrieved chunk that is in `required_ids`.

    0.0 if none of the required chunks were retrieved at all. 1.0
    (vacuous) when there is no ground truth to rank against.
    """
    required = set(required_ids)
    if not required:
        return 1.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in required:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    required_ids: list[str] | set[str],
    k: int,
    relevance: dict[str, float] | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain at rank `k`.

    `relevance` optionally supplies graded relevance per chunk_id (e.g.
    2.0 for "the exact answer chunk", 1.0 for "helpful context"); any
    chunk_id not in `relevance` (or when `relevance` is omitted entirely)
    falls back to binary relevance: 1.0 if it's in `required_ids`, else
    0.0.
    """
    required = set(required_ids)
    if not required:
        return 1.0

    def rel(chunk_id: str) -> float:
        if relevance is not None and chunk_id in relevance:
            return relevance[chunk_id]
        return 1.0 if chunk_id in required else 0.0

    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids[:k], start=1):
        gain = rel(chunk_id)
        if gain:
            dcg += gain / math.log2(i + 1)

    ideal_gains = sorted((rel(cid) for cid in required), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal_gains, start=1) if gain)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(
    retrieved_ids: list[str],
    required_ids: list[str] | set[str],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Convenience wrapper: Recall@K, Precision@K, Hit Rate@K, and
    nDCG@K for EVERY k in `k_values` (Phase 7 completion pass: an
    earlier version only computed nDCG at `max(k_values)` — a real gap
    against the spec's explicit "nDCG@1/3/5/10" requirement, not just
    "nDCG@K" for one K — fixed here to match Recall/Precision/Hit Rate's
    existing per-K behavior), plus MRR (which has no K — it's the
    reciprocal rank of the first relevant hit regardless of cutoff)."""
    out = {}
    for k in k_values:
        out[f"recall@{k}"] = round(recall_at_k(retrieved_ids, required_ids, k), 4)
        out[f"precision@{k}"] = round(precision_at_k(retrieved_ids, required_ids, k), 4)
        out[f"hit_rate@{k}"] = round(hit_rate_at_k(retrieved_ids, required_ids, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, required_ids, k), 4)
    out["mrr"] = round(mean_reciprocal_rank(retrieved_ids, required_ids), 4)
    return out