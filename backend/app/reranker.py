"""Lightweight lexical reranking of retrieved chunks.

Dense vector search is optimized for recall — it's good at finding chunks
that are broadly about the right topic, but on a small corpus its ordering
can be a poor proxy for "does this chunk actually answer the question"
(cosine scores cluster closely when there are only a handful of short
chunks — see docs/architecture.md, "TOP_K/SCORE_THRESHOLD"). Reranking
re-scores retrieved candidates against the literal words of the question
before the LLM grader ever sees them, improving ordering without
discarding recall.

This is the *lexical* reranker: a small, self-contained Okapi BM25 scorer
(see `app/rag/bm25.py`), stdlib only — no extra dependency, no model
download, no external API call. It is deliberately kept as a cheap,
always-available fallback. As of Phase 3A, `app/rag/cross_encoder.py`
provides a proper transformer cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) that is preferred whenever it's
available (`CROSS_ENCODER_ENABLED=true`, the default) — see
`app/rag/graph.py`'s `_rerank` node, which tries the cross-encoder first
and falls back to this BM25 reranker if the model can't be loaded (no
network, dependency missing, etc.). Keeping this module around means
reranking still works — just with lower semantic precision — in
environments where downloading a transformer model isn't possible.

This module ONLY reorders/selects among chunks already retrieved. It
never generates text, never invents a chunk, and never talks to an LLM —
grounding, citation validation, and answer generation are entirely
unaffected by its presence (see the `rerank` node in `app/rag/graph.py`).
"""

from app.rag.bm25 import bm25_scores, tokenize


def rerank(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Reorder `chunks` by lexical relevance to `question`, keep the best `top_k`.

    - Scoring is BM25 over each chunk's `text` — the same context-rich text
      (document title + front matter + section body; see chunking.py) the
      grader and answer step see, so the reranker's notion of "relevant" is
      grounded in exactly what downstream steps will read.
    - Ties — including "BM25 score is 0 for every candidate", which happens
      when the question shares no meaningful terms with any retrieved chunk
      (the out-of-corpus case) — fall back to preserving the input order,
      which `retrieval.py` already returns best-first by vector similarity.
      Python's `sorted` is stable, so this happens automatically. This
      means an out-of-corpus question still hands the grader the same
      (still irrelevant) candidates it always did — reranking doesn't
      invent relevance that isn't there, so abstention is unaffected.
    - `top_k` truncation only ever removes candidates that BM25 also ranks
      low; it never adds or invents a chunk.
    - Adds a `rerank_score` field to each returned chunk (useful for the
      trace/debugging); every other field from the input chunk is preserved
      unchanged.
    """
    if not chunks:
        return []

    query_tokens = tokenize(question)
    doc_token_lists = [tokenize(c.get("text", "")) for c in chunks]
    scores = bm25_scores(query_tokens, doc_token_lists)

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    reranked = []
    for i in order[:top_k]:
        item = dict(chunks[i])
        item["rerank_score"] = scores[i]
        reranked.append(item)
    return reranked
