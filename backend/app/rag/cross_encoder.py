"""Cross-Encoder reranking (Phase 3A hybrid retrieval upgrade).

`app/reranker.py`'s BM25 reranker only sees literal term overlap. A
cross-encoder jointly encodes (question, chunk) pairs through a small
transformer and outputs a genuine relevance score — it catches paraphrase
and synonym matches BM25 structurally cannot ("How much notice to end the
contract?" / "termination requires 60 days' written notice" share almost
no tokens but are obviously about the same thing).

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — a small (~80MB), widely
used, MS MARCO-trained pair-relevance model via `sentence-transformers`.
Chosen deliberately over a larger model: it's small enough to run on CPU
with acceptable latency for a handful of candidates, which matters because
this reranker sits on the synchronous request path.

Failure handling is the important part of this module: `sentence-
transformers` is an optional/heavy dependency (`CROSS_ENCODER_ENABLED` can
be turned off, and the package or model weights may simply not be present
— no network access, disk space, or the dependency was never installed).
Every failure mode here degrades to `None` (never raises), and the caller
(`app/rag/graph.py`'s `_rerank` node) falls back to the always-available
BM25 reranker. This means cross-encoder reranking is a pure quality
upgrade when available, never a hard dependency for the system to answer
questions at all.
"""

from __future__ import annotations

import logging
import threading

_log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_load_failed = False


def _load_model(model_name: str):
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _model_lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415 (lazy, optional dep)

            _model = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 — any failure means "unavailable"
            _log.warning(
                "Cross-encoder model '%s' unavailable (%s: %s); falling back to the "
                "BM25 lexical reranker for this process.",
                model_name,
                type(exc).__name__,
                exc,
            )
            _load_failed = True
            return None
    return _model


def is_available(model_name: str) -> bool:
    return _load_model(model_name) is not None


def rerank(question: str, chunks: list[dict], top_k: int, model_name: str) -> list[dict] | None:
    """Rerank `chunks` with a cross-encoder; return `None` if unavailable.

    On success, returns the best `top_k` chunks sorted by cross-encoder
    score, each carrying a `rerank_score` (cosine-BM25-style field name
    kept identical to `app/reranker.py` so downstream code and traces
    don't need to know which reranker ran) and `reranker: "cross_encoder"`.
    Never invents or drops a chunk's other fields.
    """
    if not chunks:
        return []
    model = _load_model(model_name)
    if model is None:
        return None

    try:
        pairs = [(question, c.get("text", "")) for c in chunks]
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001 — a runtime predict failure also falls back
        _log.warning("Cross-encoder predict() failed (%s: %s); falling back to BM25.", type(exc).__name__, exc)
        return None

    order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    reranked = []
    for i in order[:top_k]:
        item = dict(chunks[i])
        item["rerank_score"] = float(scores[i])
        item["reranker"] = "cross_encoder"
        reranked.append(item)
    return reranked
