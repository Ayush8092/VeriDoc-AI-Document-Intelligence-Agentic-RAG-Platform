"""Shared Okapi BM25 scorer.

This is the single BM25 implementation used by two different call sites
that need it for two different purposes:

- `app/lexical_index.py` — BM25 as a first-class *retriever* over the
  whole corpus (independent of Pinecone), so it can be fused with dense
  retrieval (see `app/rag/fusion.py`, hybrid retrieval / Phase 3A).
- `app/reranker.py` — BM25 as a lightweight *reranker* over a small
  candidate set Pinecone already returned.

Both need the identical scoring function; keeping one implementation here
avoids the two drifting apart (see engineering rule "do not introduce
duplicate implementations").

Stdlib only — no extra dependency, no model download, no external API
call. Standard textbook Okapi BM25 defaults (k1=1.5, b=0.75).
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Scorer:
    """A BM25 index over a fixed set of documents, built once and queried many times.

    Building the index (term frequencies, document frequencies, average
    document length) once and reusing it for every query is what makes
    corpus-wide lexical retrieval (`app/lexical_index.py`) practical —
    the old free-function `_bm25_scores` this was factored out of
    recomputed document frequencies on every single call, which is fine
    for reranking a handful of already-retrieved chunks but wasteful for
    scoring an entire corpus per query.
    """

    def __init__(self, doc_token_lists: list[list[str]]):
        self._doc_token_lists = doc_token_lists
        self._n_docs = len(doc_token_lists)
        self._doc_lens = [len(toks) for toks in doc_token_lists]
        self._avg_len = (sum(self._doc_lens) / self._n_docs) if self._n_docs else 0.0

        doc_freq: Counter[str] = Counter()
        for toks in doc_token_lists:
            for term in set(toks):
                doc_freq[term] += 1
        self._doc_freq = doc_freq

        self._term_freqs = [Counter(toks) for toks in doc_token_lists]

    def _idf(self, term: str) -> float:
        n_qualifying = self._doc_freq.get(term, 0)
        return math.log(1 + (self._n_docs - n_qualifying + 0.5) / (n_qualifying + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        """BM25 score of `query_tokens` against every document in the index."""
        if self._n_docs == 0:
            return []
        query_terms = set(query_tokens)
        out: list[float] = []
        for term_freq, doc_len in zip(self._term_freqs, self._doc_lens):
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term)
                if not freq:
                    continue
                numerator = freq * (K1 + 1)
                length_norm = 1 - B + B * (doc_len / self._avg_len if self._avg_len else 1.0)
                denominator = freq + K1 * length_norm
                score += self._idf(term) * (numerator / denominator)
            out.append(score)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs for the top `k` documents, best first.

        Zero-score documents (no term overlap) are excluded — a BM25 score
        of exactly 0 means "no evidence of relevance", not "weakly
        relevant", so they should never be presented as retrieved matches.
        """
        query_tokens = tokenize(query)
        scores = self.scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(i, s) for i, s in ranked if s > 0][:k]


def bm25_scores(query_tokens: list[str], doc_token_lists: list[list[str]]) -> list[float]:
    """One-shot scoring helper (builds a throwaway index). Kept for the reranker's
    small-candidate-set use case, where building a persistent `BM25Scorer` isn't
    worth it — every call reranks a different, disjoint set of chunks anyway.
    """
    return BM25Scorer(doc_token_lists).scores(query_tokens)
