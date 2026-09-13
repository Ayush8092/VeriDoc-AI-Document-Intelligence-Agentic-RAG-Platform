"""Unit tests for app.rag.bm25 (shared scorer) and app.rag.fusion (RRF)."""

from app.rag.bm25 import BM25Scorer, bm25_scores, tokenize
from app.rag.fusion import reciprocal_rank_fusion


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert tokenize("Notice-Period: 60 days!") == ["notice", "period", "60", "days"]


def test_bm25_scores_ranks_matching_document_higher():
    docs = [tokenize("completely unrelated content"), tokenize("notice period is 60 days")]
    scores = bm25_scores(tokenize("notice period"), docs)
    assert scores[1] > scores[0]


def test_bm25_scorer_top_k_excludes_zero_score_documents():
    scorer = BM25Scorer([tokenize("apples and oranges"), tokenize("bananas and pears")])
    results = scorer.top_k("completely unrelated query zzz", k=5)
    assert results == []


def test_bm25_scorer_top_k_orders_best_first():
    scorer = BM25Scorer(
        [
            tokenize("a single mention of notice"),
            tokenize("notice period notice period notice period"),
        ]
    )
    results = scorer.top_k("notice period", k=5)
    assert results[0][0] == 1  # the doc with more term repetition ranks first
    assert results[0][1] > results[1][1]


def test_rrf_promotes_documents_found_by_both_retrievers():
    dense = [
        {"chunk_id": "a", "score": 0.9},
        {"chunk_id": "b", "score": 0.8},
        {"chunk_id": "c", "score": 0.7},
    ]
    bm25 = [
        {"chunk_id": "c", "score": 5.0},
        {"chunk_id": "a", "score": 4.0},
    ]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    fused_ids = [f["chunk_id"] for f in fused]
    # "a" appears near the top of both lists -> should be ranked first.
    assert fused_ids[0] == "a"
    assert set(fused_ids) == {"a", "b", "c"}


def test_rrf_handles_disjoint_lists_without_dropping_either():
    dense = [{"chunk_id": "a", "score": 0.9}]
    bm25 = [{"chunk_id": "b", "score": 5.0}]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert {f["chunk_id"] for f in fused} == {"a", "b"}


def test_rrf_records_retrieval_sources():
    dense = [{"chunk_id": "a", "score": 0.9, "_retriever": "dense"}]
    bm25 = [{"chunk_id": "a", "score": 5.0, "_retriever": "bm25"}]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert fused[0]["retrieval_sources"] == ["bm25", "dense"]
    assert "_retriever" not in fused[0]


def test_rrf_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []]) == []
