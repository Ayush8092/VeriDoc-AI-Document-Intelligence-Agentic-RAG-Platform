"""Unit tests for app.evaluation.retrieval_metrics and app.evaluation.metrics.

All pure functions over plain data — no network, no LLM, no fixtures beyond
literal lists/dicts.
"""

from app.evaluation.metrics import (
    answerable_sensitivity,
    citation_precision_mean,
    citation_recall_mean,
    grounded_answer_rate,
    latency_percentiles,
    refusal_accuracy,
    retrieval_metrics_mean,
    summarize,
    unanswerable_specificity,
)
from app.evaluation.retrieval_metrics import (
    evaluate_ranking,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# retrieval_metrics.py
# ---------------------------------------------------------------------------


def test_recall_at_k_full_match():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5


def test_recall_at_k_respects_k_cutoff():
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
    assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0


def test_recall_at_k_no_ground_truth_is_vacuous_one():
    assert recall_at_k(["a", "b"], [], k=5) == 1.0


def test_mrr_first_hit_rank():
    assert mean_reciprocal_rank(["x", "a", "y"], ["a"]) == 0.5
    assert mean_reciprocal_rank(["a", "x", "y"], ["a"]) == 1.0


def test_mrr_no_hit_is_zero():
    assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mrr_no_ground_truth_is_vacuous_one():
    assert mean_reciprocal_rank(["x", "y"], []) == 1.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    perfect = ndcg_at_k(["a", "b"], ["a", "b"], k=2)
    worse = ndcg_at_k(["x", "a", "b"], ["a", "b"], k=3)
    assert worse < perfect


def test_ndcg_no_ground_truth_is_vacuous_one():
    assert ndcg_at_k(["a", "b"], [], k=2) == 1.0


def test_ndcg_no_relevant_retrieved_is_zero():
    assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_evaluate_ranking_returns_expected_keys():
    """Updated for retrieval_metrics.evaluate_ranking's Phase 7
    completion-pass fix: nDCG is now computed at EVERY k in k_values
    (matching Recall@K/Precision@K/Hit Rate@K's existing per-K
    behavior and the spec's explicit "nDCG@1/3/5/10" requirement), not
    just at max(k_values) — this assertion previously only expected
    `ndcg@3` (the max) and was asserting the PRE-fix, incomplete
    behavior; fixed to match the corrected, spec-compliant output
    rather than weakening the metric.
    """
    result = evaluate_ranking(["a", "b", "c"], ["a"], k_values=(1, 3))
    assert set(result.keys()) == {
        "recall@1",
        "recall@3",
        "precision@1",
        "precision@3",
        "hit_rate@1",
        "hit_rate@3",
        "ndcg@1",
        "ndcg@3",
        "mrr",
    }


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_grounded_answer_rate_only_over_answered_and_graded_results():
    results = [
        {"found": True, "grounded_claim_rate": 0.9},
        {"found": True, "grounded_claim_rate": 0.4},
        {"found": False},  # refusal — excluded
        {"found": True},  # no grounding data — excluded
    ]
    assert grounded_answer_rate(results) == 0.5


def test_grounded_answer_rate_none_when_no_data():
    assert grounded_answer_rate([{"found": True}, {"found": False}]) is None


def test_refusal_accuracy_counts_correct_answer_and_correct_refusal():
    results = [
        {"answerable": True, "found": True},  # correct
        {"answerable": False, "found": False},  # correct refusal
        {"answerable": True, "found": False},  # incorrect (should have answered)
        {"answerable": False, "found": True},  # incorrect (should have refused)
    ]
    assert refusal_accuracy(results) == 0.5


def test_refusal_accuracy_none_when_no_ground_truth():
    assert refusal_accuracy([{"found": True}]) is None


def test_answerable_sensitivity_is_true_positive_rate_among_answerable_only():
    results = [
        {"answerable": True, "found": True},  # answered correctly
        {"answerable": True, "found": False},  # missed
        {"answerable": False, "found": False},  # correctly refused -- excluded from this metric
    ]
    # 1 of 2 answerable questions was actually answered.
    assert answerable_sensitivity(results) == 0.5


def test_answerable_sensitivity_none_when_no_answerable_rows():
    assert answerable_sensitivity([{"answerable": False, "found": False}]) is None


def test_unanswerable_specificity_is_true_negative_rate_among_unanswerable_only():
    results = [
        {"answerable": False, "found": False},  # correctly refused
        {"answerable": False, "found": True},  # hallucinated an answer
        {"answerable": True, "found": True},  # answered correctly -- excluded from this metric
    ]
    # 1 of 2 unanswerable questions was actually refused.
    assert unanswerable_specificity(results) == 0.5


def test_unanswerable_specificity_none_when_no_unanswerable_rows():
    assert unanswerable_specificity([{"answerable": True, "found": True}]) is None


def test_sensitivity_and_specificity_reveal_asymmetry_refusal_accuracy_alone_hides():
    """A system that answers EVERY question: perfect sensitivity, zero
    specificity -- refusal_accuracy alone only reflects the dataset's
    answerable/unanswerable mix, not this asymmetry."""
    always_answers = [
        {"answerable": True, "found": True},
        {"answerable": True, "found": True},
        {"answerable": False, "found": True},  # should have refused, didn't
    ]
    assert answerable_sensitivity(always_answers) == 1.0
    assert unanswerable_specificity(always_answers) == 0.0
    # Blended accuracy (2 correct of 3) looks fine in isolation --
    # exactly the number that would hide the specificity failure above.
    assert refusal_accuracy(always_answers) == round(2 / 3, 4)


def test_citation_precision_mean():
    results = [{"citation_precision": 1.0}, {"citation_precision": 0.5}, {"found": True}]
    assert citation_precision_mean(results) == 0.75


def test_citation_recall_mean_computes_over_required_chunks():
    results = [
        {"required_chunk_ids": ["a", "b"], "cited_chunk_ids": ["a"]},  # 0.5
        {"required_chunk_ids": ["c"], "cited_chunk_ids": ["c"]},  # 1.0
        {"required_chunk_ids": [], "cited_chunk_ids": []},  # excluded (no ground truth)
    ]
    assert citation_recall_mean(results) == 0.75


def test_retrieval_metrics_mean_averages_recall_and_mrr():
    results = [
        {"required_chunk_ids": ["a"], "retrieved_chunk_ids": ["a", "x"]},
        {"required_chunk_ids": ["b"], "retrieved_chunk_ids": ["x", "b"]},
    ]
    out = retrieval_metrics_mean(results, k_values=(1,))
    assert out["recall@1"] == 0.5  # first result hits @1, second misses @1
    assert 0 < out["mrr"] < 1


def test_retrieval_metrics_mean_none_without_ground_truth():
    assert retrieval_metrics_mean([{"found": True}]) is None


def test_latency_percentiles_basic():
    out = latency_percentiles([0.1, 0.2, 0.3, 0.4, 0.5])
    assert out["n"] == 5
    assert out["p50_ms"] == 300.0
    assert out["mean_ms"] == 300.0


def test_latency_percentiles_empty_is_zeroed_not_fabricated():
    out = latency_percentiles([])
    assert out == {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "n": 0}


def test_summarize_reports_null_for_unavailable_metrics():
    results = [{"found": True, "answerable": True}]
    summary = summarize(results)
    assert summary["n_questions"] == 1
    assert summary["grounded_answer_rate"] is None  # no grounded_claim_rate supplied
    assert summary["refusal_accuracy"] == 1.0
    assert "retrieval" not in summary  # no retrieval ground truth supplied


def test_summarize_includes_latency_when_provided():
    summary = summarize([], latencies_seconds=[0.1, 0.2])
    assert "latency" in summary
    assert summary["latency"]["n"] == 2