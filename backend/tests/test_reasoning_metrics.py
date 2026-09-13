"""Tests for app.evaluation.reasoning_metrics (Phase 7 Part 6 gap-fill —
this module did not exist before this pass)."""

from __future__ import annotations

from app.evaluation.reasoning_metrics import (
    multi_document_reasoning_accuracy,
    query_planning_accuracy,
    table_reasoning_accuracy,
)


# ---------------------------------------------------------------------------
# query_planning_accuracy
# ---------------------------------------------------------------------------


def test_query_planning_accuracy_perfect_match():
    result = query_planning_accuracy(["LOOKUP", "COMPARISON"], ["LOOKUP", "COMPARISON"])
    assert result["accuracy"] == 1.0
    assert result["n"] == 2


def test_query_planning_accuracy_partial_match():
    result = query_planning_accuracy(["LOOKUP", "LOOKUP"], ["LOOKUP", "COMPARISON"])
    assert result["accuracy"] == 0.5


def test_query_planning_accuracy_confusion_breakdown():
    result = query_planning_accuracy(
        ["LOOKUP", "CROSS_DOCUMENT", "LOOKUP"], ["LOOKUP", "COMPARISON", "LOOKUP"]
    )
    assert result["confusion"]["COMPARISON"] == {"CROSS_DOCUMENT": 1}
    assert result["confusion"]["LOOKUP"] == {"LOOKUP": 2}


def test_query_planning_accuracy_unmapped_ground_truth_label_never_counts_correct():
    """A dataset label like FIGURE_QUERY that the classifier's enum
    doesn't even have can never be predicted -> always a miss, not a
    crash."""
    result = query_planning_accuracy(["LOOKUP"], ["FIGURE_QUERY"])
    assert result["accuracy"] == 0.0


def test_query_planning_accuracy_empty_input_returns_none():
    result = query_planning_accuracy([], [])
    assert result["accuracy"] is None
    assert result["n"] == 0


def test_query_planning_accuracy_mismatched_lengths_returns_none():
    result = query_planning_accuracy(["LOOKUP"], ["LOOKUP", "COMPARISON"])
    assert result["accuracy"] is None


def test_query_planning_accuracy_real_classifier_against_real_dataset():
    """Not a synthetic example — runs the REAL classifier against the
    REAL evaluation/datasets/phase4_v1.jsonl query_type ground truth, the
    exact use case evaluation/run_phase4_eval.py exercises."""
    import json
    from pathlib import Path

    from app.rag.query_classifier import classify_query_rule_based

    dataset_path = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "phase4_v1.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    assert len(rows) > 0

    predicted = [classify_query_rule_based(r["question"]).value for r in rows]
    ground_truth = [r["query_type"] for r in rows]

    result = query_planning_accuracy(predicted, ground_truth)
    assert result["n"] == len(rows)
    assert 0.0 <= result["accuracy"] <= 1.0  # a real, computed number, not a placeholder


# ---------------------------------------------------------------------------
# multi_document_reasoning_accuracy
# ---------------------------------------------------------------------------


def test_multi_document_reasoning_accuracy_same_maps_to_both_sided():
    result = multi_document_reasoning_accuracy(["same"], ["both_sided"])
    assert result["accuracy"] == 1.0


def test_multi_document_reasoning_accuracy_different_and_conflicting_map_to_both_sided():
    result = multi_document_reasoning_accuracy(["different", "conflicting"], ["both_sided", "both_sided"])
    assert result["accuracy"] == 1.0


def test_multi_document_reasoning_accuracy_missing_maps_to_one_sided():
    result = multi_document_reasoning_accuracy(["missing_in_a", "missing_in_b"], ["one_sided", "one_sided"])
    assert result["accuracy"] == 1.0


def test_multi_document_reasoning_accuracy_wrong_prediction():
    result = multi_document_reasoning_accuracy(["same"], ["one_sided"])
    assert result["accuracy"] == 0.0


def test_multi_document_reasoning_accuracy_unrecognized_relation_is_a_miss_not_a_crash():
    result = multi_document_reasoning_accuracy(["some_new_relation_type"], ["both_sided"])
    assert result["accuracy"] == 0.0
    assert "unrecognized:some_new_relation_type" in result["confusion"]["both_sided"]


def test_multi_document_reasoning_accuracy_real_dataset_shape():
    """Confirms the metric function accepts the REAL
    evaluation/datasets/multi_doc_v1.jsonl ground-truth shape (per-topic
    values inside expected_relations) without needing any modification —
    doesn't call a live LLM, just validates the data contract."""
    import json
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "multi_doc_v1.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    assert len(rows) > 0

    ground_truth = [v for r in rows for v in r["expected_relations"].values()]
    assert all(v in ("one_sided", "both_sided") for v in ground_truth)

    # A hypothetical perfect predictor for sanity-checking the plumbing.
    fake_predicted = ["same" if v == "both_sided" else "missing_in_a" for v in ground_truth]
    result = multi_document_reasoning_accuracy(fake_predicted, ground_truth)
    assert result["accuracy"] == 1.0
    assert result["n"] == len(ground_truth)


# ---------------------------------------------------------------------------
# table_reasoning_accuracy
# ---------------------------------------------------------------------------


def test_table_reasoning_accuracy_exact_match():
    result = table_reasoning_accuracy([100.0, 50.0], [100.0, 50.0])
    assert result["accuracy"] == 1.0


def test_table_reasoning_accuracy_within_tolerance_counts_correct():
    result = table_reasoning_accuracy([100.4], [100.0], relative_tolerance=0.01)
    assert result["accuracy"] == 1.0


def test_table_reasoning_accuracy_outside_tolerance_counts_wrong():
    result = table_reasoning_accuracy([110.0], [100.0], relative_tolerance=0.01)
    assert result["accuracy"] == 0.0


def test_table_reasoning_accuracy_none_prediction_always_wrong():
    result = table_reasoning_accuracy([None], [100.0])
    assert result["accuracy"] == 0.0


def test_table_reasoning_accuracy_zero_ground_truth_requires_exact_zero():
    assert table_reasoning_accuracy([0.0], [0.0])["accuracy"] == 1.0
    assert table_reasoning_accuracy([0.001], [0.0])["accuracy"] == 0.0


def test_table_reasoning_accuracy_no_ground_truth_returns_none():
    """The actual current state of this metric in this repository: no
    ground-truth dataset exists, so this is the only way it's ever
    reached in practice — must return None (unavailable), never fabricate
    a number."""
    assert table_reasoning_accuracy([100.0], []) is None