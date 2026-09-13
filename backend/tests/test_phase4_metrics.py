"""Tests for app/evaluation/phase4_metrics.py — every metric here is a
pure function, so these are exact-value assertions against hand-worked
synthetic examples, no mocking/fixtures needed.
"""

from __future__ import annotations

from app.evaluation.phase4_metrics import (
    bbox_correctness,
    bbox_iou,
    caption_association_accuracy,
    chart_detection_accuracy,
    chart_field_accuracy,
    chart_type_accuracy,
    character_error_rate,
    citation_correctness,
    layout_f1,
    object_correctness,
    ordering_accuracy,
    page_assignment_accuracy,
    page_correctness,
    table_detection_accuracy,
    table_structure_accuracy,
    visual_classification_accuracy,
    visual_detection_precision_recall,
    word_error_rate,
)


def test_character_error_rate_exact_match_is_zero():
    assert character_error_rate("hello world", "hello world") == 0.0


def test_character_error_rate_computes_real_edit_distance():
    # "hello" -> "hallo": 1 substitution / 5 reference chars = 0.2
    assert character_error_rate("hallo", "hello") == 0.2


def test_character_error_rate_none_on_empty_reference():
    assert character_error_rate("anything", "") is None


def test_word_error_rate_counts_word_level_edits():
    # "the cat sat" -> "the dog sat": 1 substitution / 3 reference words
    assert word_error_rate("the dog sat", "the cat sat") == round(1 / 3, 4)


def test_bbox_iou_identical_boxes_is_one():
    box = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert bbox_iou(box, box) == 1.0


def test_bbox_iou_no_overlap_is_zero():
    a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    b = {"x0": 20, "y0": 20, "x1": 30, "y1": 30}
    assert bbox_iou(a, b) == 0.0


def test_bbox_iou_partial_overlap():
    a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}   # area 100
    b = {"x0": 5, "y0": 0, "x1": 15, "y1": 10}   # area 100, overlap 5x10=50
    # union = 100+100-50=150, iou=50/150=0.3333
    assert bbox_iou(a, b) == round(50 / 150, 4)


def test_bbox_iou_degenerate_box_is_zero():
    a = {"x0": 0, "y0": 0, "x1": 0, "y1": 10}  # zero width
    b = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert bbox_iou(a, b) == 0.0


def test_layout_f1_perfect_match():
    blocks = [{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "page_number": 1, "block_type": "text"}]
    result = layout_f1(blocks, blocks)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.matched == 1


def test_layout_f1_no_predictions_but_ground_truth_exists():
    gt = [{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "page_number": 1, "block_type": "text"}]
    result = layout_f1([], gt)
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_layout_f1_no_predictions_and_no_ground_truth_is_perfect():
    result = layout_f1([], [])
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_layout_f1_respects_page_mismatch():
    pred = [{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "page_number": 1, "block_type": "text"}]
    gt = [{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "page_number": 2, "block_type": "text"}]
    result = layout_f1(pred, gt, require_same_page=True)
    assert result.matched == 0


def test_layout_f1_greedy_matching_avoids_double_counting():
    # Two predictions both overlap the SAME single ground-truth box --
    # only one should match; the other is an unmatched false positive.
    box = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    pred = [
        {"bbox": box, "page_number": 1, "block_type": "text"},
        {"bbox": box, "page_number": 1, "block_type": "text"},
    ]
    gt = [{"bbox": box, "page_number": 1, "block_type": "text"}]
    result = layout_f1(pred, gt)
    assert result.matched == 1
    assert result.precision == 0.5  # 1 matched / 2 predicted


def test_page_assignment_accuracy():
    assert page_assignment_accuracy([1, 2, 3], [1, 2, 4]) == round(2 / 3, 4)


def test_page_assignment_accuracy_none_on_length_mismatch():
    assert page_assignment_accuracy([1, 2], [1, 2, 3]) is None


def test_ordering_accuracy_perfect_order():
    assert ordering_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_ordering_accuracy_one_swap_partial_credit():
    # ground truth a,b,c -- predicted has b before a (1 of 2 adjacent
    # pairs violated: (a,b) wrong, (b,c) still correct in predicted order)
    result = ordering_accuracy(["b", "a", "c"], ["a", "b", "c"])
    assert result == 0.5


def test_table_detection_accuracy_exact_match():
    assert table_detection_accuracy(2, 2) == 1.0


def test_table_detection_accuracy_zero_ground_truth_zero_predicted():
    assert table_detection_accuracy(0, 0) == 1.0


def test_table_detection_accuracy_partial_credit():
    assert table_detection_accuracy(1, 2) == 0.5


def test_table_structure_accuracy_matching_table():
    table = {
        "rows": [["Item", "Qty"], ["Widget", "4"]],
        "header_row_count": 1,
        "cells": [{"row": 0, "column": 0, "row_span": 1, "col_span": 1}],
        "continuation_group_id": None,
    }
    result = table_structure_accuracy(table, table)
    assert result.row_accuracy == 1.0
    assert result.column_accuracy == 1.0
    assert result.cell_accuracy == 1.0
    assert result.header_accuracy == 1.0
    assert result.merge_accuracy == 1.0
    assert result.continuation_accuracy == 1.0


def test_table_structure_accuracy_wrong_shape():
    pred = {"rows": [["Item", "Qty"]]}
    gt = {"rows": [["Item", "Qty"], ["Widget", "4"]]}
    result = table_structure_accuracy(pred, gt)
    assert result.row_accuracy == 0.0


def test_table_structure_accuracy_merge_mismatch():
    pred = {"rows": [["A"]], "cells": [{"row": 0, "column": 0, "row_span": 1, "col_span": 1}]}
    gt = {"rows": [["A"]], "cells": [{"row": 0, "column": 0, "row_span": 1, "col_span": 2}]}
    result = table_structure_accuracy(pred, gt)
    assert result.merge_accuracy == 0.0


def test_visual_detection_precision_recall():
    box = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    pred = [{"bbox": box, "page_number": 1}]
    gt = [{"bbox": box, "page_number": 1}]
    result = visual_detection_precision_recall(pred, gt)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.matched == 1


def test_visual_classification_accuracy():
    assert visual_classification_accuracy(["chart", "photo"], ["chart", "diagram"]) == 0.5


def test_caption_association_accuracy_treats_both_none_as_correct():
    assert caption_association_accuracy([None, "Figure 1: X"], [None, "Figure 1: X"]) == 1.0


def test_caption_association_accuracy_penalizes_missing_caption():
    assert caption_association_accuracy([None], ["Figure 1: X"]) == 0.0


def test_chart_detection_accuracy():
    assert chart_detection_accuracy([True, False, True], [True, False, False]) == round(2 / 3, 4)


def test_chart_type_accuracy_excludes_unspecified_ground_truth():
    # Second pair has ground_truth=None -> excluded from denominator.
    result = chart_type_accuracy(["bar", "line"], ["bar", None])
    assert result == 1.0


def test_chart_field_accuracy_full_match():
    data = {
        "title": "Quarterly Revenue",
        "x_axis_label": "Quarter",
        "y_axis_label": "USD",
        "legend": ["North", "South"],
        "series": [{"name": "North", "values": [10, 20, 30, 40]}],
    }
    result = chart_field_accuracy(data, data)
    assert result.title_accuracy == 1.0
    assert result.axis_accuracy == 1.0
    assert result.legend_accuracy == 1.0
    assert result.series_accuracy == 1.0
    assert result.value_accuracy == 1.0


def test_chart_field_accuracy_none_when_ground_truth_unspecified():
    result = chart_field_accuracy({"title": "X"}, {"title": None})
    assert result.title_accuracy is None
    assert result.axis_accuracy is None
    assert result.legend_accuracy is None
    assert result.value_accuracy is None


def test_chart_field_accuracy_value_mismatch_partial_credit():
    pred = {"series": [{"name": "North", "values": [10, 99, 30]}]}
    gt = {"series": [{"name": "North", "values": [10, 20, 30]}]}
    result = chart_field_accuracy(pred, gt)
    assert result.value_accuracy == round(2 / 3, 4)


def test_object_correctness():
    assert object_correctness(["obj1", "obj2"], ["obj1", "obj3"]) == 0.5


def test_object_correctness_none_when_not_required():
    assert object_correctness(["obj1"], []) is None


def test_page_correctness():
    assert page_correctness([1, 2], [1, 3]) == 0.5


def test_bbox_correctness_matches_any_overlap():
    box = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert bbox_correctness([box], [box]) == 1.0
    assert bbox_correctness([], [box]) == 0.0


def test_citation_correctness_correct_refusal():
    assert citation_correctness([], [], should_refuse=True) == 1.0


def test_citation_correctness_incorrect_refusal_when_citations_given():
    assert citation_correctness(["c1"], [], should_refuse=True) == 0.0


def test_citation_correctness_f1_overlap():
    # predicted {c1,c2}, ground truth {c1,c3}: precision=1/2, recall=1/2, f1=0.5
    assert citation_correctness(["c1", "c2"], ["c1", "c3"]) == 0.5


def test_citation_correctness_perfect_match():
    assert citation_correctness(["c1", "c2"], ["c1", "c2"]) == 1.0