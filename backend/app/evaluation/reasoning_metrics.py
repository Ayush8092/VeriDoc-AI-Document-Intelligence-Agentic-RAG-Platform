"""Reasoning accuracy metrics (Phase 7 metrics-layer completion — Part 6
audit finding: these were entirely absent before this pass; every other
metric category — retrieval, application/grounding, RAGAS, TruLens,
Phase 4 multimodal — had at least a module; this one didn't exist).

Three metrics, three different availability stories — stated honestly
rather than uniformly claimed "done":

- `query_planning_accuracy`: **genuinely computable right now.**
  `evaluation/datasets/phase4_v1.jsonl` carries a real `query_type`
  ground-truth label per question (independently authored when the row
  was written, not derived from the classifier), so this can be — and
  is, by `evaluation/run_phase4_eval.py` — measured for real against
  `app.rag.query_classifier.classify_query_rule_based`.

- `multi_document_reasoning_accuracy`: **the metric function is real and
  tested**, comparing a predicted per-topic relation (from
  `app.rag.comparison.compare_documents`'s real output shape:
  `"same"`/`"different"`/`"missing_in_a"`/`"missing_in_b"`/`"conflicting"`)
  against `evaluation/datasets/multi_doc_v1.jsonl`'s coarser
  `"one_sided"`/`"both_sided"` ground truth. Actually RUNNING it end-to-
  end needs a live LLM (`compare_documents` calls one) — see
  `evaluation/run_ablation.py`'s Config G (`run_multi_doc_ablation`) for
  where that would be wired in; this module supplies the scoring
  function that step needs, it does not itself call any LLM.

- `table_reasoning_accuracy`: **the metric function is real and tested**
  (numeric comparison with a relative-tolerance band, since a table
  computation like "average revenue" can differ from ground truth by
  float rounding without being wrong), but there is currently NO ground-
  truth dataset of (table, operation, expected value) triples anywhere
  in this repository to run it against — `evaluation/datasets/*.jsonl`
  records citation/chunk/page ground truth, not table computation
  results. Marked UNAVAILABLE for lack of ground truth, not fabricated —
  see `evaluation/README.md`'s metrics table.
"""

from __future__ import annotations

import math


def query_planning_accuracy(predicted_types: list[str], ground_truth_types: list[str]) -> dict:
    """Accuracy of predicted query classification labels against ground
    truth, plus a confusion breakdown (which true type gets predicted as
    which other type, and how often) — a bare accuracy number hides
    whether errors are random noise or one systematic confusion (e.g.
    "COMPARISON questions keep getting classified as CROSS_DOCUMENT").

    Both lists must be the same length and order (predicted_types[i]
    corresponds to ground_truth_types[i]) — this function does not
    re-sort or re-match by question id; that pairing is the caller's
    responsibility (see `evaluation/run_phase4_eval.py`, which builds
    both lists from the same dataset iteration order).

    A ground-truth label with no corresponding value in the predicted
    classifier's own enum (e.g. this dataset's `FIGURE_QUERY`, which
    predates `app.rag.query_classifier.QueryType` and has no matching
    member — see that module's docstring) is still scored: it simply
    never counts as correct, since the classifier can never produce a
    label it doesn't have. This is intentional — it's real information
    about label-set drift between the dataset and the classifier, not a
    bug to hide.

    Returns `{"accuracy": float, "n": int, "confusion": {true_type:
    {predicted_type: count}}}`, or `{"accuracy": None, "n": 0,
    "confusion": {}}` for empty input.
    """
    if not predicted_types or len(predicted_types) != len(ground_truth_types):
        return {"accuracy": None, "n": 0, "confusion": {}}

    n = len(ground_truth_types)
    correct = 0
    confusion: dict[str, dict[str, int]] = {}
    for predicted, truth in zip(predicted_types, ground_truth_types):
        if predicted == truth:
            correct += 1
        confusion.setdefault(truth, {})
        confusion[truth][predicted] = confusion[truth].get(predicted, 0) + 1

    return {"accuracy": round(correct / n, 4), "n": n, "confusion": confusion}


# Maps app.rag.comparison's real per-topic relation labels onto this
# dataset's coarser one_sided/both_sided ground truth. A topic is
# "one_sided" exactly when comparison found it in only one of the two
# documents (missing_in_a/missing_in_b); every other real relation value
# (same/different/conflicting) means the topic appears in BOTH documents,
# i.e. "both_sided" — the two schemas encode overlapping but not
# identical information, and this mapping is the deliberate, documented
# bridge between them (see this module's docstring).
_RELATION_TO_SIDEDNESS = {
    "same": "both_sided",
    "different": "both_sided",
    "conflicting": "both_sided",
    "missing_in_a": "one_sided",
    "missing_in_b": "one_sided",
}


def multi_document_reasoning_accuracy(predicted_relations: list[str], ground_truth_sidedness: list[str]) -> dict:
    """Accuracy of predicted comparison relations against
    `evaluation/datasets/multi_doc_v1.jsonl`'s `expected_relations`
    ground truth (`"one_sided"` | `"both_sided"` per topic), via the
    `_RELATION_TO_SIDEDNESS` mapping above.

    `predicted_relations` are the RAW relation strings
    `app.rag.comparison.compare_documents` actually returns (`"same"`,
    `"different"`, `"missing_in_a"`, `"missing_in_b"`, `"conflicting"`)
    — this function does the mapping internally so a caller never needs
    to duplicate `_RELATION_TO_SIDEDNESS` itself.

    An unrecognized `predicted_relations` value (a real bug in the
    comparison module, or a genuinely new relation type added later
    without updating this map) is treated as an automatic miss, never a
    crash and never silently "correct" — the caller can distinguish an
    unrecognized value in the confusion breakdown's predicted-side keys.

    Returns the same `{"accuracy", "n", "confusion"}` shape as
    `query_planning_accuracy`, for a consistent reporting shape across
    both reasoning-accuracy metrics.
    """
    if not predicted_relations or len(predicted_relations) != len(ground_truth_sidedness):
        return {"accuracy": None, "n": 0, "confusion": {}}

    n = len(ground_truth_sidedness)
    correct = 0
    confusion: dict[str, dict[str, int]] = {}
    for relation, truth in zip(predicted_relations, ground_truth_sidedness):
        predicted_sidedness = _RELATION_TO_SIDEDNESS.get(relation, f"unrecognized:{relation}")
        if predicted_sidedness == truth:
            correct += 1
        confusion.setdefault(truth, {})
        confusion[truth][predicted_sidedness] = confusion[truth].get(predicted_sidedness, 0) + 1

    return {"accuracy": round(correct / n, 4), "n": n, "confusion": confusion}


def table_reasoning_accuracy(
    predicted_values: list[float | None], ground_truth_values: list[float], relative_tolerance: float = 0.01
) -> dict | None:
    """Accuracy of `app.rag.table_reasoning.try_compute_from_chunks`'s
    numeric results against ground-truth table-computation values, using
    a relative-tolerance comparison (`abs(predicted - truth) <=
    relative_tolerance * abs(truth)`, default 1%) rather than exact
    equality — a correct "average revenue" computation can differ from
    hand-computed ground truth by float rounding without being wrong.

    `predicted_values[i] is None` (table reasoning declined to compute —
    see that module's `try_compute_from_chunks`, which returns `None`
    rather than guessing when it can't confidently parse the table/
    question) always counts as incorrect for THIS metric — a
    non-answer is not a correct numeric answer, even though it may be
    the right *behavior* (compare against `refusal_accuracy` for that
    separate judgment).

    **Currently always returns `None`** when called with no ground-truth
    values, which is the ONLY way this function is currently reachable —
    see this module's docstring: no (table, operation, expected value)
    ground-truth dataset exists in this repository yet. The scoring
    logic itself is implemented and unit-tested with synthetic ground
    truth (see tests/test_reasoning_metrics.py) so it's ready the moment
    such a dataset exists; it correctly reports "no ground truth
    supplied" rather than fabricating an accuracy number in its absence.
    """
    if not ground_truth_values or len(predicted_values) != len(ground_truth_values):
        return None

    n = len(ground_truth_values)
    correct = 0
    for predicted, truth in zip(predicted_values, ground_truth_values):
        if predicted is None:
            continue
        if truth == 0:
            is_correct = predicted == 0
        else:
            is_correct = abs(predicted - truth) <= relative_tolerance * abs(truth)
        if is_correct and not math.isnan(predicted):
            correct += 1

    return {"accuracy": round(correct / n, 4), "n": n}