"""Custom deterministic metrics (Phase 3A requirement #39).

These are computed directly from recorded pipeline outputs — no LLM
judge involved (RAGAS/TruLens integration, which DOES use an LLM judge
for faithfulness/relevance, lives separately; see `evaluation/README.md`).
Kept deterministic and dependency-free so they're cheap to run on every
benchmark pass and trivially unit-testable.

Each function operates on one of two shapes:

- A single "result" record (one benchmark question's outcome) — the
  per-question grading functions.
- A list of result records — the aggregate functions, which are simple
  means over the per-question functions and are the numbers that actually
  go in a README/dashboard.

A result record is expected to look like:
    {
      "id": "q001",
      "answerable": true,
      "found": true,
      "cited_chunk_ids": ["corpus::...::0"],
      "required_chunk_ids": ["corpus::...::0"],
      "retrieved_chunk_ids": ["corpus::...::0", ...],   # best-first
      "grounded_claim_rate": 0.9,          # optional, from grounding.py
      "citation_precision": 1.0,           # optional, from grounding.py
    }
This shape is produced by `evaluation/run_ablation.py` when it runs the
real pipeline against `evaluation/datasets/v1.jsonl` — see that file.
"""

from __future__ import annotations

import statistics

from app.evaluation.retrieval_metrics import evaluate_ranking


def grounded_answer_rate(results: list[dict]) -> float | None:
    """Fraction of answered (found=True) results whose claims were
    sufficiently grounded (`grounded_claim_rate >= 0.8`, i.e. at least
    4/5 of extracted claims were SUPPORTED). Only defined over answered
    questions — refusals aren't "grounded" or "ungrounded", they're
    absent. Returns None if no result carries `grounded_claim_rate`
    (claim-level grounding wasn't run for this batch).
    """
    answered = [r for r in results if r.get("found")]
    graded = [r for r in answered if r.get("grounded_claim_rate") is not None]
    if not graded:
        return None
    grounded = sum(1 for r in graded if r["grounded_claim_rate"] >= 0.8)
    return round(grounded / len(graded), 4)


def refusal_accuracy(results: list[dict]) -> float | None:
    """Fraction of results where the system's answer/refuse decision
    matched the benchmark's `answerable` ground truth:
    - answerable=True  -> correct if found=True
    - answerable=False -> correct if found=False (correctly refused)
    Returns None if no result carries `answerable` (not a benchmark run).

    This is a single BLENDED accuracy across both directions -- see
    `answerable_sensitivity`/`unanswerable_specificity` below for the
    split version. An earlier pass judged the split unnecessary ("no
    separate meaning beyond what this already computes"); that
    judgment is revisited here on an explicit, direct request (Phase 7
    audit) treating them as genuinely distinct diagnostics, which,
    statistically, they are: a system that answers everything scores
    100% sensitivity / 0% specificity, and `refusal_accuracy` alone
    would only reflect the dataset's answerable/unanswerable ratio,
    not reveal that asymmetry. Kept alongside the split metrics (not
    replaced by them) since a single blended number is still useful
    for at-a-glance comparison across ablation configs.
    """
    graded = [r for r in results if "answerable" in r and r["answerable"] is not None]
    if not graded:
        return None
    correct = sum(1 for r in graded if bool(r["found"]) == bool(r["answerable"]))
    return round(correct / len(graded), 4)


def answerable_sensitivity(results: list[dict]) -> float | None:
    """Of the questions that SHOULD have been answered (`answerable=True`
    in the benchmark), the fraction the system actually answered
    (`found=True`) -- the true-positive rate. Distinct failure mode from
    `unanswerable_specificity`: a low sensitivity means the system is
    too trigger-happy about refusing, missing answerable questions.
    Returns None if no result in this run has `answerable=True` (e.g. a
    dataset slice, or a config that refused/failed everything before
    grading -- not the same as a genuine 0.0, which means every
    answerable question was actually attempted and refused).
    """
    answerable_only = [r for r in results if r.get("answerable") is True]
    if not answerable_only:
        return None
    return round(sum(1 for r in answerable_only if r["found"]) / len(answerable_only), 4)


def unanswerable_specificity(results: list[dict]) -> float | None:
    """Of the questions that SHOULD have been refused (`answerable=False`
    in the benchmark), the fraction the system actually refused
    (`found=False`) -- the true-negative rate. Distinct failure mode from
    `answerable_sensitivity`: a low specificity means the system
    hallucinates answers to out-of-corpus questions rather than
    refusing them -- the more safety-critical of the two failure modes
    for a citation-grounded product. Returns None if no result in this
    run has `answerable=False`.
    """
    unanswerable_only = [r for r in results if r.get("answerable") is False]
    if not unanswerable_only:
        return None
    return round(sum(1 for r in unanswerable_only if not r["found"]) / len(unanswerable_only), 4)


def correct_answer_rate(results: list[dict]) -> float | None:
    """Fraction of ANSWERABLE questions (`answerable=True`) where the
    system both answered (`found=True`) AND cited every chunk the
    benchmark says was required (`citation_recall == 1.0` — i.e. full
    coverage of the expected evidence, not just *some* overlap).

    This is a **deterministic, citation-based proxy for correctness**,
    not a semantic judgment of whether the prose answer is actually
    right — that's what RAGAS's `answer_correctness` and TruLens's
    `answer_relevance` are for (see `app/evaluation/ragas_metrics.py`/
    `trulens_feedback.py`), and this module deliberately doesn't
    duplicate an LLM-judge metric with a weaker deterministic stand-in
    under the same name. Stated here explicitly rather than left
    ambiguous: "correct" in `correct_answer_rate` means "answered, and
    grounded in exactly the evidence expected" — a stricter, narrower
    claim than "the prose is factually right", but one requiring no LLM
    judge and always available.

    Returns None if no result carries both `answerable` and
    `required_chunk_ids`/`cited_chunk_ids` — i.e. wasn't run against a
    benchmark with citation ground truth.
    """
    graded = [
        r
        for r in results
        if r.get("answerable") is True and r.get("required_chunk_ids")
    ]
    if not graded:
        return None
    correct = 0
    for r in graded:
        if not r.get("found"):
            continue
        required = set(r["required_chunk_ids"])
        cited = set(r.get("cited_chunk_ids") or [])
        if required <= cited:
            correct += 1
    return round(correct / len(graded), 4)


def claim_grounding_rate_mean(results: list[dict]) -> float | None:
    """Mean of `grounded_claim_rate` (SUPPORTED claims / total claims —
    see `app.rag.grounding.ground_answer`) across every answered result
    that has it — the raw continuous rate, distinct from
    `grounded_answer_rate`'s binary "was this answer sufficiently
    grounded (>=0.8)" threshold. Both are useful: this one shows the
    average grounding QUALITY; `grounded_answer_rate` shows what
    fraction of answers clear a usability bar.
    """
    answered = [r for r in results if r.get("found")]
    values = [r["grounded_claim_rate"] for r in answered if r.get("grounded_claim_rate") is not None]
    if not values:
        return None
    return round(statistics.mean(values), 4)


def unsupported_claim_rate_mean(results: list[dict]) -> float | None:
    """Mean fraction of claims per answer that were NOT supported by
    their cited evidence (`1 - grounded_claim_rate`, averaged) —
    computed independently from `claim_grounding_rate_mean` (not just
    `1 - claim_grounding_rate_mean`) so a caller can trust each number
    on its own without assuming the complement relationship holds
    exactly under rounding.
    """
    answered = [r for r in results if r.get("found")]
    values = [1.0 - r["grounded_claim_rate"] for r in answered if r.get("grounded_claim_rate") is not None]
    if not values:
        return None
    return round(statistics.mean(values), 4)


def hallucination_rate(results: list[dict]) -> float | None:
    """Fraction of ANSWERED results with AT LEAST ONE unsupported claim
    (`grounded_claim_rate < 1.0`) — an answer-level "did this response
    hallucinate at all" binary rate, distinct from
    `unsupported_claim_rate_mean`'s continuous per-claim average. A
    single unsupported claim in an otherwise well-grounded 10-claim
    answer counts as a hallucination here (rate) but only drags the
    continuous rate down by 10%, on purpose — the two metrics answer
    different questions ("how often does hallucination happen at all"
    vs. "how much of a typical answer is affected"), and reporting only
    one would hide the other's answer.

    Note the exact same underlying signal (`grounded_claim_rate < 1.0`)
    is used here as a proxy for "contains an unsupported claim" —  a
    genuinely precise version would need the full per-claim `claims`
    list (each with its own SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED
    label — see `app.rag.grounding.ground_answer`'s `"claims"` field) to
    distinguish "one fully UNSUPPORTED claim" from "several
    PARTIALLY_SUPPORTED claims", which `grounded_claim_rate` alone
    cannot. `evaluation/run_ablation.py` does not currently persist the
    full per-claim list in its result records (only the aggregate rate)
    — see `evaluation/README.md`'s "Known gaps" for this documented
    precision limitation, not silently assumed away.
    """
    answered = [r for r in results if r.get("found")]
    graded = [r for r in answered if r.get("grounded_claim_rate") is not None]
    if not graded:
        return None
    hallucinated = sum(1 for r in graded if r["grounded_claim_rate"] < 1.0)
    return round(hallucinated / len(graded), 4)


def citation_precision_mean(results: list[dict]) -> float | None:
    """Mean chunk-level citation precision (from `grounding.ground_answer`)
    across results that have it. None if not computed for this batch.
    """
    values = [r["citation_precision"] for r in results if r.get("citation_precision") is not None]
    if not values:
        return None
    return round(statistics.mean(values), 4)


def citation_recall_mean(results: list[dict]) -> float | None:
    """Mean citation recall (required chunks that were actually cited)
    across results that have ground-truth `required_chunk_ids`.
    """
    values = []
    for r in results:
        required = r.get("required_chunk_ids") or []
        if not required:
            continue
        cited = set(r.get("cited_chunk_ids") or [])
        values.append(len(set(required) & cited) / len(required))
    if not values:
        return None
    return round(statistics.mean(values), 4)


def retrieval_metrics_mean(results: list[dict], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict | None:
    """Mean Recall@K / MRR / nDCG across every result that has both
    `retrieved_chunk_ids` and `required_chunk_ids` ground truth.
    """
    per_question = []
    for r in results:
        required = r.get("required_chunk_ids")
        retrieved = r.get("retrieved_chunk_ids")
        if not required or retrieved is None:
            continue
        per_question.append(evaluate_ranking(retrieved, required, k_values))
    if not per_question:
        return None
    keys = per_question[0].keys()
    return {key: round(statistics.mean(row[key] for row in per_question), 4) for key in keys}


def latency_percentiles(latencies_seconds: list[float]) -> dict:
    """p50/p95/p99 + mean, rounded to milliseconds. Empty input -> zeros
    (never fabricated — an empty run has no latency data).
    """
    if not latencies_seconds:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "n": 0}
    values = sorted(latencies_seconds)

    def _pct(p: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
        return values[idx]

    return {
        "p50_ms": round(_pct(0.50) * 1000, 2),
        "p95_ms": round(_pct(0.95) * 1000, 2),
        "p99_ms": round(_pct(0.99) * 1000, 2),
        "mean_ms": round(statistics.mean(values) * 1000, 2),
        "n": len(values),
    }


def usage_summary(results: list[dict]) -> dict:
    """Aggregate per-question `llm_calls` (token/cost — see
    `app.observability.trace.RequestTrace.to_dict`) across a whole
    benchmark run. Phase 7 completion pass: centralized here (not
    duplicated in `evaluation/run_ablation.py` AND
    `evaluation/run_phase4_eval.py` separately) so both runners — and
    `evaluation/modality.py`'s per-modality-bucket calls into
    `summarize()` — get real, consistent token/cost reporting for free.

    `results[i].get("llm_calls")` is populated by a runner ONLY once it
    has actually called `app.observability.trace.start_trace()` around
    its `service.ask()` call (see `run_ablation.py::run_config`'s
    comment for the full history of why this was previously silently
    empty everywhere) — a result dict with no `llm_calls` key at all
    (e.g. `tests/test_evaluation_metrics.py`'s synthetic fixtures, which
    predate this field) is treated the same as an empty list, not an
    error.

    Returns `{"status": "MEASURED", ...}` when at least one real LLM
    call was captured, or `{"status": "UNAVAILABLE", "reason": ...}` —
    NEVER a fabricated zero-cost/zero-token summary.
    """
    all_calls = [call for r in results for call in (r.get("llm_calls") or [])]
    if not all_calls:
        return {
            "status": "UNAVAILABLE",
            "reason": (
                "No LLM response in this run exposed a `.usage` attribute (or every "
                "question was refused before any LLM call was made) — token/cost genuinely "
                "were not captured, not silently reported as zero."
            ),
        }

    total_prompt = sum(c["prompt_tokens"] for c in all_calls)
    total_completion = sum(c["completion_tokens"] for c in all_calls)
    by_purpose: dict[str, dict] = {}
    for purpose in sorted({c["purpose"] for c in all_calls}):
        calls = [c for c in all_calls if c["purpose"] == purpose]
        by_purpose[purpose] = {
            "n_calls": len(calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            "total_cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
        }
    return {
        "status": "MEASURED",
        "n_llm_calls": len(all_calls),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "total_cost_usd": round(sum(c["cost_usd"] for c in all_calls), 6),
        "by_purpose": by_purpose,
        # See app/observability/cost.py's own module docstring before
        # trusting total_cost_usd for anything beyond rough
        # order-of-magnitude — it's a configured PRICING ESTIMATE applied
        # to real measured token counts, not a provider-reported bill.
        "cost_is_estimated_not_billed": True,
    }


def summarize(results: list[dict], latencies_seconds: list[float] | None = None) -> dict:
    """Everything the /evaluations dashboard needs from one batch of
    benchmark results, in one call. Any metric with no applicable data
    is reported as `null` rather than a fabricated number.
    """
    summary = {
        "n_questions": len(results),
        "grounded_answer_rate": grounded_answer_rate(results),
        "refusal_accuracy": refusal_accuracy(results),
        "answerable_sensitivity": answerable_sensitivity(results),
        "unanswerable_specificity": unanswerable_specificity(results),
        "correct_answer_rate": correct_answer_rate(results),
        "citation_precision": citation_precision_mean(results),
        "citation_recall": citation_recall_mean(results),
        "claim_grounding_rate": claim_grounding_rate_mean(results),
        "unsupported_claim_rate": unsupported_claim_rate_mean(results),
        "hallucination_rate": hallucination_rate(results),
    }
    retrieval = retrieval_metrics_mean(results)
    if retrieval:
        summary["retrieval"] = retrieval
    if latencies_seconds:
        summary["latency"] = latency_percentiles(latencies_seconds)
    summary["usage"] = usage_summary(results)
    # TTFT (time to first token) is architecturally not measurable
    # through either benchmark runner: `RequestTrace.mark_first_token()`
    # is only ever called from `app/api/ask_stream.py` (the streaming
    # endpoint) — it has no meaning for `DocumentQAService.ask()`'s
    # synchronous, non-streaming call shape, which is what both runners
    # use. Reported explicitly (NOT_APPLICABLE) rather than omitted, so
    # a reader doesn't have to infer its absence from a missing key —
    # and never silently computed as an approximation of end-to-end
    # latency, which would NOT be the same measurement.
    summary["ttft"] = {
        "status": "NOT_APPLICABLE",
        "reason": (
            "mark_first_token() is only called from app/api/ask_stream.py (the streaming "
            "endpoint) — TTFT has no meaning for DocumentQAService.ask()'s synchronous, "
            "non-streaming call shape, which is what the benchmark runners use."
        ),
    }
    return summary