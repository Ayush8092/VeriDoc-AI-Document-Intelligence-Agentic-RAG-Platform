"""Tests for evaluation/run_ablation.py's retrieval-metrics wiring.

Phase 3A hardening problems #2 and #3: the ablation runner used to only
capture final validated citations, leaving Recall@K/MRR/nDCG unwired
(the pure functions existed in app/evaluation/retrieval_metrics.py but
nothing fed them real rankings) and had no way to tell whether a
"cross_encoder" config actually used the cross-encoder or silently fell
back to BM25. These tests exercise `run_config` against a fake
`DocumentQAService` (no network/LLM calls) to prove both are now wired
end-to-end, not just that the pure metric functions work in isolation
(already covered by test_bm25_and_fusion.py / retrieval_metrics tests).
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from evaluation.run_ablation import CONFIGS, run_config


class _FakeService:
    """Stands in for DocumentQAService.ask() with scripted, deterministic
    per-question outcomes carrying the same keys the real graph.ask()
    returns (see app/rag/graph.py's DocumentQAService.ask)."""

    def __init__(self, settings):
        self.settings = settings

    def ask(self, question: str) -> dict:
        # One required chunk is retrieved but only ranked #2 by the
        # (fake) reranker, and only #1 by dense/fused — enough to make
        # recall@1 vs recall@3 actually differ and prove the metric is
        # computed from the real per-stage lists, not a stub.
        dense = [{"chunk_id": "docA::0"}, {"chunk_id": "docB::0"}]
        bm25 = [{"chunk_id": "docB::0"}, {"chunk_id": "docA::0"}]
        fused = [{"chunk_id": "docA::0"}, {"chunk_id": "docB::0"}]
        reranked = [{"chunk_id": "docB::0"}, {"chunk_id": "docA::0"}]
        found = True
        citations = [{"chunk_id": "docB::0"}]
        return {
            "answer": "stub answer",
            "found": found,
            "citations": citations,
            "trace": [],
            "query_type": "LOOKUP",
            "claims": [],
            "grounded_claim_rate": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "stage_latencies": [],
            "dense_candidates": dense,
            "bm25_candidates": bm25,
            "fused_candidates": fused,
            "reranked_candidates": reranked,
            "final_evidence": citations,
            "configured_reranker": "cross_encoder" if self.settings.cross_encoder_enabled else "bm25",
            # Simulate the cross-encoder being unavailable so the
            # fallback-visibility path is exercised too.
            "actual_reranker": "bm25",
            "fallback_used": bool(self.settings.cross_encoder_enabled),
        }


def test_run_config_wires_real_retrieval_metrics(monkeypatch):
    monkeypatch.setattr("evaluation.run_ablation.DocumentQAService", _FakeService)

    dataset = [
        {
            "id": "q1",
            "question": "What is the notice period?",
            "query_type": "LOOKUP",
            "answerable": True,
            "required_chunks": ["docB::0"],
        }
    ]

    report = run_config(CONFIGS["D"], dataset, Settings(_env_file=None))

    result = report["results"][0]
    # The required chunk is ranked #1 in the (fake) reranked/BM25 output
    # but only #2 in dense/fused -- proves retrieved_chunk_ids actually
    # reflects the reranked stage, not a fabricated or default list.
    assert result["retrieved_chunk_ids"] == ["docB::0", "docA::0"]
    assert result["reranked_chunk_ids"] == ["docB::0", "docA::0"]
    assert result["dense_chunk_ids"] == ["docA::0", "docB::0"]

    retrieval = report["summary"]["retrieval"]
    assert retrieval["recall@1"] == 1.0  # required chunk IS top-1 of the reranked list
    assert retrieval["mrr"] == 1.0

    integrity = report["summary"]["reranker_integrity"]
    assert integrity["reranker_invocations"] == 1
    assert integrity["fallback_invocations"] == 1
    assert integrity["fallback_rate"] == 1.0
    assert integrity["configured_reranker"] == "cross_encoder"


def test_run_config_config_a_reports_no_reranker_fallback(monkeypatch):
    """Config A (dense-only, RERANK_ENABLED=False) never invokes a
    reranker at all -> fallback_invocations must be 0/None, never a
    fabricated non-zero rate."""
    monkeypatch.setattr("evaluation.run_ablation.DocumentQAService", _FakeService)

    class _NoRerankFakeService(_FakeService):
        def ask(self, question: str) -> dict:
            result = super().ask(question)
            result["actual_reranker"] = "none"
            result["fallback_used"] = False
            return result

    monkeypatch.setattr("evaluation.run_ablation.DocumentQAService", _NoRerankFakeService)

    dataset = [
        {
            "id": "q1",
            "question": "What is the notice period?",
            "query_type": "LOOKUP",
            "answerable": True,
            "required_chunks": ["docB::0"],
        }
    ]
    report = run_config(CONFIGS["A"], dataset, Settings(_env_file=None))
    integrity = report["summary"]["reranker_integrity"]
    assert integrity["reranker_invocations"] == 0
    assert integrity["fallback_rate"] is None
    assert integrity["configured_reranker"] == "none"


def test_ablation_config_overrides_actually_apply():
    """Regression test for a bug found while fixing problem #3:
    `Settings.model_copy(update=...)` matches fields by their real
    (lower-case) attribute name, not `validation_alias` — so
    `CONFIGS`' override dicts must use e.g. `rerank_enabled`, never the
    env-style `RERANK_ENABLED`, or the override is silently a no-op and
    every ablation config ends up running identical, all-default
    settings (see run_ablation.py's module docstring for the full story).
    """
    base = Settings(_env_file=None)
    assert base.rerank_enabled is True
    assert base.cross_encoder_enabled is True
    assert base.hybrid_retrieval_enabled is True
    assert base.claim_grounding_enabled is True

    for name, config in CONFIGS.items():
        applied = base.model_copy(update=config.overrides)
        for field_name, expected in config.overrides.items():
            # Every override key must be a real Settings field name...
            assert hasattr(base, field_name), (
                f"config {name}: '{field_name}' is not a real Settings field "
                "(did this regress to an env-style key?)"
            )
            # ...and must have actually taken effect on the copy.
            assert getattr(applied, field_name) == expected, (
                f"config {name}: override '{field_name}={expected}' did not apply"
            )

    # Spot-check the configs are actually distinct from each other (the
    # whole point of an ablation study) now that overrides apply.
    a = base.model_copy(update=CONFIGS["A"].overrides)
    e = base.model_copy(update=CONFIGS["E"].overrides)
    assert a.rerank_enabled != e.rerank_enabled
    assert a.hybrid_retrieval_enabled != e.hybrid_retrieval_enabled


class _FakeServiceWithRealLLMCall(_FakeService):
    """Same scripted outcome as `_FakeService`, but its `.ask()` ALSO
    calls `current_trace().record_llm_usage(...)` — simulating what
    `app/rag/llm.py::_record_usage` actually does on every real LLM
    call. `_FakeService` alone can't test the trace-integration fix
    (Phase 7 completion pass): its `.ask()` never touches
    `current_trace()` at all, so it would pass whether or not
    `run_config` remembered to call `start_trace()` per question.
    """

    def ask(self, question: str) -> dict:
        from app.observability.trace import current_trace

        trace = current_trace()
        if trace is not None:
            trace.record_llm_usage(model="fake-model", prompt_tokens=100, completion_tokens=50, purpose="answer")
        return super().ask(question)


def test_run_config_captures_token_usage_via_per_question_start_trace(monkeypatch):
    """Phase 7 completion pass: `run_config` must call `start_trace()`
    itself around each `service.ask()` call, since
    `evaluation/run_ablation.py` calls `DocumentQAService.ask()` directly
    rather than going through the HTTP layer
    (`app/api/ask.py`/`app/api/ask_stream.py`, the only other place that
    starts a trace) — without this, `app.rag.llm._record_usage`'s
    `current_trace()` read would silently return `None` for the whole
    run and every token/cost record would be a no-op, even though the
    LLM call genuinely happened. This proves the fix actually threads
    through end-to-end: a fake service that performs a real
    `record_llm_usage()` call (simulating a real LLM response) must have
    that call captured in both the per-question `results[i]["llm_calls"]`
    and the aggregate `summary["usage"]`.
    """
    monkeypatch.setattr("evaluation.run_ablation.DocumentQAService", _FakeServiceWithRealLLMCall)

    dataset = [
        {
            "id": "q1",
            "question": "What is the notice period?",
            "query_type": "LOOKUP",
            "answerable": True,
            "required_chunks": ["docB::0"],
        },
        {
            "id": "q2",
            "question": "What is the termination fee?",
            "query_type": "LOOKUP",
            "answerable": True,
            "required_chunks": ["docB::0"],
        },
    ]
    report = run_config(CONFIGS["A"], dataset, Settings(_env_file=None))

    # Per-question capture: each question made exactly one LLM call.
    assert len(report["results"]) == 2
    for result in report["results"]:
        assert result["llm_calls"] == [
            {
                "model": "fake-model",
                "purpose": "answer",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost_usd": result["llm_calls"][0]["cost_usd"],  # fallback-pricing value, not asserted exactly
            }
        ]

    # Aggregate capture: summary["usage"] must be MEASURED (not
    # UNAVAILABLE) and reflect BOTH questions' calls, not just one --
    # proving start_trace() correctly isolates each question (no
    # cross-question leakage or overwrite) while still aggregating
    # correctly afterward.
    usage = report["summary"]["usage"]
    assert usage["status"] == "MEASURED"
    assert usage["n_llm_calls"] == 2
    assert usage["total_prompt_tokens"] == 200
    assert usage["total_completion_tokens"] == 100
    assert usage["by_purpose"]["answer"]["n_calls"] == 2

    # TTFT is honestly reported as not applicable for this synchronous
    # runner, never silently omitted or fabricated as a number.
    assert report["summary"]["ttft"]["status"] == "NOT_APPLICABLE"


def test_run_config_usage_is_unavailable_not_fabricated_zero_when_no_llm_calls_happen(monkeypatch):
    """The other side of the same fix: a run where NO LLM call ever
    touches current_trace() (e.g. every question is refused before
    generation, or the fake/real service genuinely made zero calls)
    must report `summary["usage"]["status"] == "UNAVAILABLE"` with a
    reason -- never a fabricated `total_cost_usd: 0.0` that looks like a
    real measurement of "this run cost nothing"."""
    monkeypatch.setattr("evaluation.run_ablation.DocumentQAService", _FakeService)  # no LLM calls at all

    dataset = [
        {
            "id": "q1",
            "question": "What is the notice period?",
            "query_type": "LOOKUP",
            "answerable": True,
            "required_chunks": ["docB::0"],
        }
    ]
    report = run_config(CONFIGS["A"], dataset, Settings(_env_file=None))
    assert report["summary"]["usage"]["status"] == "UNAVAILABLE"
    assert "reason" in report["summary"]["usage"]
    assert "total_cost_usd" not in report["summary"]["usage"]


# ---------------------------------------------------------------------------
# Checkpoint / resume / rate-limit tests
#
# Everything below monkeypatches `evaluation.run_ablation.REPORTS_DIR` to an
# isolated tmp_path (never the real evaluation/reports/) and
# `_classify_rate_limit_error` to a scripted stand-in (rather than
# constructing a real groq.APIError, which needs a real httpx.Request —
# see groq.APIError's constructor; that SDK-shape-parsing logic is this
# function's own responsibility, not something these control-flow tests
# need to re-verify). What IS real: `run_config`'s actual try/except,
# checkpoint file I/O, and resume-skip logic — the exact behavior the
# spec asked to be proven, not simulated.
# ---------------------------------------------------------------------------

import json as _json

from evaluation.run_ablation import (
    DEFAULT_DATASET,
    _checkpoint_path,
    _load_checkpoint,
    _rate_limit_path,
    _RateLimitHit,
)


class _RateLimitSignal(Exception):
    """A stand-in "this call hit a rate limit" exception — recognized only
    because the test below monkeypatches `_classify_rate_limit_error` to
    recognize it, not because it's a real groq/genai SDK type."""


class _FakeServiceFailsAfterN(_FakeService):
    """Answers the first `fail_after` questions normally, then raises
    `_RateLimitSignal` on every question after that — simulating a
    provider rate limit kicking in partway through a run. Tracks every
    question it was actually asked, in order, so a test can assert
    exactly which ones were (and weren't) re-sent on resume.
    """

    def __init__(self, settings, fail_after: int):
        super().__init__(settings)
        self.fail_after = fail_after
        self.asked: list[str] = []

    def ask(self, question: str) -> dict:
        self.asked.append(question)
        if len(self.asked) > self.fail_after:
            raise _RateLimitSignal("simulated 429")
        return super().ask(question)


def _five_question_dataset() -> list[dict]:
    return [
        {
            "id": f"q00{i}",
            "question": f"Question number {i}?",
            "query_type": "LOOKUP",
            "answerable": True,
            "required_chunks": ["docB::0"],
        }
        for i in range(1, 6)
    ]


def _patch_rate_limit_classifier(monkeypatch, recognized_exc_type):
    """Makes `_classify_rate_limit_error` recognize `recognized_exc_type`
    as a rate-limit hit (returning a scripted `_RateLimitHit`) and
    nothing else — a real, un-recognized exception (e.g. ValueError)
    still falls through to `None`, matching the real function's "only a
    genuine provider rate-limit error is caught here" contract.
    """
    import evaluation.run_ablation as run_ablation_module

    def _fake_classify(exc: Exception):
        if isinstance(exc, recognized_exc_type):
            return _RateLimitHit(
                provider="groq",
                status_code=429,
                retry_after_seconds=None,
                detail="Rate limit reached for model `llama-3.1-8b-instant` on tokens per day (TPD): "
                "Limit 100000, Used 100000. Please try again in 23h59m45s.",
            )
        return None

    monkeypatch.setattr(run_ablation_module, "_classify_rate_limit_error", _fake_classify)


def test_run_config_rate_limit_checkpoints_and_reports_gracefully(monkeypatch, tmp_path):
    """Spec items 2 + 4: a rate-limit error partway through a run must
    (a) never raise/crash with a traceback out of run_config, (b) return
    a structured report with config/last-completed-id/completed-remaining
    counts/provider/retry info, and (c) leave a checkpoint on disk
    containing exactly the questions that succeeded before the failure —
    not the one that failed, not any that never ran.
    """
    import evaluation.run_ablation as run_ablation_module

    monkeypatch.setattr(run_ablation_module, "REPORTS_DIR", tmp_path)
    _patch_rate_limit_classifier(monkeypatch, _RateLimitSignal)

    fake = _FakeServiceFailsAfterN(Settings(_env_file=None), fail_after=2)
    monkeypatch.setattr(run_ablation_module, "DocumentQAService", lambda settings: fake)

    dataset = _five_question_dataset()
    report = run_config(CONFIGS["A"], dataset, Settings(_env_file=None))

    # (a) + (b): no exception escaped; a structured, honest report instead.
    assert report["status"] == "rate_limited"
    assert report["config"] == "A"
    assert report["provider"] == "groq"
    assert report["last_completed_question_id"] == "q002"
    assert report["completed_count"] == 2
    assert report["remaining_count"] == 3
    assert report["total_count"] == 5
    assert "23h59m45s" in report["detail"]

    # (c): the checkpoint on disk has exactly the 2 that succeeded — not
    # the failed 3rd question, not any of the never-attempted 4th/5th.
    checkpoint = _load_checkpoint("A")
    assert checkpoint is not None
    assert [r["id"] for r in checkpoint["results"]] == ["q001", "q002"]

    # The provider's own detail (which states the ACTUAL retry window —
    # the real gap this test guards against regressing) is persisted to
    # disk, not just printed — see _rate_limit_path's docstring.
    rate_limit_file = _rate_limit_path("A")
    assert rate_limit_file.exists()
    persisted = _json.loads(rate_limit_file.read_text(encoding="utf-8"))
    assert "23h59m45s" in persisted["detail"]

    # The service was asked exactly 3 times (2 successes + the 1 that
    # failed) — never for q004/q005, which never should have been
    # attempted once the limit was hit.
    assert fake.asked == ["Question number 1?", "Question number 2?", "Question number 3?"]


def test_run_config_resume_skips_already_completed_questions(monkeypatch, tmp_path):
    """Spec item 3 + 8, the core requirement: on resume, questions already
    present in the checkpoint must be skipped entirely — never re-sent to
    the (expensive, rate-limited) provider — and the final report must
    still contain the full, correct result set once the remaining
    questions complete.
    """
    import evaluation.run_ablation as run_ablation_module

    monkeypatch.setattr(run_ablation_module, "REPORTS_DIR", tmp_path)

    dataset = _five_question_dataset()

    # Pre-seed a checkpoint as if q001/q002 already completed in an
    # earlier, since-interrupted run.
    from evaluation.run_ablation import _write_checkpoint

    seeded_results = [
        {**_FakeService(Settings(_env_file=None)).ask(row["question"]), "id": row["id"], "question": row["question"]}
        for row in dataset[:2]
    ]
    _write_checkpoint(
        "A",
        {
            "config": "A",
            "dataset_path": str(DEFAULT_DATASET),
            "results": seeded_results,
            "latencies": [0.01, 0.01],
            "reranker_calls": 0,
            "reranker_fallbacks": 0,
            "llm_calls_all": [],
        },
    )

    fake = _FakeService(Settings(_env_file=None))
    original_ask = fake.ask
    asked_questions: list[str] = []

    def _tracking_ask(question: str) -> dict:
        asked_questions.append(question)
        return original_ask(question)

    fake.ask = _tracking_ask
    monkeypatch.setattr(run_ablation_module, "DocumentQAService", lambda settings: fake)

    report = run_config(CONFIGS["A"], dataset, Settings(_env_file=None), dataset_path=None, resume=True)

    # The 2 already-completed questions were NEVER re-asked — only the 3
    # genuinely remaining ones were.
    assert asked_questions == ["Question number 3?", "Question number 4?", "Question number 5?"]

    # The final report still has all 5, in order, not just the 3 new ones.
    assert [r["id"] for r in report["results"]] == ["q001", "q002", "q003", "q004", "q005"]

    # A fully-completed run cleans up its own checkpoint — a leftover
    # would make the NEXT --resume invocation ambiguous about whether
    # this config still needs work.
    assert _load_checkpoint("A") is None


def test_run_config_resume_without_a_checkpoint_behaves_like_a_fresh_run(monkeypatch, tmp_path):
    """Resume is opt-in via a flag, not a hard requirement of a prior
    checkpoint existing — if `--resume` is passed for a config that has
    no checkpoint at all (e.g. it never ran, or already finished and was
    cleaned up), run_config must just run every question normally rather
    than erroring."""
    import evaluation.run_ablation as run_ablation_module

    monkeypatch.setattr(run_ablation_module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(run_ablation_module, "DocumentQAService", _FakeService)

    dataset = _five_question_dataset()
    report = run_config(CONFIGS["A"], dataset, Settings(_env_file=None), resume=True)

    assert [r["id"] for r in report["results"]] == ["q001", "q002", "q003", "q004", "q005"]


def test_run_config_non_rate_limit_exception_still_propagates(monkeypatch, tmp_path):
    """Spec item 4's carve-out, item 6's "do not substitute results": an
    exception `_classify_rate_limit_error` does NOT recognize as a
    provider rate limit (a real bug — bad config, a programming error,
    etc.) must propagate normally, with its real traceback — never get
    silently reinterpreted as a graceful rate-limit pause.
    """
    import evaluation.run_ablation as run_ablation_module

    monkeypatch.setattr(run_ablation_module, "REPORTS_DIR", tmp_path)
    _patch_rate_limit_classifier(monkeypatch, _RateLimitSignal)  # ValueError below is NOT this type

    class _FakeServiceRaisesRealBug(_FakeService):
        def ask(self, question: str) -> dict:
            raise ValueError("a genuine bug, unrelated to rate limiting")

    monkeypatch.setattr(run_ablation_module, "DocumentQAService", _FakeServiceRaisesRealBug)

    with pytest.raises(ValueError, match="a genuine bug"):
        run_config(CONFIGS["A"], _five_question_dataset(), Settings(_env_file=None))

    # And nothing was left behind on disk for an error that was never a
    # recoverable rate-limit checkpoint in the first place.
    assert _load_checkpoint("A") is None