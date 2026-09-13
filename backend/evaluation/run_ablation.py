"""Retrieval ablation study (Phase 3A requirement #25; extended to A-G
in Phase 6, spec item 12).

**Phase 6 completion pass — reconciliation note (fix_phase_6, Problem
2)**: this file and `app/evaluation/run_ablation.py` had diverged — the
`app/` copy had the complete A-G implementation (this one), but every
real caller (`evaluation/run_phase4_eval.py`, `evaluation/modality.py`,
`evaluation/dashboard.py`, `tests/test_run_ablation.py`, and
`python -m evaluation.run_ablation` itself) imports from THIS location,
`evaluation/run_ablation.py` — making Config F/G unreachable through any
actual entry point. Verified by grepping every `run_ablation` reference
in the repository before touching anything (no caller anywhere imports
`app.evaluation.run_ablation`). This file now contains the reconciled,
complete implementation; `app/evaluation/run_ablation.py` has been
removed (confirmed zero importers, so removing it deletes no reachable
functionality — see the Phase 6 completion report). Two real bugs caught
during reconciliation, present in BOTH the stale and the "complete"
copy, fixed here:
  - `DEFAULT_DATASET` pointed at `evaluation/datasets/v1.jsonl`, which
    doesn't exist in this repository (superseded by `phase4_v1.jsonl` —
    see that file's own header comment) — every invocation with no
    explicit `--dataset` would have raised `FileNotFoundError`.
  - `load_multi_doc_dataset`'s docstring claimed `expected_relations`
    values are `"same"|"different"|"missing"|"conflicting"`, but the
    actual matching code in `run_multi_doc_ablation` only ever checks
    for `"one_sided"`/`"both_sided"`. The docstring was wrong, not the
    code — fixed the docstring, and built
    `evaluation/datasets/multi_doc_v1.jsonl` (Problem 1) using the
    values the code actually reads.

Runs the REAL pipeline — real vector-store retrieval (Pinecone or the
free/local backend — see `app.vectorstore_local`), real Gemini
embeddings, real Groq generation — against
`evaluation/datasets/phase4_v1.jsonl` under several retrieval
configurations, and writes measured results to
`evaluation/reports/ablation_<config>.json`.

This script requires:
    1. GEMINI_API_KEY, GROQ_API_KEY set (backend/.env); PINECONE_API_KEY
       only if VECTORSTORE_BACKEND=pinecone (the default,
       VECTORSTORE_BACKEND=local, needs no API key)
    2. The corpus already ingested (`python -m app.services.ingestion_service`)
       so both the vector index and `data/lexical_index.json` exist.

It has NOT been executed in the environment this was built in (no live API
keys available there) — see evaluation/README.md. Every number this
script would produce is a genuine measurement of the pipeline's actual
behavior at run time; nothing here is a stand-in or placeholder value.

Configurations (each isolates one retrieval/pipeline decision so its
effect can be read off the results independently):

    A - baseline           dense-only, no rerank            (Phase 1/2 behavior)
    B - +BM25 fusion       dense + BM25 (RRF), no rerank
    C - +BM25 lexical rerank   dense + BM25 fusion, BM25 lexical rerank
                                (an intermediate rerank tier this
                                project's reranker already supports as a
                                fallback — kept as its own measurable
                                stage rather than collapsed away)
    D - +cross-encoder     dense + BM25 fusion, cross-encoder rerank
    E - +claim grounding + citation validation (both ON) — the project's
        normal default pipeline
    F - +LLM query planning    same as E, plus LLM-refined query
                                classification (Phase 6 spec item 12,
                                "F = multi-query/query planning") — see
                                `app.core.config.Settings.query_classifier_llm_enabled`
                                and `app.rag.query_classifier`.
    H - E with citation validation DISABLED (Phase 7 spec item 2/9,
        "citation-validation bypass/toggle") — isolates citation
        validation's own marginal effect versus E: identical pipeline,
        only `citation_validation_enabled` differs. See
        `Settings.citation_validation_enabled`'s docstring and
        `app.rag.graph._validate_citations`'s ablation-only bypass
        branch for exactly what changes (never drops/refuses on a bad
        citation — the citations E would have dropped are marked
        `"verified": False` with empty fields instead of just vanishing,
        so H's degraded output is itself never a fabrication).

`FINAL_TABLE_ROW_CONFIGS` below maps the Phase 7 spec's exact 5-row
"Vector / +BM25 / +Cross Encoder / +Grounding / +Citation Validation"
table onto these stable config letters (using D and H/E for the last
three rows, NOT C — see that mapping's own comment for why) rather than
renaming any existing letter, since A-F already have established
meanings and existing tests (`tests/test_run_ablation.py`) that must not
be disturbed by a relettering.

Config G ("multi-document reasoning") is intentionally NOT one more entry
in `CONFIGS` above: A-F/H all answer the SAME single-document-question
dataset under different retrieval settings, but "multi-document
reasoning" isn't a retrieval setting to toggle — it's a genuinely
different pipeline (`app.rag.comparison.compare_multiple`, not
`DocumentQAService.ask`) that needs its own dataset shape (a set of
`documents` (filenames) plus a comparison topic, not one `question`
string). See `run_multi_doc_ablation` below — same "REAL pipeline, only
measured results, nothing fabricated" discipline, over
`evaluation/datasets/multi_doc_v1.jsonl`.

Reproducibility (Phase 7 spec item 10, "reproducible configuration"):
every report (`run_config`/`run_multi_doc_ablation`) includes a
`run_metadata` block — dataset path, question count, UTC timestamp, and
the embedding/answer model + vector-store backend actually in effect —
so a report read later self-describes exactly what produced it, without
needing to cross-reference which script version or corpus state was
live at the time.

Retrieval metrics (Phase 3A hardening — problem #2; Phase 7 spec item
10's full metric set): `service.ask()` returns the full per-stage
candidate breakdown (dense/BM25/fused/reranked chunk IDs, best-first)
captured directly from the graph's final state, not re-derived after the
fact. This run captures that breakdown per question and feeds
`retrieved_chunk_ids` into
`app/evaluation/metrics.py::retrieval_metrics_mean`, which calls
`app/evaluation/retrieval_metrics.py::evaluate_ranking` — Recall@K,
Precision@K, Hit Rate@K, MRR, and nDCG@K (every metric family the spec
asks for, all already implemented there — verified by reading both
modules directly, not assumed) computed against the benchmark's
`required_chunks` ground truth for real.

Reranker integrity (Phase 3A hardening — problem #3): each result also
carries `configured_reranker`/`actual_reranker`/`fallback_used` straight
from the graph's `_rerank` node, and the aggregate summary reports a
`reranker_integrity` block (invocation/fallback counts) — so config D/E's
results can never be silently attributed to the cross-encoder if the
model was actually unavailable for some or all of the run.

Config override keys (Phase 3A hardening — a bug found while fixing
problem #3, not itself one of the ten enumerated problems): `CONFIGS`
below previously used upper-case, `.env`-style keys (e.g.
`"RERANK_ENABLED"`) as `Settings.model_copy(update=...)` arguments.
`model_copy(update=...)` replaces fields by their actual (lower-case)
attribute name, NOT by `validation_alias` — env-style keys are silently
accepted as extra, inert dict entries and never actually override
anything. Every config A-E was therefore running with identical,
all-default settings; the ablation study has never actually varied any
retrieval component since it was introduced. Fixed by using the real
lower-case field names (`rerank_enabled`, `cross_encoder_enabled`, ...).
See `tests/test_run_ablation.py::test_ablation_config_overrides_actually_apply`
for a regression test.

Usage:
    python -m evaluation.run_ablation
    python -m evaluation.run_ablation --configs A B D F
    python -m evaluation.run_ablation --dataset evaluation/datasets/phase4_v1.jsonl
    python -m evaluation.run_ablation --multi-doc

**Resuming after a rate-limit failure (Phase 7 completion pass — "resume
safely after an API rate-limit failure")**: a 300-question run against a
free-tier Groq/Gemini quota can genuinely exhaust its rate limit partway
through. Rather than losing the completed questions when that happens:

    python -m evaluation.run_ablation --configs A B D H E
    # ... Config A gets partway through, Groq returns 429 after its
    # retries are exhausted, the script prints a structured report and
    # exits -- see `_print_rate_limit_report` -- WITHOUT a raw traceback,
    # and WITHOUT overwriting any config's already-completed
    # evaluation/reports/ablation_<X>.json.

    # wait out the rate limit / quota window, then:
    python -m evaluation.run_ablation --configs A B D H E --resume
    # Config A resumes from evaluation/reports/ablation_A.checkpoint.json
    # (skips every question already completed); configs already fully
    # finished in the first invocation are detected and skipped entirely
    # (zero extra API calls for them).

See `run_config`'s docstring for exactly what's checkpointed and when,
and `_classify_rate_limit_error`'s docstring for exactly which
exceptions are treated as "pause and resume" vs. "a real bug -- let it
crash with its real traceback".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.evaluation.metrics import summarize
from app.rag.graph import DocumentQAService

# Phase 7 completion pass ("update ablation configuration imports"):
# `AblationConfig`, `CONFIGS`, and `FINAL_TABLE_ROW_CONFIGS` used to be
# DEFINED here — they are now pure-data and live in
# `evaluation/ablation_config.py` instead (see that module's docstring
# for the real, confirmed bug this fixes: `evaluation/dashboard.py`
# needing to import a plain dict was pulling in this module's own
# `pydantic`/`langgraph`/full-app-stack imports above). Imported, not
# duplicated, and re-exported under their original names so every
# existing `from evaluation.run_ablation import CONFIGS`-style import
# (this file's own CLI below, `tests/test_run_ablation.py`,
# `evaluation/generate_pipeline_outputs.py`) keeps working completely
# unchanged — this is a refactor of WHERE these are defined, not of
# what they mean or how they're used.
from evaluation.ablation_config import AblationConfig, CONFIGS, FINAL_TABLE_ROW_CONFIGS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "phase4_v1.jsonl"
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_metadata(settings: Settings, dataset_path: Path, n_rows: int) -> dict:
    """Phase 7 spec item 10 ("reproducible configuration"): every report
    self-describes exactly what produced it — dataset file, question
    count, timestamp, and the handful of base-settings values that
    affect results but are NOT part of any config's own `overrides`
    (embedding/answer model, vector-store backend) — so a report read
    months later doesn't require cross-referencing which script version
    or corpus state produced it. Deliberately NOT a full settings dump
    (most fields, e.g. API keys, are irrelevant to reproducibility and
    would just be noise/a secrets-leak risk in a checked-in report file).
    """
    import datetime as dt

    return {
        "dataset": str(dataset_path),
        "n_questions": n_rows,
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model,
        "answer_model": settings.answer_model,
        "vectorstore_backend": settings.vectorstore_backend,
    }


def _checkpoint_path(config_name: str) -> Path:
    """Reads the module-level `REPORTS_DIR` at CALL time (not captured
    into a default argument at import time), so a test that does
    `monkeypatch.setattr("evaluation.run_ablation.REPORTS_DIR", tmp_path)`
    redirects checkpoint writes too, exactly like it already redirects
    everything else `main()` writes.
    """
    return REPORTS_DIR / f"ablation_{config_name}.checkpoint.json"


def _load_checkpoint(config_name: str) -> dict | None:
    """Returns `None` for "no checkpoint" AND for "checkpoint file exists
    but is corrupt/unreadable" — a partially-written checkpoint (e.g. the
    process was killed mid-write, though `_write_checkpoint`'s atomic
    replace makes that unlikely) must never crash a `--resume` attempt;
    worst case, resuming just starts that config over from scratch.
    """
    path = _checkpoint_path(config_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_checkpoint(config_name: str, state: dict) -> None:
    """Atomic write (temp file + `Path.replace`, same pattern
    `app/storage/local.py::LocalStorageBackend.save` already uses) —
    a reader (a `--resume` invocation started while this one is still
    running, or a crash mid-write) never sees a torn/partial checkpoint,
    only the complete previous version or the complete new version.
    """
    path = _checkpoint_path(config_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _delete_checkpoint(config_name: str) -> None:
    """Called once a config's run reaches 100% completion — a leftover
    checkpoint for an already-fully-reported config would otherwise make
    a later `--resume` invocation ambiguous about whether that config
    still needs work."""
    path = _checkpoint_path(config_name)
    if path.exists():
        path.unlink()


def _rate_limit_path(config_name: str) -> Path:
    """Where the provider's own rate-limit error detail is persisted —
    see `run_config`'s except-branch. Distinct from the checkpoint file:
    the checkpoint is COMPLETED work, this is a record of the failure
    that stopped the run, kept only until the next successful
    `--resume` clears it (see `_delete_rate_limit_detail`).

    This closes a real, observed gap: `_print_rate_limit_report` only
    ever printed to stdout, so the provider's own stated retry window
    (e.g. Groq's "please try again in 23h59m" for a daily-quota hit, vs.
    a much shorter per-minute-limit message) was lost the moment the
    terminal scrollback was — leaving no way to tell, after the fact,
    which kind of limit was actually hit.
    """
    return REPORTS_DIR / f"ablation_{config_name}.ratelimit.json"


def _write_rate_limit_detail(config_name: str, report: dict) -> None:
    path = _rate_limit_path(config_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _delete_rate_limit_detail(config_name: str) -> None:
    """Called once a `--resume` run for this config completes (or fully
    finishes) without hitting the rate limit again — a stale rate-limit
    record left behind after a successful resume would misleadingly
    suggest the run is still blocked."""
    path = _rate_limit_path(config_name)
    if path.exists():
        path.unlink()


@dataclass
class _RateLimitHit:
    """Structured detail about a detected rate-limit failure — exactly
    the fields the spec's "stop gracefully and report" requirement asks
    for (config/last-completed-id/counts are added by the `run_config`
    caller, which is the one that actually knows them; this dataclass
    only carries what `_classify_rate_limit_error` itself determines
    from the exception).
    """

    provider: str  # "groq" | "gemini"
    detail: str
    status_code: int | None = None
    retry_after_seconds: float | None = None


def _classify_rate_limit_error(exc: Exception) -> "_RateLimitHit | None":
    """Best-effort detection of a rate-limit-shaped failure from either
    provider SDK. Returns `None` for anything else, so the caller
    re-raises unknown/unexpected exceptions with their REAL traceback
    intact -- this function's whole job is telling "the quota is
    exhausted, pause and let the user resume later" apart from "there is
    an actual bug in this code", and it must never accidentally swallow
    the latter.

    Deliberately its own small classifier here rather than reusing/
    extending `app.rag.llm`'s or `app.api.ask`'s `_UPSTREAM_ERRORS`
    handling: those exist to turn a provider failure into an HTTP 503
    for ONE live request. This one exists to pause a long-running BATCH
    job and preserve everything completed so far -- a different action
    entirely, needing different information (retry-after, how many
    questions are left) than an HTTP error response does. Per the
    spec's "keep the existing metrics and configuration logic
    unchanged", neither of those modules is touched by this file.

    Both provider SDKs are imported lazily (inside this function, not at
    module level) -- consistent with how the rest of this codebase
    already treats them as optional-until-actually-used dependencies
    (e.g. `app/services/ingestion_jobs.py`'s own lazy `import groq`).
    """
    try:
        import groq
    except ImportError:
        groq = None
    try:
        from google.genai import errors as genai_errors
    except ImportError:
        genai_errors = None

    if groq is not None and isinstance(exc, groq.APIError):
        status_code = getattr(exc, "status_code", None)
        looks_like_rate_limit = (
            status_code == 429
            or type(exc).__name__ == "RateLimitError"
            or "rate limit" in str(exc).lower()
            or "rate_limit" in str(exc).lower()
        )
        if not looks_like_rate_limit:
            return None
        retry_after_seconds = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        raw_retry_after = headers.get("retry-after") if headers and hasattr(headers, "get") else None
        if raw_retry_after:
            try:
                retry_after_seconds = float(raw_retry_after)
            except (TypeError, ValueError):
                retry_after_seconds = None
        return _RateLimitHit(
            provider="groq", detail=str(exc), status_code=status_code, retry_after_seconds=retry_after_seconds
        )

    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        text = str(exc)
        looks_like_rate_limit = "RESOURCE_EXHAUSTED" in text or "429" in text or "rate limit" in text.lower()
        if not looks_like_rate_limit:
            return None
        return _RateLimitHit(provider="gemini", detail=text, status_code=429)

    return None


def _print_rate_limit_report(report: dict) -> None:
    """The exact human-readable report the spec asks for: config, last
    completed question id, completed/remaining count, provider, and
    retry information -- printed to stdout, never a raw traceback."""
    print(f"\n{'=' * 70}")
    print(f"RATE LIMIT HIT -- Config {report['config']} ({report['label']})")
    print(f"{'=' * 70}")
    print(f"  Provider:                {report['provider']}")
    print(f"  HTTP status:             {report.get('status_code')}")
    retry_after = report.get("retry_after_seconds")
    print(f"  Retry-After (from API):  {retry_after if retry_after is not None else 'not provided by API'}")
    print(f"  Last completed question: {report.get('last_completed_question_id') or '(none completed yet)'}")
    print(f"  Completed / Remaining:   {report['completed_count']} / {report['remaining_count']} (of {report['total_count']})")
    print(f"  Checkpoint saved to:     {report['checkpoint_path']}")
    print(f"  Full detail also saved to: {_rate_limit_path(report['config'])}")
    print(f"  Detail:                  {report['detail']}")
    print()
    print(f"  Every completed question's result for this config is safely checkpointed.")
    print(f"  Wait for the rate limit / quota window to reset, then resume with:")
    print(f"      python -m evaluation.run_ablation --configs {report['config']} --resume")
    print(f"{'=' * 70}\n")


def run_config(config: AblationConfig, dataset: list[dict], base_settings: Settings, dataset_path: Path | None = None, resume: bool = False) -> dict:
    """Runs one ablation config end-to-end. See the module docstring for
    the resume/checkpoint/rate-limit-handling contract this adds on top
    of the original (still-unchanged) per-question logic below.

    **Checkpointing (spec items 1-2)**: after EVERY question, the
    complete in-progress state (`results`, `latencies`, reranker/usage
    accumulators) is atomically written to
    `evaluation/reports/ablation_<config>.checkpoint.json` via
    `_write_checkpoint`. If the process is killed at any point --
    rate-limited, crashed, `Ctrl-C` -- everything up to the last
    successfully answered question is already durably on disk; nothing
    completed is ever lost.

    **Resuming (spec items 3, 5)**: `resume=True` loads that checkpoint
    (if present and for the SAME dataset file -- a checkpoint for a
    different dataset is never silently reused) and skips every question
    whose `id` already has a result, continuing only with what's left.

    **Rate-limit handling (spec item 4)**: if `service.ask()` raises an
    exception `_classify_rate_limit_error` recognizes as a genuine
    provider rate-limit, this function does NOT propagate a raw
    traceback. It checkpoints immediately (so the failing question isn't
    even re-attempted from scratch on the next `--resume`) and returns a
    `{"status": "rate_limited", ...}` dict instead of the normal report
    shape -- `main()` detects this and prints the structured report (see
    `_print_rate_limit_report`) before exiting. Any OTHER exception
    (a real bug, a malformed response, etc.) is deliberately NOT caught
    here and propagates with its real traceback, per spec item 4's
    "transient API failures" scope -- this is not a general
    try/except-everything wrapper.

    **Nothing about the normal-completion return shape changed**: when a
    run finishes (whether in one shot or across several `--resume`
    invocations), the returned dict is byte-for-byte the same shape this
    function already returned before this pass -- see
    `tests/test_run_ablation.py`'s existing assertions, none of which
    needed to change.
    """
    settings = base_settings.model_copy(update=config.overrides)
    service = DocumentQAService(settings=settings)
    effective_dataset_path = str(dataset_path or DEFAULT_DATASET)

    results: list[dict] = []
    latencies: list[float] = []
    reranker_calls = 0
    reranker_fallbacks = 0
    llm_calls_all: list[dict] = []

    if resume:
        checkpoint = _load_checkpoint(config.name)
        if checkpoint is not None:
            if checkpoint.get("dataset_path") == effective_dataset_path:
                results = checkpoint.get("results", [])
                latencies = checkpoint.get("latencies", [])
                reranker_calls = checkpoint.get("reranker_calls", 0)
                reranker_fallbacks = checkpoint.get("reranker_fallbacks", 0)
                llm_calls_all = checkpoint.get("llm_calls_all", [])
                print(
                    f"  [resume] config {config.name}: loaded checkpoint with "
                    f"{len(results)} question(s) already completed"
                )
            else:
                print(
                    f"  [resume] config {config.name}: found a checkpoint, but it was for a "
                    f"different dataset ({checkpoint.get('dataset_path')!r} != {effective_dataset_path!r}) "
                    f"-- ignoring it and starting this config from scratch"
                )

    def _checkpoint_now() -> None:
        _write_checkpoint(
            config.name,
            {
                "config": config.name,
                "dataset_path": effective_dataset_path,
                "results": results,
                "latencies": latencies,
                "reranker_calls": reranker_calls,
                "reranker_fallbacks": reranker_fallbacks,
                "llm_calls_all": llm_calls_all,
            },
        )

    completed_ids = {r["id"] for r in results}
    remaining_rows = [row for row in dataset if row["id"] not in completed_ids]

    for row in remaining_rows:
        # Phase 7 completion pass — trace/token/cost integration gap
        # (flagged, not yet fixed, in the prior session's continuation
        # notes): `service.ask()` calls straight into
        # `app.rag.llm`'s `_record_usage`, which does
        # `trace = current_trace(); if trace is None: return` — with NO
        # trace ever started here, every token/cost record was silently
        # a no-op for the whole lifetime of this runner. `start_trace()`
        # is normally called once per HTTP request by
        # `app/api/ask.py`/`app/api/ask_stream.py`; this runner IS its
        # own "request" per benchmark question, so it now does the same
        # thing those do, explicitly, per question.
        #
        # What this DOES fix: prompt/completion/total tokens and
        # estimated cost per LLM call, attributed by `purpose` (grade/
        # rewrite/answer/grounding/...), aggregated below.
        #
        # What this does NOT fix, and cannot: TTFT (`mark_first_token()`)
        # is only ever called from `app/api/ask_stream.py` — it is a
        # streaming-specific concept (the moment the first output token
        # is available to the CLIENT) that has no meaning for
        # `service.ask()`'s synchronous, non-streaming call shape. TTFT
        # is reported as `null`/absent here, not approximated from
        # total latency — see `summary["usage"]`'s docstring-equivalent
        # comment below.
        from app.observability.trace import start_trace

        req_trace = start_trace()
        started = time.monotonic()
        try:
            outcome = service.ask(row["question"])
        except Exception as exc:  # noqa: BLE001 - classified below; unrecognized exceptions re-raise untouched
            rate_limit = _classify_rate_limit_error(exc)
            if rate_limit is None:
                raise
            _checkpoint_now()
            rate_limited_report = {
                "status": "rate_limited",
                "config": config.name,
                "label": config.label,
                "provider": rate_limit.provider,
                "status_code": rate_limit.status_code,
                "retry_after_seconds": rate_limit.retry_after_seconds,
                "detail": rate_limit.detail,
                "last_completed_question_id": results[-1]["id"] if results else None,
                "completed_count": len(results),
                "remaining_count": len(dataset) - len(results),
                "total_count": len(dataset),
                "checkpoint_path": str(_checkpoint_path(config.name)),
            }
            # Persisted to disk (not just printed — see
            # `_rate_limit_path`'s docstring for the exact gap this
            # closes: the provider's own stated retry window, which is
            # often the ONLY place the difference between a short
            # per-minute limit and a long per-day quota is stated, used
            # to only ever reach the terminal, not a file).
            _write_rate_limit_detail(config.name, rate_limited_report)
            return rate_limited_report
        latencies.append(time.monotonic() - started)
        llm_calls_all.extend(req_trace.to_dict()["llm_calls"])

        cited_ids = [c["chunk_id"] for c in outcome["citations"]] if outcome["citations"] else []

        # Phase 3A hardening (problem #2): `service.ask()` now returns the
        # full per-stage candidate breakdown (dense/BM25/fused/reranked),
        # captured straight from the graph's final state — not
        # re-derived or approximated — so Recall@K/MRR/nDCG below are
        # computed against what retrieval actually returned at each
        # stage, not just the final validated citations.
        reranked_ids = [c["chunk_id"] for c in outcome.get("reranked_candidates") or []]
        fused_ids = [c["chunk_id"] for c in outcome.get("fused_candidates") or []]
        # `retrieved_chunk_ids` (consumed by app/evaluation/metrics.py's
        # retrieval_metrics_mean) is the last ranking the pipeline actually
        # acted on before grading: reranked output when reranking ran,
        # otherwise the fused dense+BM25 candidate set.
        retrieved_ids = reranked_ids or fused_ids

        if outcome.get("actual_reranker") not in (None, "", "none"):
            reranker_calls += 1
            if outcome.get("fallback_used"):
                reranker_fallbacks += 1

        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "query_type": row["query_type"],
                "answerable": row["answerable"],
                "found": outcome["found"],
                "cited_chunk_ids": cited_ids,
                "required_chunk_ids": row["required_chunks"],
                "retrieved_chunk_ids": retrieved_ids,
                "dense_chunk_ids": [c["chunk_id"] for c in outcome.get("dense_candidates") or []],
                "bm25_chunk_ids": [c["chunk_id"] for c in outcome.get("bm25_candidates") or []],
                "fused_chunk_ids": fused_ids,
                "reranked_chunk_ids": reranked_ids,
                "grounded_claim_rate": outcome.get("grounded_claim_rate"),
                "citation_precision": outcome.get("citation_precision"),
                # Phase 7 dashboard consistency fix: `citation_recall_mean`
                # (app/evaluation/metrics.py, feeding dashboard.py's
                # "Citation recall (mean)" row) reads this exact key from
                # each result — it was never recorded here even though
                # `outcome["citation_recall"]` (app/rag/graph.py's `ask()`)
                # already computes it for free, so that row could never
                # have rendered anything but "—" regardless of a real run.
                "citation_recall": outcome.get("citation_recall"),
                "configured_reranker": outcome.get("configured_reranker"),
                "actual_reranker": outcome.get("actual_reranker"),
                "fallback_used": outcome.get("fallback_used"),
                # Phase 7 completion pass: per-question token/cost, now
                # that start_trace()/current_trace() actually captures
                # something during this runner's execution (see comment
                # above the start_trace() call). Empty list is a
                # genuinely different, honest signal from "not measured"
                # — it means the LLM SDK response for every call this
                # question made had no `.usage` attribute (e.g. a
                # provider that doesn't return usage data), not that
                # tracing itself failed.
                "llm_calls": req_trace.to_dict()["llm_calls"],
            }
        )
        # spec item 2: checkpoint after EVERY question, not just at the
        # end — see this function's own docstring.
        _checkpoint_now()

    summary = summarize(results, latencies_seconds=latencies)
    # Phase 4.1(7) item 4: attach a modality breakdown (text/ocr/table/
    # figure/chart/visual/multi_hop/cross_modal/unanswerable) on top of
    # the existing aggregate summary — additive only, computed by
    # `evaluation.modality` from fields already present on `results`, so
    # this never changes what a plain `summary` looked like before.
    from evaluation.modality import summarize_by_modality

    latencies_by_id = {row["id"]: lat for row, lat in zip(dataset, latencies)}
    summary["modality_breakdown"] = summarize_by_modality(dataset, results, latencies_by_id)
    # Phase 3A hardening (problem #3): record whether ANY question in this
    # run silently fell back from the configured reranker, at the
    # aggregate level — so a report can never claim "Cross Encoder"
    # performance for a run where the model was actually unavailable for
    # some or all questions.
    if settings.rerank_enabled:
        configured_label = "cross_encoder" if settings.cross_encoder_enabled else "bm25"
    else:
        configured_label = "none"
    summary["reranker_integrity"] = {
        "configured_reranker": configured_label,
        "reranker_invocations": reranker_calls,
        "fallback_invocations": reranker_fallbacks,
        "fallback_rate": round(reranker_fallbacks / reranker_calls, 4) if reranker_calls else None,
    }

    # Phase 7 completion pass: real token/cost aggregation, now that
    # start_trace() actually runs per question (see the loop above).
    # `status` is always present and explicit — never silently 0 when
    # nothing was captured. "MEASURED" requires at least one real
    # LLM_calls record; a run with zero (e.g. every question refused
    # before any LLM call, or the provider's SDK response genuinely
    # carried no `.usage`) is reported as "UNAVAILABLE" with a reason,
    # not a fabricated zero-cost total.
    if llm_calls_all:
        total_prompt = sum(c["prompt_tokens"] for c in llm_calls_all)
        total_completion = sum(c["completion_tokens"] for c in llm_calls_all)
        by_purpose: dict[str, dict] = {}
        for purpose in sorted({c["purpose"] for c in llm_calls_all}):
            calls = [c for c in llm_calls_all if c["purpose"] == purpose]
            by_purpose[purpose] = {
                "n_calls": len(calls),
                "total_tokens": sum(c["total_tokens"] for c in calls),
                "total_cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
            }
        summary["usage"] = {
            "status": "MEASURED",
            "n_llm_calls": len(llm_calls_all),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": round(sum(c["cost_usd"] for c in llm_calls_all), 6),
            "by_purpose": by_purpose,
            # See app/observability/cost.py's own module docstring before
            # trusting total_cost_usd for anything beyond rough
            # order-of-magnitude — it's a configured PRICING ESTIMATE
            # applied to real measured token counts, not a provider
            # -reported bill.
            "cost_is_estimated_not_billed": True,
        }
    else:
        summary["usage"] = {
            "status": "UNAVAILABLE",
            "reason": (
                "No LLM response in this run exposed a `.usage` attribute (or every question "
                "was refused before any LLM call was made) — token/cost genuinely were not "
                "captured, not silently reported as zero."
            ),
        }

    # TTFT is architecturally not measurable through this runner — see
    # the comment above the start_trace() call in the loop. Reported
    # explicitly rather than omitted, so a reader doesn't have to infer
    # its absence from a missing key.
    summary["ttft"] = {
        "status": "NOT_APPLICABLE",
        "reason": (
            "mark_first_token() is only called from app/api/ask_stream.py (the streaming "
            "endpoint) — TTFT has no meaning for DocumentQAService.ask()'s synchronous, "
            "non-streaming call shape, which is what this runner uses."
        ),
    }

    # spec item 1: every question in `dataset` was either resumed from a
    # prior checkpoint or just answered above — this config is now fully
    # complete, so its checkpoint (whose only purpose was surviving an
    # interruption mid-run) is no longer needed. A stale leftover here
    # would otherwise make a later `--resume` invocation ambiguous about
    # whether this config still has work left.
    _delete_checkpoint(config.name)
    # Same reasoning for a leftover rate-limit record from an EARLIER
    # attempt at this config: this run just finished without hitting the
    # limit again, so any prior "we stopped here" detail is now stale.
    _delete_rate_limit_detail(config.name)

    return {
        "config": config.name,
        "label": config.label,
        "overrides": config.overrides,
        "run_metadata": _run_metadata(settings, dataset_path or DEFAULT_DATASET, len(dataset)),
        "summary": summary,
        "results": results,
    }


def load_multi_doc_dataset(path: Path) -> list[dict]:
    """Load a Config-G ("multi-document reasoning") benchmark row set —
    each row is `{"id", "topic", "documents": ["filename1", ...],
    "expected_relations": {"<aspect substring>": "one_sided"|"both_sided"}}`.
    `expected_relations`'s value is checked against
    `evidence_by_document`'s key COUNT for a matched point (see
    `run_multi_doc_ablation`'s `_shape_matches`): `"one_sided"` means
    exactly one document contributed evidence to that point, `"both_sided"`
    means two or more did. (This is the actual contract the matching code
    below checks — an earlier version of this docstring claimed
    `"same"|"different"|"missing"|"conflicting"` values, which the code
    never actually read; fixed during the Phase 6 completion pass'
    run_ablation.py reconciliation.) See
    `evaluation/datasets/multi_doc_v1.jsonl` for the current seed set and
    its header comment on why it's intentionally small.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_multi_doc_ablation(dataset: list[dict], base_settings: Settings, dataset_path: Path | None = None) -> dict:
    """Config G ("multi-document reasoning" — Phase 6 spec item 12).

    Runs the REAL `app.rag.comparison.compare_multiple` pipeline (same
    function `app/api/reasoning.py`'s `POST /compare` calls for 2+
    documents — this is not a separate/simplified re-implementation)
    against `dataset`, resolving each row's `documents` (filenames)
    against the corpus. Not folded into `run_config`/`CONFIGS` above —
    see this module's docstring for why multi-document reasoning isn't a
    retrieval-setting ablation over the SAME dataset shape as A-F.

    Measures, per row: for each of the row's `expected_relations`
    (`{"<aspect substring>": "one_sided" | "both_sided"}`), whether
    `compare_multiple` produced a matching point with the expected
    evidence coverage shape — "one_sided" means only one document
    contributed evidence to that point (the honest, verifiable signal
    for "the topic is genuinely one-sided in this corpus", checked
    directly against `evidence_by_document`'s keys rather than against
    the model's own free-text `consensus` label, which this measurement
    deliberately doesn't need to agree with verbatim). Also reports
    citation coverage: does every returned point actually carry evidence
    for every document it claims to compare (enforced by
    `compare_multiple` itself via its fail-closed handling — reported
    anyway to actually verify that guarantee held for this run rather
    than trust it silently).
    """
    from sqlalchemy import select

    from app.clients import get_chat, get_embeddings, get_index, get_pinecone
    from app.db.models import Document
    from app.db.session import session_scope
    from app.rag.comparison import ComparisonError, compare_multiple
    from app.rag.multi_doc import ResolvedDocument

    settings = base_settings
    chat = get_chat(settings)
    embeddings = get_embeddings(settings)
    pc = get_pinecone(settings)
    index = get_index(pc, settings)

    results = []
    for row in dataset:
        with session_scope(settings) as session:
            docs = session.execute(select(Document).where(Document.filename.in_(row["documents"]))).scalars().all()
            resolved = [
                ResolvedDocument(
                    id=d.id, filename=d.filename, title=d.title or d.filename,
                    source_root=d.source_root, file_type=d.file_type,
                )
                for d in docs
            ]
        missing_filenames = set(row["documents"]) - {r.filename for r in resolved}
        if missing_filenames:
            results.append({"id": row["id"], "error": f"documents not found in DB: {sorted(missing_filenames)}"})
            continue

        try:
            comparison = compare_multiple(index, embeddings, chat, settings, row["topic"], resolved, allowed_owner_ids=None)
        except ComparisonError as exc:
            results.append({"id": row["id"], "error": str(exc)})
            continue

        points = comparison["points"]
        points_with_full_evidence = sum(1 for p in points if p.get("evidence_by_document"))
        total_points = len(points)

        matched = 0
        for substring, expected_shape in row.get("expected_relations", {}).items():
            def _shape_matches(p: dict) -> bool:
                n_docs_with_evidence = len(p.get("evidence_by_document") or {})
                if expected_shape == "one_sided":
                    return n_docs_with_evidence == 1
                if expected_shape == "both_sided":
                    return n_docs_with_evidence >= 2
                return False

            hit = any(substring.lower() in p.get("aspect", "").lower() and _shape_matches(p) for p in points)
            matched += int(hit)
        expected_total = len(row.get("expected_relations", {}))

        results.append(
            {
                "id": row["id"],
                "topic": row["topic"],
                "points_found": total_points,
                "points_with_evidence": points_with_full_evidence,
                "citation_coverage": round(points_with_full_evidence / total_points, 4) if total_points else None,
                "relation_matches": matched,
                "relation_expected": expected_total,
                "relation_accuracy": round(matched / expected_total, 4) if expected_total else None,
                "comparison": comparison,
            }
        )

    valid = [r for r in results if "error" not in r]
    summary = {
        "n_rows": len(dataset),
        "n_errored": len(results) - len(valid),
        "mean_citation_coverage": (
            round(sum(r["citation_coverage"] for r in valid if r["citation_coverage"] is not None) / len(valid), 4)
            if valid
            else None
        ),
        "mean_relation_accuracy": (
            round(sum(r["relation_accuracy"] for r in valid if r["relation_accuracy"] is not None) / len(valid), 4)
            if valid
            else None
        ),
    }
    return {
        "config": "G",
        "label": "multi-document reasoning (compare_multiple pipeline)",
        "run_metadata": _run_metadata(
            base_settings, dataset_path or (REPO_ROOT / "evaluation" / "datasets" / "multi_doc_v1.jsonl"), len(dataset)
        ),
        "summary": summary,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--multi-doc",
        action="store_true",
        help="Run Config G (multi-document reasoning, evaluation/datasets/multi_doc_v1.jsonl) instead of A-F.",
    )
    parser.add_argument(
        "--multi-doc-dataset", type=Path, default=REPO_ROOT / "evaluation" / "datasets" / "multi_doc_v1.jsonl"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from evaluation/reports/ablation_<config>.checkpoint.json for any config that "
            "was interrupted (e.g. by a rate limit) mid-run, skipping already-completed questions. "
            "A config with no checkpoint but an existing final ablation_<config>.json report is "
            "treated as already fully done and skipped entirely (zero extra API calls). See this "
            "module's docstring for the full resume workflow."
        ),
    )
    args = parser.parse_args()

    base_settings = get_settings()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.multi_doc:
        dataset = load_multi_doc_dataset(args.multi_doc_dataset)
        print(f"Loaded {len(dataset)} multi-document benchmark rows from {args.multi_doc_dataset}")
        report = run_multi_doc_ablation(dataset, base_settings, dataset_path=args.multi_doc_dataset)
        out_path = REPORTS_DIR / "ablation_G.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  mean_citation_coverage={report['summary']['mean_citation_coverage']} "
              f"mean_relation_accuracy={report['summary']['mean_relation_accuracy']}")
        print(f"  -> {out_path}")
        return

    dataset = load_dataset(args.dataset)

    print(f"Loaded {len(dataset)} benchmark questions from {args.dataset}")
    all_summaries = {}
    for name in args.configs:
        config = CONFIGS[name]
        out_path = REPORTS_DIR / f"ablation_{name}.json"

        if args.resume and out_path.exists() and _load_checkpoint(name) is None:
            # spec item 3/5: a config with a final report already on disk
            # and no leftover checkpoint was already fully completed in a
            # prior invocation — resuming it would waste real API calls
            # re-answering questions that already have a valid result.
            print(f"\n=== Config {name}: {config.label} ===")
            print(f"  [resume] {out_path} already exists and no checkpoint is pending — skipping, 0 API calls made.")
            all_summaries[name] = json.loads(out_path.read_text(encoding="utf-8"))["summary"]
            continue

        print(f"\n=== Config {name}: {config.label} ===")
        report = run_config(config, dataset, base_settings, dataset_path=args.dataset, resume=args.resume)

        if report.get("status") == "rate_limited":
            # spec item 4: stop gracefully, report structured detail, no
            # raw traceback — and do NOT touch any other config or write
            # a (necessarily incomplete) final report for this one. The
            # checkpoint `run_config` already wrote is this config's only
            # on-disk state until a `--resume` run completes it.
            _print_rate_limit_report(report)
            sys.exit(1)

        all_summaries[name] = report["summary"]

        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  refusal_accuracy={report['summary']['refusal_accuracy']} "
              f"grounded_answer_rate={report['summary']['grounded_answer_rate']} "
              f"citation_precision={report['summary']['citation_precision']}")
        retrieval = report["summary"].get("retrieval")
        if retrieval:
            print(f"  retrieval: {retrieval}")
        integrity = report["summary"].get("reranker_integrity")
        if integrity and integrity["reranker_invocations"]:
            print(
                f"  reranker_integrity: configured={integrity['configured_reranker']} "
                f"fallback_rate={integrity['fallback_rate']}"
            )
        print(f"  -> {out_path}")

    combined_path = REPORTS_DIR / "ablation_summary.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nCombined summary written to {combined_path}")


if __name__ == "__main__":
    main()