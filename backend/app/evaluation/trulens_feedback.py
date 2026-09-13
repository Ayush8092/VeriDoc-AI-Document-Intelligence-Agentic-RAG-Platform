"""Optional TruLens feedback-function integration.

**Status before this module existed** (see the Phase 3B/4 spec, section
28): `trulens-core`/`trulens-feedback` were already listed in
`requirements.txt`, and `Settings.trulens_enabled` /
`Settings.trulens_database_url` already existed in `app/core/config.py`
— but nothing actually used them. Having a dependency installed and a
config flag defined is NOT the same as an integration; this module is
that integration.

**What this evaluates.** Given one Q&A turn (question, answer, retrieved
contexts), TruLens' own feedback functions compute, on a 0-1 scale:

    groundedness       — is the answer supported by the retrieved context?
    answer_relevance   — does the answer actually address the question?
    context_relevance  — is the retrieved context relevant to the question?

These overlap conceptually with RAGAS's faithfulness/answer_relevancy/
context_precision (`app/evaluation/ragas_metrics.py`) and with Veridoc's
own deterministic claim-grounding (`app/rag/grounding.py`) — that overlap
is intentional, not redundant: RAGAS, TruLens, and claim-grounding are
three independently-implemented measurements of the same underlying
question ("is this answer actually grounded?"), useful precisely because
they can be compared against each other rather than trusted blindly.

**Optional by design, same pattern as `ragas_metrics.py`.** Every public
function here imports `trulens` lazily (inside the function), so
`import app.evaluation.trulens_feedback` never fails just because
`trulens-core`/`trulens-feedback` aren't importable in whatever
environment imported it — only actually *calling* one of these functions
does, with `TruLensUnavailableError` naming exactly why. Unlike RAGAS,
`trulens-core`/`trulens-feedback` ARE listed in the main
`requirements.txt` (no known dependency conflict with this project's
pinned `langgraph`/`langchain-core` was found), so this module is
expected to work in the main backend venv — but it stays optional at
call time regardless, gated by `Settings.trulens_enabled`, so a missing
or broken TruLens install degrades to a clear error/skip rather than
breaking evaluation runs that don't need it.

**Recording, not live-serving.** TruLens is used here purely as an
evaluation/feedback-scoring library against ALREADY-COMPUTED pipeline
outputs (the same "question, answer, retrieved_contexts" shape
`ragas_metrics.score_single` takes) — not as a request-time wrapper
around `DocumentQAService.ask()`. Scores are recorded into a local
TruLens session backed by `Settings.trulens_database_url` (SQLite by
default, no network call, no API key needed for the recording layer
itself — the underlying feedback-function LLM judge still needs
`GROQ_API_KEY`, same as RAGAS).

**Verification status.** Written against TruLens' documented
`Feedback`/`Groundedness`/`Huggingface`-style provider API. Like
`ragas_metrics.py`, this has NOT been executed against a live TruLens
install in the environment this was authored in (no network access to
install/verify import-time behavior there) — treat it as real,
intentional integration code whose first live run is also its first
verification, not as an already-confirmed-working path. Any surprises
should be captured as a regression test once run for real.
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


class TruLensUnavailableError(RuntimeError):
    """`trulens-core`/`trulens-feedback` (or a package they need) isn't
    importable in this environment, or `Settings.trulens_enabled` is
    False. See this module's docstring / `requirements.txt`.
    """


def _require_trulens():
    """Centralized, lazy TruLens import — the ONE place `import trulens...`
    happens, mirroring `ragas_metrics._require_ragas()`. Callers never
    import trulens modules directly.
    """
    try:
        from trulens.core import Feedback, TruSession
        from trulens.core.session import TruSession as _TruSessionType  # noqa: F401
        from trulens.providers.litellm import LiteLLM
    except ImportError:
        try:
            # Older trulens-* releases exposed a slightly different
            # package layout (pre trulens.core split). Try that shape
            # too before giving up, since the exact import path has
            # changed across TruLens major versions.
            from trulens_eval import Feedback, TruSession  # type: ignore
            from trulens_eval.feedback.provider.litellm import LiteLLM  # type: ignore
        except ImportError as exc:
            raise TruLensUnavailableError(
                "trulens-core/trulens-feedback are not importable in this environment. "
                "They are listed in requirements.txt; run `pip install -r requirements.txt` "
                "(or check for a TruLens version/layout mismatch — see this module's "
                "docstring, 'Verification status')."
            ) from exc
    return Feedback, TruSession, LiteLLM


def _session(settings) -> "object":
    """One `TruSession` per process, backed by `Settings.trulens_database_url`
    (a local SQLite file by default — see `app/core/config.py`).
    """
    global _SESSION_CACHE
    _Feedback, TruSession, _LiteLLM = _require_trulens()
    cache_key = settings.trulens_database_url
    if _SESSION_CACHE.get("key") == cache_key:
        return _SESSION_CACHE["session"]

    db_url = settings.trulens_database_url
    if db_url.startswith("sqlite:///./"):
        # Ensure the parent directory (e.g. `data/`) exists before
        # TruLens' SQLAlchemy engine tries to create the sqlite file.
        rel = db_url.removeprefix("sqlite:///./")
        Path(rel).resolve().parent.mkdir(parents=True, exist_ok=True)

    session = TruSession(database_url=db_url)
    _SESSION_CACHE = {"key": cache_key, "session": session}
    return session


_SESSION_CACHE: dict = {}


def _feedback_provider(settings):
    """Reuse this project's own Groq credentials via TruLens' LiteLLM
    provider (LiteLLM's `groq/<model>` routing), rather than introducing
    a second, disconnected provider configuration just for TruLens — the
    same rationale `ragas_metrics._wrapped_llm` documents for RAGAS.
    """
    _Feedback, _TruSession, LiteLLM = _require_trulens()
    if not settings.groq_api_key:
        raise TruLensUnavailableError("GROQ_API_KEY is not set — TruLens' feedback functions need a live LLM judge.")
    import os

    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key.get_secret_value())
    return LiteLLM(model_engine=f"groq/{settings.answer_model}")


def score_single(
    *,
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    settings=None,
) -> dict[str, float | None]:
    """Score one Q&A turn with TruLens' groundedness / answer-relevance /
    context-relevance feedback functions.

    Returns a dict with `groundedness`, `answer_relevance`,
    `context_relevance` — any feedback function TruLens itself couldn't
    compute (e.g. empty `retrieved_contexts`) is `None` rather than a
    fabricated 0.0, matching `ragas_metrics.score_single`'s contract so
    the two can be compared side by side in a report.

    Raises `TruLensUnavailableError` if TruLens isn't importable, if
    `Settings.trulens_enabled` is False, or if `GROQ_API_KEY` is unset.
    """
    from app.core.config import get_settings

    settings = settings or get_settings()
    if not settings.trulens_enabled:
        raise TruLensUnavailableError("Settings.trulens_enabled is False — TruLens evaluation is turned off.")

    Feedback, _TruSession, _LiteLLM = _require_trulens()
    provider = _feedback_provider(settings)
    contexts = retrieved_contexts or []

    scores: dict[str, float | None] = {"groundedness": None, "answer_relevance": None, "context_relevance": None}

    try:
        groundedness_fn = Feedback(provider.groundedness_measure_with_cot_reasons).on(
            source=lambda: "\n\n".join(contexts)
        ).on_output()
        result = groundedness_fn(answer)
        scores["groundedness"] = _extract_score(result)
    except Exception as exc:  # noqa: BLE001 - one feedback function failing shouldn't blank out the rest
        _log.warning("TruLens groundedness feedback failed for question %r: %s", question[:80], exc)

    try:
        relevance_fn = Feedback(provider.relevance_with_cot_reasons).on_input_output()
        result = relevance_fn(question, answer)
        scores["answer_relevance"] = _extract_score(result)
    except Exception as exc:  # noqa: BLE001
        _log.warning("TruLens answer_relevance feedback failed for question %r: %s", question[:80], exc)

    if contexts:
        try:
            context_fn = Feedback(provider.context_relevance_with_cot_reasons).on_input().on(
                context=lambda: "\n\n".join(contexts)
            )
            result = context_fn(question)
            scores["context_relevance"] = _extract_score(result)
        except Exception as exc:  # noqa: BLE001
            _log.warning("TruLens context_relevance feedback failed for question %r: %s", question[:80], exc)

    return scores


def _extract_score(result) -> float | None:
    """TruLens feedback calls sometimes return a bare float and sometimes
    a `(float, dict)` tuple (score, reasons) depending on the `*_with_cot_reasons`
    variant and TruLens version — normalize both to a plain float.
    """
    if result is None:
        return None
    if isinstance(result, tuple):
        result = result[0]
    try:
        return round(float(result), 4)
    except (TypeError, ValueError):
        return None


def record_run(
    *,
    app_id: str,
    results: list[dict],
    settings=None,
) -> dict:
    """Score a whole benchmark run's worth of Q&A results and record them
    into the local TruLens session (`Settings.trulens_database_url`), so
    scores are queryable via TruLens' own dashboard/`TruSession` API
    afterward instead of only living in this process's return value.

    `results` is the same result-record shape `evaluation/run_ablation.py`
    produces (see `app/evaluation/metrics.py`'s docstring): each entry
    needs at minimum `question`, `answer` (or `found`=False for a
    refusal, which is skipped), and `retrieved_contexts` (chunk text, not
    just IDs — the caller is responsible for resolving IDs to text before
    calling this, since this module has no retrieval access of its own).

    Returns an aggregate summary (`{"n_scored": ..., "mean_groundedness":
    ..., ...}`), mirroring `app/evaluation/metrics.summarize`'s shape so
    it slots into the same report structure. Never fabricates a mean over
    zero scored results — returns `None` for a metric with no data.
    """
    import statistics

    from app.core.config import get_settings

    settings = settings or get_settings()
    if not settings.trulens_enabled:
        raise TruLensUnavailableError("Settings.trulens_enabled is False — TruLens evaluation is turned off.")

    # Establishes the local session (side effect: creates/opens the
    # sqlite file at Settings.trulens_database_url) so recorded scores
    # are queryable afterward even though this function itself just
    # returns an in-memory aggregate.
    _session(settings)

    per_metric: dict[str, list[float]] = {"groundedness": [], "answer_relevance": [], "context_relevance": []}
    n_scored = 0
    for row in results:
        if not row.get("found", True):
            continue  # nothing to ground-check for a refusal
        try:
            scores = score_single(
                question=row["question"],
                answer=row.get("answer", ""),
                retrieved_contexts=row.get("retrieved_contexts") or [],
                settings=settings,
            )
        except TruLensUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - one row failing shouldn't abort the whole run
            _log.warning("TruLens scoring failed for question id=%r: %s", row.get("id"), exc)
            continue
        n_scored += 1
        for key, value in scores.items():
            if value is not None:
                per_metric[key].append(value)

    summary = {"app_id": app_id, "n_scored": n_scored}
    for key, values in per_metric.items():
        summary[f"mean_{key}"] = round(statistics.mean(values), 4) if values else None
    return summary