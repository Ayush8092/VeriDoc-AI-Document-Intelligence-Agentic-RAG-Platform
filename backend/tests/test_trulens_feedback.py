"""Tests for app/evaluation/trulens_feedback.py.

Mirrors the spirit of how ragas_metrics.py is exercised (see its
docstring): trulens itself may or may not be importable in the test
environment, so these tests only assert the parts that are safe to
assert either way — module import never fails, disabled-by-config raises
a clear typed error, and the score-normalization helper is pure and
independent of whether trulens is installed.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings


def test_module_imports_without_trulens_installed():
    """Importing the module must never require trulens to be installed —
    only calling score_single()/record_run() should, same contract as
    app.evaluation.ragas_metrics.
    """
    import app.evaluation.trulens_feedback as mod

    assert hasattr(mod, "score_single")
    assert hasattr(mod, "record_run")
    assert hasattr(mod, "TruLensUnavailableError")


def test_score_single_raises_when_disabled():
    from app.evaluation.trulens_feedback import TruLensUnavailableError, score_single

    settings = Settings(trulens_enabled=False)
    with pytest.raises(TruLensUnavailableError):
        score_single(question="q", answer="a", retrieved_contexts=["c"], settings=settings)


def test_record_run_raises_when_disabled():
    from app.evaluation.trulens_feedback import TruLensUnavailableError, record_run

    settings = Settings(trulens_enabled=False)
    with pytest.raises(TruLensUnavailableError):
        record_run(app_id="veridoc-test", results=[], settings=settings)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.75, 0.75),
        ((0.5, {"reason": "because"}), 0.5),
        (None, None),
        ("not-a-number", None),
    ],
)
def test_extract_score_normalizes_variants(raw, expected):
    from app.evaluation.trulens_feedback import _extract_score

    assert _extract_score(raw) == expected


def test_require_trulens_raises_typed_error_if_unimportable(monkeypatch):
    """If trulens genuinely isn't installed in this environment, calling
    a public function must raise TruLensUnavailableError (not a bare
    ImportError several frames deep), same contract as
    ragas_metrics._require_ragas().
    """
    import app.evaluation.trulens_feedback as mod

    try:
        mod._require_trulens()
    except mod.TruLensUnavailableError:
        pass  # expected in an env without trulens installed
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"_require_trulens() leaked a non-typed exception: {type(exc).__name__}: {exc}")