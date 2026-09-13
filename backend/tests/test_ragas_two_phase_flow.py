"""Tests for the two-phase RAGAS flow (Phase 7 completion pass):
`evaluation/generate_pipeline_outputs.py` (runs in the main venv, has
langgraph) writes a JSON file; `evaluation/run_ragas.py` (runs in the
isolated `.venv-ragas`, does NOT have langgraph) reads it and scores it.

The split exists because an earlier single-script version of
`run_ragas.py` imported `app.rag.graph.DocumentQAService`, which
imports `langgraph` — unavailable (and unavailable BY DESIGN) in
`.venv-ragas`. These tests run in the main venv (this project's normal
test environment) and prove the DATA FLOW between the two scripts is
correct; they cannot prove `run_ragas.py` imports cleanly in
`.venv-ragas` itself (that was verified manually against a real
isolated-venv install — see `requirements-ragas.txt`'s header) since
this test suite's own venv has langgraph installed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from evaluation import generate_pipeline_outputs as gen
from evaluation import run_ragas


def test_run_ragas_module_does_not_import_app_rag_graph():
    """The specific regression this whole fix exists to prevent: if
    `run_ragas.py` ever again imports `app.rag.graph` (directly or
    transitively at module scope), it would break in `.venv-ragas`
    exactly like the pre-fix version did. Checked against the module's
    actual IMPORT STATEMENTS specifically (not its docstring text, which
    legitimately mentions `app.rag.graph` while explaining this very
    fix) — this suite's own venv has langgraph installed, so an
    accidental reintroduced import wouldn't fail here; this assertion
    is what actually catches the regression in THIS venv.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(run_ragas))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    assert not any(name.startswith("app.rag.graph") for name in imported_names)
    assert not any(name.startswith("langgraph") for name in imported_names)
    assert "DocumentQAService" not in dir(run_ragas)


def test_generate_pipeline_outputs_applies_config_overrides(tmp_path):
    fake_rows = [{"id": "q1", "question": "What is X?", "query_type": "LOOKUP", "answerable": True}]

    with patch.object(gen, "load_dataset", return_value=fake_rows), patch.object(gen, "DocumentQAService") as MockService:
        instance = MockService.return_value
        instance.ask.return_value = {
            "found": True,
            "answer": "X is Y.",
            "citations": [{"chunk_id": "c1", "snippet": "X is defined as Y."}],
        }
        report = gen.generate(tmp_path / "unused.jsonl", "D", None)

    # The settings passed to DocumentQAService must carry config D's
    # actual overrides, not defaults.
    constructed_settings = MockService.call_args.kwargs["settings"]
    for key, value in gen.CONFIGS["D"].overrides.items():
        assert getattr(constructed_settings, key) == value

    assert report["config"] == "D"
    assert report["config_overrides"] == gen.CONFIGS["D"].overrides
    assert report["rows"][0]["retrieved_contexts"] == ["X is defined as Y."]


def test_generate_pipeline_outputs_rejects_unknown_config():
    with pytest.raises(SystemExit, match="Unknown config"):
        gen.generate(gen.DEFAULT_DATASET, "Z", None)


def test_generate_pipeline_outputs_no_config_means_default_settings(tmp_path):
    fake_rows = [{"id": "q1", "question": "What is X?"}]
    with patch.object(gen, "load_dataset", return_value=fake_rows), patch.object(gen, "DocumentQAService") as MockService:
        instance = MockService.return_value
        instance.ask.return_value = {"found": False, "answer": "", "citations": []}
        report = gen.generate(tmp_path / "unused.jsonl", None, None)

    assert report["config"] is None
    assert report["config_overrides"] == {}


def test_run_ragas_loads_pipeline_outputs_and_scores_them(tmp_path):
    pipeline_outputs = {
        "config": "D",
        "dataset": "evaluation/datasets/phase4_v1.jsonl",
        "n_questions": 2,
        "rows": [
            {
                "id": "q1",
                "question": "What is X?",
                "query_type": "LOOKUP",
                "answerable": True,
                "found": True,
                "answer": "X is Y.",
                "retrieved_contexts": ["X is defined as Y."],
            },
            {
                "id": "q2",
                "question": "Unanswerable?",
                "query_type": "LOOKUP",
                "answerable": False,
                "found": False,
                "answer": "",
                "retrieved_contexts": [],
            },
        ],
    }
    path = tmp_path / "pipeline_outputs_D.json"
    path.write_text(json.dumps(pipeline_outputs), encoding="utf-8")

    fake_scores = {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}
    with patch.object(run_ragas, "score_single", return_value=fake_scores) as mock_score:
        report = run_ragas.run(path)

    # Only the ANSWERED question is scored — q2 was correctly refused,
    # nothing to score RAGAS metrics against.
    mock_score.assert_called_once()
    assert mock_score.call_args.kwargs["question"] == "What is X?"
    assert mock_score.call_args.kwargs["retrieved_contexts"] == ["X is defined as Y."]

    assert report["summary"]["config"] == "D"
    assert report["summary"]["n_questions"] == 2
    assert report["summary"]["n_answered"] == 1
    assert report["summary"]["n_scored"] == 1
    assert report["summary"]["mean_faithfulness"] == 0.9
    assert report["results"][1]["scores"] is None  # q2 never scored


def test_run_ragas_respects_limit(tmp_path):
    pipeline_outputs = {
        "config": None,
        "rows": [
            {"id": f"q{i}", "question": f"Q{i}", "found": False, "answer": "", "retrieved_contexts": []}
            for i in range(5)
        ],
    }
    path = tmp_path / "pipeline_outputs.json"
    path.write_text(json.dumps(pipeline_outputs), encoding="utf-8")

    report = run_ragas.run(path, limit=2)
    assert report["summary"]["n_questions"] == 2


def test_run_ragas_stops_immediately_on_ragas_unavailable(tmp_path):
    """A RagasUnavailableError on the first question aborts the whole
    run rather than wasting further scoring calls that would fail
    identically."""
    pipeline_outputs = {
        "config": "A",
        "rows": [
            {"id": "q1", "question": "Q1", "found": True, "answer": "A1", "retrieved_contexts": ["ctx"]},
            {"id": "q2", "question": "Q2", "found": True, "answer": "A2", "retrieved_contexts": ["ctx"]},
        ],
    }
    path = tmp_path / "pipeline_outputs_A.json"
    path.write_text(json.dumps(pipeline_outputs), encoding="utf-8")

    from app.evaluation.ragas_metrics import RagasUnavailableError

    with patch.object(run_ragas, "score_single", side_effect=RagasUnavailableError("ragas not installed")):
        with pytest.raises(SystemExit, match="RAGAS unavailable"):
            run_ragas.run(path)