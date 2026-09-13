"""Tests for evaluation/dashboard.py's `_render_final_spec_table` — the
Phase 7 spec's exact 5-row required final table (Vector / Vector+BM25 /
+CrossEncoder / +Grounding / +CitationValidation), added in the Phase 7
completion pass to actually consume `run_ablation.FINAL_TABLE_ROW_CONFIGS`
(which existed but had no reader until this fix).
"""

from __future__ import annotations

import json

from evaluation.dashboard import _render_final_spec_table, build_dashboard
from evaluation.run_ablation import FINAL_TABLE_ROW_CONFIGS


def test_empty_reports_dir_produces_honest_no_data_message(tmp_path):
    result = _render_final_spec_table(tmp_path, {})
    assert "No data for any row yet" in result
    # Never a table with fabricated/blank cells when nothing exists yet.
    assert "|" not in result


def test_all_five_spec_rows_appear_even_with_only_one_config_available(tmp_path):
    ablation_reports = {"A": {"summary": {"retrieval": {"recall@10": 0.5}}}}
    result = _render_final_spec_table(tmp_path, ablation_reports)
    for row_label in FINAL_TABLE_ROW_CONFIGS:
        assert row_label in result


def test_missing_config_rows_show_em_dash_not_fabricated_values(tmp_path):
    ablation_reports = {"A": {"summary": {"retrieval": {"recall@10": 0.82}}}}
    result = _render_final_spec_table(tmp_path, ablation_reports)
    lines = result.splitlines()
    vector_row = next(l for l in lines if l.startswith("| Vector |"))
    bm25_row = next(l for l in lines if l.startswith("| Vector + BM25 |"))
    assert "0.8200" in vector_row
    # Config B has no report -- every cell in that row must be "—", not 0/blank.
    cells = [c.strip() for c in bm25_row.split("|")[2:-1]]
    assert all(c == "—" for c in cells)


def test_merges_ablation_and_ragas_reports_by_config_letter(tmp_path):
    (tmp_path / "ablation_A.json").write_text(
        json.dumps(
            {
                "summary": {
                    "retrieval": {"recall@10": 0.82, "mrr": 0.61, "ndcg@10": 0.71},
                    "citation_precision": 0.93,
                    "grounded_answer_rate": 0.88,
                    "refusal_accuracy": 0.79,
                    "latency": {"p50_ms": 812.3, "p95_ms": 1502.7},
                }
            }
        )
    )
    (tmp_path / "ragas_A.json").write_text(
        json.dumps(
            {
                "summary": {
                    "mean_context_precision": 0.75,
                    "mean_context_recall": 0.68,
                    "mean_faithfulness": 0.91,
                    "mean_answer_relevancy": 0.87,
                }
            }
        )
    )
    ablation_reports = {"A": json.loads((tmp_path / "ablation_A.json").read_text())}

    result = _render_final_spec_table(tmp_path, ablation_reports)
    vector_row = next(l for l in result.splitlines() if l.startswith("| Vector |"))
    assert "0.8200" in vector_row  # recall@10 from ablation
    assert "0.9100" in vector_row  # faithfulness from ragas
    assert "0.7500" in vector_row  # context precision from ragas


def test_ttft_columns_are_always_explicitly_not_applicable(tmp_path):
    """TTFT has no meaning for the non-streaming ablation runner -- must
    never render as a blank/zero cell that looks like a real
    measurement of 0."""
    ablation_reports = {"A": {"summary": {"retrieval": {"recall@10": 0.5}}}}
    result = _render_final_spec_table(tmp_path, ablation_reports)
    vector_row = next(l for l in result.splitlines() if l.startswith("| Vector |"))
    assert vector_row.count("N/A (non-streaming runner)") == 2  # median + p95 TTFT


def test_footer_names_exactly_which_configs_are_missing_ragas_reports(tmp_path):
    (tmp_path / "ablation_A.json").write_text(json.dumps({"summary": {}}))
    ablation_reports = {"A": {"summary": {}}}
    result = _render_final_spec_table(tmp_path, ablation_reports)
    assert "generate_pipeline_outputs" in result
    assert "run_ragas" in result


def test_build_dashboard_includes_the_required_table_section(tmp_path):
    output = build_dashboard(tmp_path)
    assert "Required final evaluation table" in output