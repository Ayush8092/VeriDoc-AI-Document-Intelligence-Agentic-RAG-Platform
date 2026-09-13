"""Tests for evaluation.dataset_validator — covers both dataset schemas
(QA: phase4_v1.jsonl; multi-document: multi_doc_v1.jsonl), Phase 7
development pass. Previously had zero test coverage, and the validator
itself only knew about the QA schema (confirmed: running the old
validator against the real multi_doc_v1.jsonl rejected every single row
with "missing required fields").
"""

from __future__ import annotations

import json

import pytest

from evaluation.dataset_validator import (
    VALID_SIDEDNESS_VALUES,
    detect_schema,
    validate_file,
    validate_multi_doc_row,
    validate_qa_row,
    validate_row,
)


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "policy.md").write_text("policy content")
    (d / "handbook.docx").write_text("handbook content")
    return d


def _write_jsonl(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Schema auto-detection
# ---------------------------------------------------------------------------


def test_detect_schema_qa():
    row = {"id": "q1", "question": "What?", "query_type": "LOOKUP", "answerable": True}
    assert detect_schema(row) == "qa"


def test_detect_schema_multi_doc():
    row = {"id": "md1", "topic": "t", "documents": ["a.md", "b.md"], "expected_relations": {"t": "both_sided"}}
    assert detect_schema(row) == "multi_doc"


def test_detect_schema_unrecognized_returns_none():
    assert detect_schema({"id": "x", "some_other_field": 1}) is None


def test_detect_schema_extra_fields_still_detected():
    """A row can carry BOTH schemas' fields at once (e.g. superset rows)
    and still be unambiguously detected as QA, since QA is checked
    first and its required set is satisfied."""
    row = {
        "id": "q1", "question": "What?", "query_type": "LOOKUP", "answerable": True,
        "expected_chunks": [],
    }
    assert detect_schema(row) == "qa"


# ---------------------------------------------------------------------------
# QA schema validation — valid + invalid cases
# ---------------------------------------------------------------------------


def _qa_row(**overrides):
    row = {
        "id": "q1",
        "question": "What is the notice period?",
        "query_type": "LOOKUP",
        "answerable": True,
        "expected_chunks": ["corpus::policy.md::notice::0"],
    }
    row.update(overrides)
    return row


def test_valid_qa_row_passes(corpus_dir):
    assert validate_qa_row(_qa_row(), 1, corpus_dir) == []


def test_qa_row_missing_required_field_via_validate_row(corpus_dir):
    row = {"id": "q1", "question": "What?", "query_type": "LOOKUP"}  # no 'answerable'
    errors = validate_row(row, 1, corpus_dir)
    assert len(errors) == 1
    assert "neither the QA schema" in errors[0]


def test_qa_row_answerable_wrong_type_rejected(corpus_dir):
    errors = validate_qa_row(_qa_row(answerable="yes"), 1, corpus_dir)
    assert any("must be a bool" in e for e in errors)


def test_qa_row_refusal_with_expected_chunks_rejected(corpus_dir):
    errors = validate_qa_row(
        _qa_row(answerable=False, should_refuse=True, expected_chunks=["corpus::policy.md::x::0"]), 1, corpus_dir
    )
    assert any("should not also list expected_chunks" in e for e in errors)


def test_qa_row_answerable_true_with_no_chunks_rejected(corpus_dir):
    errors = validate_qa_row(_qa_row(expected_chunks=[]), 1, corpus_dir)
    assert any("expected_chunks/required_chunks is empty" in e for e in errors)


def test_qa_row_unknown_ground_truth_status_rejected(corpus_dir):
    errors = validate_qa_row(_qa_row(ground_truth_status="made_up_status"), 1, corpus_dir)
    assert any("unknown ground_truth_status" in e for e in errors)


def test_qa_row_pending_ingestion_with_expected_objects_rejected(corpus_dir):
    errors = validate_qa_row(
        _qa_row(ground_truth_status="pending_ingestion", expected_objects=[{"type": "table"}]), 1, corpus_dir
    )
    assert any("pending_ingestion" in e for e in errors)


def test_qa_row_expected_pages_wrong_type_rejected(corpus_dir):
    errors = validate_qa_row(_qa_row(expected_pages="not a list"), 1, corpus_dir)
    assert any("expected_pages" in e and "must be a list" in e for e in errors)


def test_qa_row_references_nonexistent_corpus_file_rejected(corpus_dir):
    errors = validate_qa_row(_qa_row(expected_chunks=["corpus::does_not_exist.md::x::0"]), 1, corpus_dir)
    assert any("does_not_exist.md" in e and "does not exist" in e for e in errors)


def test_qa_row_references_real_corpus_file_passes(corpus_dir):
    errors = validate_qa_row(_qa_row(expected_chunks=["corpus::policy.md::notice::0"]), 1, corpus_dir)
    assert errors == []


# ---------------------------------------------------------------------------
# Multi-document schema validation — valid + invalid cases
# ---------------------------------------------------------------------------


def _md_row(**overrides):
    row = {
        "id": "md1",
        "topic": "approval requirements",
        "documents": ["policy.md", "handbook.docx"],
        "expected_relations": {"approval": "both_sided"},
    }
    row.update(overrides)
    return row


def test_valid_multi_doc_row_passes(corpus_dir):
    assert validate_multi_doc_row(_md_row(), 1, corpus_dir) == []


def test_multi_doc_row_via_validate_row_dispatch(corpus_dir):
    assert validate_row(_md_row(), 1, corpus_dir) == []


def test_multi_doc_row_empty_topic_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(topic="   "), 1, corpus_dir)
    assert any("'topic' must be a non-empty string" in e for e in errors)


def test_multi_doc_row_single_document_rejected(corpus_dir):
    """A comparison needs at least 2 documents — one document has
    nothing to relate to."""
    errors = validate_multi_doc_row(_md_row(documents=["policy.md"]), 1, corpus_dir)
    assert any("at least 2 documents" in e for e in errors)


def test_multi_doc_row_documents_wrong_type_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(documents="policy.md"), 1, corpus_dir)
    assert any("'documents' must be a list" in e for e in errors)


def test_multi_doc_row_nonexistent_document_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(documents=["policy.md", "ghost.md"]), 1, corpus_dir)
    assert any("ghost.md" in e and "does not exist" in e for e in errors)


def test_multi_doc_row_invalid_relation_value_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(expected_relations={"approval": "sort_of_both"}), 1, corpus_dir)
    assert any("sort_of_both" in e for e in errors)


def test_multi_doc_row_empty_relations_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(expected_relations={}), 1, corpus_dir)
    assert any("'expected_relations' must be a non-empty object" in e for e in errors)


def test_multi_doc_row_relations_wrong_type_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(expected_relations=["both_sided"]), 1, corpus_dir)
    assert any("'expected_relations' must be a non-empty object" in e for e in errors)


def test_multi_doc_row_notes_wrong_type_rejected(corpus_dir):
    errors = validate_multi_doc_row(_md_row(notes=123), 1, corpus_dir)
    assert any("'notes' must be a string" in e for e in errors)


def test_multi_doc_row_notes_string_is_fine(corpus_dir):
    errors = validate_multi_doc_row(_md_row(notes="some context"), 1, corpus_dir)
    assert errors == []


def test_both_sidedness_values_accepted(corpus_dir):
    for value in VALID_SIDEDNESS_VALUES:
        errors = validate_multi_doc_row(_md_row(expected_relations={"x": value}), 1, corpus_dir)
        assert errors == [], f"{value!r} should be a valid sidedness value"


# ---------------------------------------------------------------------------
# validate_file — end to end, including duplicate ids and mixed-row files
# ---------------------------------------------------------------------------


def test_validate_file_qa_dataset_all_valid(tmp_path, corpus_dir):
    path = _write_jsonl(tmp_path, "qa.jsonl", [_qa_row(id="q1"), _qa_row(id="q2")])
    assert validate_file(path, corpus_dir) == []


def test_validate_file_multi_doc_dataset_all_valid(tmp_path, corpus_dir):
    path = _write_jsonl(tmp_path, "md.jsonl", [_md_row(id="md1"), _md_row(id="md2")])
    assert validate_file(path, corpus_dir) == []


def test_validate_file_detects_duplicate_ids_across_schemas(tmp_path, corpus_dir):
    path = _write_jsonl(tmp_path, "mixed.jsonl", [_qa_row(id="dup"), _md_row(id="dup")])
    errors = validate_file(path, corpus_dir)
    assert any("duplicate id 'dup'" in e for e in errors)


def test_validate_file_rejects_malformed_json_line(tmp_path, corpus_dir):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "q1", "question": "ok"\n', encoding="utf-8")
    errors = validate_file(path, corpus_dir)
    assert any("invalid JSON" in e for e in errors)


def test_validate_file_rejects_non_object_row(tmp_path, corpus_dir):
    path = tmp_path / "list_row.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")
    errors = validate_file(path, corpus_dir)
    assert any("row must be a JSON object" in e for e in errors)


def test_validate_file_skips_blank_lines(tmp_path, corpus_dir):
    path = _write_jsonl(tmp_path, "qa.jsonl", [_qa_row(id="q1")])
    path.write_text(path.read_text() + "\n\n\n", encoding="utf-8")
    assert validate_file(path, corpus_dir) == []


# ---------------------------------------------------------------------------
# Real datasets — the actual regression this file exists to guard
# ---------------------------------------------------------------------------


def test_real_phase4_dataset_is_valid():
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "phase4_v1.jsonl"
    errors = validate_file(dataset_path)
    assert errors == [], f"phase4_v1.jsonl has real validation errors: {errors}"


def test_real_multi_doc_dataset_is_valid():
    """The exact regression this whole file exists to catch: before this
    pass, this call raised/reported errors on every row."""
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "multi_doc_v1.jsonl"
    errors = validate_file(dataset_path)
    assert errors == [], f"multi_doc_v1.jsonl has real validation errors: {errors}"


def test_real_multi_doc_dataset_is_detected_as_multi_doc_schema_not_qa():
    import json
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "multi_doc_v1.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert detect_schema(row) == "multi_doc"