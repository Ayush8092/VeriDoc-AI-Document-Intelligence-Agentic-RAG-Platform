"""Structural validation for evaluation dataset files.

Two distinct schemas exist in `evaluation/datasets/` and are BOTH
validated by this module (Phase 7 development pass — this used to only
know about the QA schema, and would reject every row of
`multi_doc_v1.jsonl` with "missing required fields" — confirmed by
actually running the old validator against it before this rewrite):

- **QA schema** (`evaluation/datasets/phase4_v1.jsonl`, and the
  original `v1.jsonl` before it): one row per benchmark question —
  `id`/`question`/`query_type`/`answerable`, plus optional
  `expected_chunks`/`expected_pages`/`expected_objects`/
  `expected_citations`/`should_refuse`/`ground_truth_status`.
- **Multi-document schema** (`evaluation/datasets/multi_doc_v1.jsonl`):
  one row per cross-document TOPIC (not a question) —
  `id`/`topic`/`documents`/`expected_relations`, feeding
  `app.evaluation.reasoning_metrics.multi_document_reasoning_accuracy`.

Schema is auto-detected PER ROW (not once per file) from which
required-field set the row actually has, and a row matching NEITHER
schema's required fields is a validation error, not a silent skip — see
`detect_schema`.

Deliberately does not touch a live index/DB for chunk-level content
(only structural + real-file-on-disk checks — see `_referenced_corpus_files`
and `validate_corpus_references`), so this can run in CI or a throwaway
venv with just the standard library and read access to `data/corpus/`.

Usage:
    python -m evaluation.dataset_validator evaluation/datasets/phase4_v1.jsonl
    python -m evaluation.dataset_validator evaluation/datasets/*.jsonl
    python -m evaluation.dataset_validator evaluation/datasets/multi_doc_v1.jsonl --corpus-dir data/corpus
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = BACKEND_ROOT / "data" / "corpus"

# ---------------------------------------------------------------------------
# QA schema (phase4_v1.jsonl / v1.jsonl)
# ---------------------------------------------------------------------------

QA_REQUIRED_FIELDS = {"id", "question", "query_type", "answerable"}
KNOWN_GROUND_TRUTH_STATUS = {"verified_v1", "verified_phase4", "pending_ingestion"}

# `corpus::<filename>::<section>::<index>` — see app/chunking.py's
# chunk_id scheme. Only the filename segment matters here.
_CHUNK_ID_RE = re.compile(r"^corpus::([^:]+)::")

# ---------------------------------------------------------------------------
# Multi-document schema (multi_doc_v1.jsonl)
# ---------------------------------------------------------------------------

MULTI_DOC_REQUIRED_FIELDS = {"id", "topic", "documents", "expected_relations"}
# Matches app.evaluation.reasoning_metrics._RELATION_TO_SIDEDNESS's
# target domain exactly — kept as a literal set here (not imported) so
# this module stays dependency-free/importable without the rest of the
# app package; the two are duplicated by value, not by reference, since
# a mismatch between them would itself be worth a test catching (see
# tests/test_dataset_validator.py's cross-check).
VALID_SIDEDNESS_VALUES = {"one_sided", "both_sided"}


def detect_schema(row: dict) -> str | None:
    """Returns `"qa"`, `"multi_doc"`, or `None` (matches neither schema's
    required-field set — a real error, not something to guess past).
    A row could in principle satisfy both sets only if it were missing
    fields from both (impossible, since the two required-field sets are
    disjoint) — so this is an unambiguous, order-independent check.
    """
    keys = row.keys()
    if QA_REQUIRED_FIELDS <= keys:
        return "qa"
    if MULTI_DOC_REQUIRED_FIELDS <= keys:
        return "multi_doc"
    return None


def _referenced_corpus_files(row: dict) -> set[str]:
    """Every corpus filename a QA row's chunk-id-shaped fields reference
    (`expected_chunks`/`required_chunks`), extracted via `_CHUNK_ID_RE`.
    Non-`corpus::...`-shaped entries (e.g. a future `uploads::...`
    scheme) are silently not extracted here — they're not "the shipped
    corpus" this validator's `--corpus-dir` check is about, and
    `validate_qa_row`'s pre-existing type check already covers "is this
    a list of strings" regardless.
    """
    files = set()
    for field in ("expected_chunks", "required_chunks"):
        for chunk_id in row.get(field) or []:
            if not isinstance(chunk_id, str):
                continue
            m = _CHUNK_ID_RE.match(chunk_id)
            if m:
                files.add(m.group(1))
    return files


def validate_qa_row(row: dict, line_number: int, corpus_dir: Path) -> list[str]:
    errors = []
    row_id = row.get("id", "?")

    if not isinstance(row["id"], str) or not row["id"]:
        errors.append(f"line {line_number}: 'id' must be a non-empty string")
    if not isinstance(row["answerable"], bool):
        errors.append(f"line {line_number} (id={row_id}): 'answerable' must be a bool")

    should_refuse = row.get("should_refuse")
    if should_refuse is not None and not isinstance(should_refuse, bool):
        errors.append(f"line {line_number} (id={row_id}): 'should_refuse' must be a bool when present")
    effective_refuse = should_refuse if should_refuse is not None else (row["answerable"] is False)

    expected_chunks = row.get("expected_chunks") or row.get("required_chunks") or []
    if not isinstance(expected_chunks, list) or not all(isinstance(c, str) for c in expected_chunks):
        errors.append(f"line {line_number} (id={row_id}): 'expected_chunks'/'required_chunks' must be a list of strings")
    elif effective_refuse and expected_chunks:
        errors.append(
            f"line {line_number} (id={row_id}): a refusal row (should_refuse/answerable=False) "
            f"should not also list expected_chunks — got {expected_chunks}"
        )
    elif not effective_refuse and row["answerable"] and not expected_chunks and not row.get("question", "").strip() == "":
        errors.append(f"line {line_number} (id={row_id}): answerable=True but expected_chunks/required_chunks is empty")

    for field in ("expected_pages", "expected_objects", "expected_citations"):
        value = row.get(field)
        if value is not None and not isinstance(value, list):
            errors.append(f"line {line_number} (id={row_id}): '{field}' must be a list when present")

    status = row.get("ground_truth_status")
    if status is not None and status not in KNOWN_GROUND_TRUTH_STATUS:
        errors.append(
            f"line {line_number} (id={row_id}): unknown ground_truth_status {status!r} "
            f"(expected one of {sorted(KNOWN_GROUND_TRUTH_STATUS)})"
        )
    if status == "pending_ingestion" and (row.get("expected_objects") or row.get("expected_citations")):
        errors.append(
            f"line {line_number} (id={row_id}): ground_truth_status='pending_ingestion' rows must not "
            f"carry expected_objects/expected_citations yet — see SCHEMA.md"
        )

    for filename in sorted(_referenced_corpus_files(row)):
        if not (corpus_dir / filename).is_file():
            errors.append(
                f"line {line_number} (id={row_id}): expected_chunks references corpus file "
                f"{filename!r}, which does not exist under {corpus_dir}"
            )

    return errors


def validate_multi_doc_row(row: dict, line_number: int, corpus_dir: Path) -> list[str]:
    errors = []
    row_id = row.get("id", "?")

    if not isinstance(row["id"], str) or not row["id"]:
        errors.append(f"line {line_number}: 'id' must be a non-empty string")
    if not isinstance(row.get("topic"), str) or not row["topic"].strip():
        errors.append(f"line {line_number} (id={row_id}): 'topic' must be a non-empty string")

    documents = row.get("documents")
    if not isinstance(documents, list) or not all(isinstance(d, str) and d for d in documents):
        errors.append(f"line {line_number} (id={row_id}): 'documents' must be a list of non-empty strings")
        documents = []
    elif len(documents) < 2:
        errors.append(
            f"line {line_number} (id={row_id}): 'documents' must list at least 2 documents to compare "
            f"(a single document has nothing to relate to) — got {documents}"
        )

    for filename in documents:
        if not (corpus_dir / filename).is_file():
            errors.append(
                f"line {line_number} (id={row_id}): 'documents' references {filename!r}, "
                f"which does not exist under {corpus_dir}"
            )

    relations = row.get("expected_relations")
    if not isinstance(relations, dict) or not relations:
        errors.append(f"line {line_number} (id={row_id}): 'expected_relations' must be a non-empty object")
    else:
        for topic_key, value in relations.items():
            if not isinstance(topic_key, str) or not topic_key:
                errors.append(f"line {line_number} (id={row_id}): 'expected_relations' has a non-string/empty key")
            if value not in VALID_SIDEDNESS_VALUES:
                errors.append(
                    f"line {line_number} (id={row_id}): expected_relations[{topic_key!r}]={value!r} is not one of "
                    f"{sorted(VALID_SIDEDNESS_VALUES)}"
                )

    notes = row.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append(f"line {line_number} (id={row_id}): 'notes' must be a string when present")

    return errors


def validate_row(row: dict, line_number: int, corpus_dir: Path) -> list[str]:
    """Dispatch to the correct schema validator, per-row. Do not weaken
    validation to make a dataset pass: a row matching NEITHER schema is
    always an error, never silently accepted or skipped.
    """
    schema = detect_schema(row)
    if schema == "qa":
        return validate_qa_row(row, line_number, corpus_dir)
    if schema == "multi_doc":
        return validate_multi_doc_row(row, line_number, corpus_dir)

    missing_qa = QA_REQUIRED_FIELDS - row.keys()
    missing_multi_doc = MULTI_DOC_REQUIRED_FIELDS - row.keys()
    return [
        f"line {line_number} (id={row.get('id', '?')}): row matches neither the QA schema "
        f"(missing {sorted(missing_qa)}) nor the multi-document schema (missing {sorted(missing_multi_doc)})"
    ]


def validate_file(path: Path, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc})")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: row must be a JSON object")
            continue
        errors.extend(validate_row(row, line_number, corpus_dir))
        row_id = row.get("id")
        if row_id in seen_ids:
            errors.append(f"line {line_number}: duplicate id {row_id!r}")
        elif row_id:
            seen_ids.add(row_id)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Directory referenced corpus files must exist in (default: {DEFAULT_CORPUS_DIR})",
    )
    args = parser.parse_args()

    total_errors = 0
    for dataset in args.datasets:
        errors = validate_file(dataset, args.corpus_dir)
        if errors:
            total_errors += len(errors)
            print(f"{len(errors)} problem(s) found in {dataset}:")
            for e in errors:
                print(f"  - {e}")
        else:
            n_rows = len([l for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()])
            print(f"{dataset}: {n_rows} rows, all valid.")

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()