"""Tests for `app.rag.table_reasoning` — both the pure deterministic
arithmetic (Phase 6 spec item 9) and its actual wiring into the graph's
`_generate_answer` node (`app.rag.graph`).

Phase 6 audit, Part 5/5.1: `table_reasoning.py`'s existence alone proves
nothing about whether a real user query can ever reach it — Part 5's
code-path trace confirmed `app.rag.graph.DocumentQAService._generate_answer`
calls `table_reasoning.try_compute_from_chunks` unconditionally (on
every answered question, gated by evidence content + question wording,
not by `query_type`), so the routing tests below exercise that exact
node directly rather than merely re-testing the module in isolation.

Real fixture data, not fabricated: the "Revenue by Region" table used
throughout these tests is the ACTUAL table `data/corpus/04_quarterly_report_excerpt.pdf`
extracts (verified directly via pdfplumber before writing this file):

    Region          Q1 Revenue   Q2 Revenue   YoY Growth
    North America   $1.2M        $1.4M        12%
    Europe          $0.8M        $0.9M        9%
    Asia Pacific    $0.5M        $0.7M        22%

Follows `tests/test_comparison.py`'s established pattern for the
retrieval-level tests (a real `LocalVectorIndex`, a fixed-vector fake
embedder — no live Pinecone/embedding model needed) and
`tests/test_graph_structure.py`'s pattern for the graph-node-level tests
(`object.__new__(DocumentQAService)` to skip `__init__`, since node
methods only need `self.settings`/`self.chat`, both fakeable).
"""

from __future__ import annotations

from app.core.config import Settings
from app.rag import table_reasoning
from app.rag.graph import DocumentQAService
from app.rag.table_reasoning import (
    TableOperation,
    compute_table_operation,
    detect_operation,
    try_compute_from_chunks,
)
from app.vectorstore_local import LocalVectorIndex

REVENUE_TABLE_ROWS = [
    ["Region", "Q1 Revenue", "Q2 Revenue", "YoY Growth"],
    ["North America", "$1.2M", "$1.4M", "12%"],
    ["Europe", "$0.8M", "$0.9M", "9%"],
    ["Asia Pacific", "$0.5M", "$0.7M", "22%"],
]


def _table_chunk(chunk_id="corpus::04_quarterly_report_excerpt.pdf::content::1", rows=None, **overrides):
    chunk = {
        "chunk_id": chunk_id,
        "source_file": "04_quarterly_report_excerpt.pdf",
        "section": "Revenue by Region",
        "text": "Revenue by Region table",
        "score": 0.9,
        "block_type": "table",
        "table_rows": rows if rows is not None else REVENUE_TABLE_ROWS,
        "page_start": 1,
        "page_end": 1,
        "source": "native",
        "confidence": 1.0,
        "object_id": "table-0001",
        "bbox": {"x0": 10, "y0": 20, "x1": 300, "y1": 120},
        "coordinate_space": "pdf_points",
    }
    chunk.update(overrides)
    return chunk


# ---------------------------------------------------------------------
# Part A — pure deterministic arithmetic (no retrieval, no LLM)
# ---------------------------------------------------------------------


def test_detect_operation_recognizes_max_min_sum_average():
    assert detect_operation("What is the highest Q2 revenue?") == TableOperation.MAX
    assert detect_operation("Which region had the lowest growth?") == TableOperation.MIN
    assert detect_operation("What is the total Q1 revenue?") == TableOperation.SUM
    assert detect_operation("What is the average Q1 revenue?") == TableOperation.AVERAGE


def test_detect_operation_returns_none_for_a_plain_lookup():
    assert detect_operation("What is North America's Q1 revenue?") is None


def test_compute_max_finds_correct_row_and_value():
    result = compute_table_operation(REVENUE_TABLE_ROWS, "Which region had the highest Q2 revenue?")
    assert result is not None
    assert result.operation == TableOperation.MAX
    assert result.row_label == "North America"
    assert result.value == 1_400_000
    assert result.column_label == "Q2 Revenue"


def test_compute_min_finds_correct_row_and_value():
    result = compute_table_operation(REVENUE_TABLE_ROWS, "Which region had the lowest Q1 revenue?")
    assert result is not None
    assert result.row_label == "Asia Pacific"
    assert result.value == 500_000


def test_compute_sum_across_a_column():
    result = compute_table_operation(REVENUE_TABLE_ROWS, "What is the total Q1 revenue across all regions?")
    assert result is not None
    assert result.operation == TableOperation.SUM
    assert result.value == 1_200_000 + 800_000 + 500_000


def test_compute_average_across_a_column():
    result = compute_table_operation(REVENUE_TABLE_ROWS, "What is the average YoY growth?")
    assert result is not None
    assert result.operation == TableOperation.AVERAGE
    assert abs(result.value - ((12 + 9 + 22) / 3)) < 1e-6


def test_compute_percentage_change_for_a_matched_row():
    result = compute_table_operation(
        REVENUE_TABLE_ROWS, "What was the percentage change in revenue for Europe from Q1 to Q2?"
    )
    assert result is not None
    assert result.operation == TableOperation.PERCENTAGE_CHANGE
    assert result.row_label == "Europe"
    expected_pct = (900_000 - 800_000) / 800_000 * 100
    assert abs(result.value - expected_pct) < 1e-6


def test_compute_returns_none_when_column_cannot_be_matched():
    """Fail-closed: a question with no confident column/row match must
    return `None`, never a guessed answer — this is the module's core
    safety property (see its docstring)."""
    result = compute_table_operation(REVENUE_TABLE_ROWS, "What is the highest employee satisfaction score?")
    assert result is None


def test_compute_returns_none_for_a_non_table_grid():
    assert compute_table_operation([], "What is the highest value?") is None
    assert compute_table_operation([["only one row"]], "What is the highest value?") is None


def test_compute_percentage_change_returns_none_on_division_by_zero():
    rows = [["Region", "Q1", "Q2"], ["Zeroville", "0", "100"]]
    result = compute_table_operation(rows, "What is the percentage change from Q1 to Q2 for Zeroville?")
    assert result is None


# ---------------------------------------------------------------------
# Part B — try_compute_from_chunks: citation/provenance preservation,
# missing/invalid evidence
# ---------------------------------------------------------------------


def test_try_compute_from_chunks_preserves_full_citation_provenance():
    """The exact property Part 5 question 8 asks about: page, bbox,
    object_id, coordinate_space must survive into the deterministic
    answer's citations, not just chunk_id/score."""
    chunk = _table_chunk()
    result = try_compute_from_chunks([chunk], "Which region had the highest Q2 revenue?")
    assert result is not None
    assert result["answered"] is True
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["chunk_id"] == chunk["chunk_id"]
    assert citation["page_start"] == 1
    assert citation["object_id"] == "table-0001"
    assert citation["bbox"] == {"x0": 10, "y0": 20, "x1": 300, "y1": 120}
    assert citation["coordinate_space"] == "pdf_points"
    assert citation["table_rows"] == REVENUE_TABLE_ROWS


def test_try_compute_from_chunks_ignores_non_table_chunks():
    text_chunk = {
        "chunk_id": "corpus::01_product_overview.md::intro::0",
        "source_file": "01_product_overview.md",
        "section": "Intro",
        "text": "Veridoc is a document intelligence platform.",
        "score": 0.8,
        "block_type": "text",
    }
    result = try_compute_from_chunks([text_chunk], "What is the highest Q2 revenue?")
    assert result is None


def test_try_compute_from_chunks_ignores_table_chunk_with_missing_rows():
    chunk = _table_chunk(rows=None)
    chunk["table_rows"] = None
    result = try_compute_from_chunks([chunk], "What is the highest Q2 revenue?")
    assert result is None


def test_try_compute_from_chunks_returns_none_when_no_chunks_given():
    assert try_compute_from_chunks([], "What is the highest Q2 revenue?") is None


def test_try_compute_from_chunks_returns_none_for_a_plain_lookup_question():
    """A question that isn't asking for a calculation at all must not
    be forced through the table-reasoning path — `detect_operation`
    returning `None` should short-circuit before any grid matching."""
    chunk = _table_chunk()
    result = try_compute_from_chunks([chunk], "What is North America's Q1 revenue?")
    assert result is None


def test_try_compute_from_chunks_picks_first_computable_table_among_several():
    unrelated = _table_chunk(
        chunk_id="corpus::05_employee_handbook_excerpt.docx::approval-thresholds::0",
        rows=[["Expense Type", "Approval Needed"], ["Travel", "Manager"]],
    )
    revenue = _table_chunk()
    result = try_compute_from_chunks([unrelated, revenue], "Which region had the highest Q2 revenue?")
    assert result is not None
    assert result["citations"][0]["chunk_id"] == revenue["chunk_id"]


# ---------------------------------------------------------------------
# Part C — tenant isolation (answer_table_question's own retrieval call)
# ---------------------------------------------------------------------


class _FakeSettings:
    embedding_dimension = 4
    pinecone_namespace = "test-ns"
    top_k = 10
    score_threshold = 0.0
    hybrid_retrieval_enabled = False


class _FixedEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def _upsert_table_chunk(index, chunk_id, owner_id, rows):
    """Builds metadata matching the REAL on-disk contract
    `app.vectorstore.upsert_chunks`/`_match_to_result` use — table rows
    are stored JSON-serialized under the `table_json` key (see
    `app/vectorstore.py`), not as a raw list under `table_rows`.
    Writing `table_rows` directly (an earlier version of this fixture
    did exactly that) bypasses that contract and silently produces a
    chunk `_match_to_result` can never recover `table_rows` from —
    `answer_table_question` would then find zero table evidence
    regardless of the owner filter, making both tenant-isolation tests
    below pass VACUOUSLY (for "no evidence exists" instead of "no
    evidence exists for THIS tenant"). Confirmed via a direct
    `hybrid_retrieve_single` call during Phase 6 audit that the raw-key
    version returned `table_rows: None` on an otherwise-correct,
    correctly-owner-filtered match.
    """
    import json

    index.upsert(
        [
            {
                "id": chunk_id,
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": "revenue.pdf",
                    "owner_id": owner_id,
                    "text": "Revenue table",
                    "block_type": "table",
                    "table_json": json.dumps(rows),
                    "page_start": 1,
                    "page_end": 1,
                },
            }
        ],
        namespace="test-ns",
    )


def test_answer_table_question_respects_tenant_scoping(tmp_path):
    """`answer_table_question` (the standalone entry point, distinct from
    the graph-embedded `try_compute_from_chunks`) does its own retrieval
    via `hybrid_retrieve_single` — this proves that retrieval call is
    actually scoped by `allowed_owner_ids`, the same mechanism every
    other retrieval path in this project uses for tenant isolation.
    """
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    # Two DIFFERENT tables under two different owners — if isolation
    # were broken, querying as owner "alice" could see owner "bob"'s
    # table and compute an answer from data alice must never see.
    _upsert_table_chunk(
        index, "alice-table", owner_id="alice", rows=[["Region", "Revenue"], ["Alice-Only Region", "999"]]
    )
    _upsert_table_chunk(
        index, "bob-table", owner_id="bob", rows=[["Region", "Revenue"], ["Bob-Only Region", "1"]]
    )

    result = table_reasoning.answer_table_question(
        index,
        _FixedEmbeddings(),
        _FakeSettings(),
        "What is the highest revenue?",
        allowed_owner_ids=frozenset({"alice"}),
    )
    assert result["answered"] is True
    assert result["citations"][0]["chunk_id"] == "alice-table"
    assert "Bob-Only Region" not in str(result)


def test_answer_table_question_raises_when_no_table_evidence_for_this_tenant(tmp_path):
    """If the only table evidence in the index belongs to a different
    tenant, this tenant must see "no table evidence" — never silently
    fall through to someone else's data."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert_table_chunk(
        index, "bob-table", owner_id="bob", rows=[["Region", "Revenue"], ["Bob-Only Region", "1"]]
    )

    try:
        table_reasoning.answer_table_question(
            index,
            _FixedEmbeddings(),
            _FakeSettings(),
            "What is the highest revenue?",
            allowed_owner_ids=frozenset({"alice"}),
        )
        raised = False
    except table_reasoning.TableReasoningError:
        raised = True
    assert raised, "expected TableReasoningError when no table evidence exists for this tenant"


# ---------------------------------------------------------------------
# Part D — graph routing: does a real table query actually reach
# try_compute_from_chunks through DocumentQAService._generate_answer?
# ---------------------------------------------------------------------


def _make_service(**settings_overrides) -> DocumentQAService:
    service = object.__new__(DocumentQAService)
    service.settings = Settings(_env_file=None, **settings_overrides)
    return service


def _graph_state(question, chunks, relevant_ids):
    return {
        "question": question,
        "chunks": chunks,
        "grade": {"sufficient": True, "relevant_chunk_ids": relevant_ids, "reason": "test"},
    }


def test_generate_answer_node_routes_a_table_query_to_deterministic_path():
    """The central Part 5 proof: calling the REAL `_generate_answer`
    graph node (not a re-implementation of it) with table evidence and a
    computable question must produce a deterministic, table-reasoning
    -sourced answer — and must do so WITHOUT ever touching `self.chat`,
    proving no LLM call was needed to compute the number.
    """
    service = _make_service()

    class _ChatThatMustNotBeCalled:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise AssertionError(
                        "self.chat was called — the deterministic table path did not short-circuit "
                        "before reaching the LLM generation fallback"
                    )

    service.chat = _ChatThatMustNotBeCalled()

    chunk = _table_chunk()
    state = _graph_state("Which region had the highest Q2 revenue?", [chunk], [chunk["chunk_id"]])

    result = service._generate_answer(state)

    assert result["found"] is True
    assert "North America" in result["answer"]
    assert isinstance(result["citations"], list)
    assert result["citations"][0]["chunk_id"] == chunk["chunk_id"]
    assert "deterministic" in result["trace"][0]


def test_generate_answer_node_falls_back_to_llm_when_table_evidence_is_not_computable():
    """Graceful failure / deterministic fallback (Part 5 questions 10 &
    11): when table evidence exists but doesn't answer a computable
    question, `_generate_answer` must fall through to the normal LLM
    generation path rather than erroring or hanging.
    """
    service = _make_service()

    class _FakeMessage:
        content = '{"found": true, "answer": "From the LLM path.", "cited_chunk_ids": ["irrelevant-chunk"]}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    called = {"count": 0}

    class _FakeChat:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    called["count"] += 1
                    return _FakeCompletion()

    service.chat = _FakeChat()

    # A table chunk is present, but the question isn't a recognized
    # calculation at all (detect_operation returns None) — table
    # reasoning must decline (return None) and the LLM path must run.
    chunk = _table_chunk()
    state = _graph_state("What is North America's Q1 revenue?", [chunk], [chunk["chunk_id"]])

    result = service._generate_answer(state)

    assert called["count"] == 1, "expected the LLM generation path to run when table reasoning declines"
    assert result["found"] is True
    assert result["answer"] == "From the LLM path."


def test_generate_answer_node_is_unaffected_by_table_reasoning_when_no_table_evidence_present():
    """A question that LOOKS computable ('highest revenue') but has NO
    table evidence at all among the graded-relevant chunks must not
    error — it simply falls through to the normal LLM path, same as any
    other question."""
    service = _make_service()

    class _FakeMessage:
        content = '{"found": true, "answer": "Text-only answer.", "cited_chunk_ids": ["text-chunk"]}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeChat:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeCompletion()

    service.chat = _FakeChat()

    text_chunk = {
        "chunk_id": "text-chunk",
        "source_file": "01_product_overview.md",
        "section": "Intro",
        "text": "Some prose with no table in it.",
        "score": 0.7,
        "block_type": "text",
    }
    state = _graph_state("What is the highest revenue?", [text_chunk], ["text-chunk"])

    result = service._generate_answer(state)

    assert result["found"] is True
    assert result["answer"] == "Text-only answer."