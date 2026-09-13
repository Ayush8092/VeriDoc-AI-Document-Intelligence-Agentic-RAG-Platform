"""Table intelligence: deterministic arithmetic over already-extracted
table grids (Phase 6, spec item 9).

**Core design constraint, taken directly from the spec**: "For
calculations, prefer deterministic Python calculations over LLM
arithmetic." This module contains ZERO LLM calls. Every number it
reports comes from `float()` parsing real cell text and plain Python
`min`/`max`/`sum`/division — never from asking a model to compute
anything. Column/row target matching (turning "highest Q2 revenue" into
"the column literally labeled Q2 Revenue") is also done with plain
string matching (`difflib`), not an LLM — see `_best_match_index`'s
docstring for why, and its `fail-closed` behavior when no match is
confident.

**Scope, stated honestly** (this is a deliberately bounded first version,
not a general spreadsheet engine):
- Operates on ONE table at a time — the most relevant table chunk
  `hybrid_retrieve_single` returns for the question. A question that
  genuinely needs data from two different tables isn't handled here (it
  falls through to the normal `/ask` LLM-prose path, which can at least
  attempt it, just without this module's arithmetic guarantee).
- Assumes the table's FIRST ROW is the header row and its FIRST COLUMN
  holds row labels (e.g. region/department names) — the common case for
  the tables this project's `app.documents.tables` pipeline extracts,
  but not universal. `Cell.row`/`header_row_count` (see
  `app.documents.models.TableData`) would let a future version do this
  more precisely; that metadata isn't currently propagated into chunk
  metadata at retrieval time (only the flat `rows` grid is — see
  `app.vectorstore._match_to_result`), so this module works with what's
  actually available today rather than assuming a richer shape than the
  retrieval layer provides.
- `compute_table_operation` returns `None` (not a guess) whenever the
  requested column/row can't be matched with reasonable confidence, or
  the operation needs two targets and only one was found. Callers
  (`answer_table_question`) treat `None` as "this module can't answer
  this deterministically" and should let the normal RAG path handle it,
  not report a wrong number.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.rag.llm import validate_citations
from app.retrieval import hybrid_retrieve_single

_log = logging.getLogger(__name__)


class TableReasoningError(ValueError):
    """Raised when no table evidence at all was found for the question —
    distinct from `compute_table_operation` returning `None` (evidence
    exists, but this module couldn't confidently compute an answer from
    it; see module docstring).
    """


class TableOperation(str, Enum):
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    AVERAGE = "average"
    PERCENTAGE_CHANGE = "percentage_change"
    COMPARE = "compare"


_MAX_WORDS = re.compile(r"\b(highest|largest|maximum|greatest|\bmax\b|most)\b", re.IGNORECASE)
_MIN_WORDS = re.compile(r"\b(lowest|smallest|minimum|least|\bmin\b)\b", re.IGNORECASE)
_SUM_WORDS = re.compile(r"\b(total|sum(?:med)?|add up|combined)\b", re.IGNORECASE)
_AVG_WORDS = re.compile(r"\b(average|mean)\b", re.IGNORECASE)
_PCT_CHANGE_WORDS = re.compile(r"\b(percentage change|percent change|%\s*change|change in percentage)\b", re.IGNORECASE)
_COMPARE_WORDS = re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.IGNORECASE)

# Cell text -> float: strips currency symbols, thousands separators, a
# trailing "%", and surrounding whitespace. "$1.2M"/"1,234"/"12%" all
# parse; anything that still isn't a plain number after stripping
# (dates, free text, "N/A") correctly returns None rather than a wrong
# guess.
_NUMERIC_STRIP_RE = re.compile(r"[,$%\s]")
_MILLION_SUFFIX_RE = re.compile(r"([\d.]+)\s*[Mm]\b")
_THOUSAND_SUFFIX_RE = re.compile(r"([\d.]+)\s*[Kk]\b")


def _parse_numeric(cell_text: str) -> float | None:
    if not cell_text:
        return None
    text = cell_text.strip()
    million_match = _MILLION_SUFFIX_RE.fullmatch(text.replace(",", "").replace("$", "").strip())
    if million_match:
        try:
            return float(million_match.group(1)) * 1_000_000
        except ValueError:
            return None
    thousand_match = _THOUSAND_SUFFIX_RE.fullmatch(text.replace(",", "").replace("$", "").strip())
    if thousand_match:
        try:
            return float(thousand_match.group(1)) * 1_000
        except ValueError:
            return None
    cleaned = _NUMERIC_STRIP_RE.sub("", text)
    if not cleaned or not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).strip()


#: Below this similarity score, a candidate label match is treated as
#: "not actually referenced" rather than accepted — fail-closed (see
#: module docstring) rather than picking the least-bad option.
_MATCH_THRESHOLD = 0.55

_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are"})


def _match_score(q_norm: str, label_norm: str) -> float:
    """How confidently `label_norm` (a normalized column header or row
    label) is being referenced by `q_norm` (the normalized question).
    Shared by `_best_match_index` (single-column ops, row matching) and
    `_compute_two_column` (percentage-change/compare) so the two never
    drift into different-but-similar matching behavior.

    Three signals, the best of which wins:
    1. Exact substring — the header appears verbatim in the question
       ("Q4 Revenue" in "...highest Q4 Revenue..."). Score 1.0.
    2. Whole-string sequence ratio — good when the question IS
       basically just the label plus a word or two ("Q1 Revenue?").
       Degrades badly for a full sentence, which is exactly why signal
       3 exists.
    3. **Word overlap** (found necessary by direct execution — see
       `_compute_single_column`'s bugfix comment): the fraction of the
       label's own significant (non-stopword) words that appear
       verbatim as whole words in the question. Fixes natural phrasing
       like "from Q1 to Q4" correctly matching a header literally named
       "Q1 Revenue" (shares the word "q1") even though "Q1 Revenue"
       never appears in the question as a contiguous substring, and a
       whole-string ratio against the full sentence badly underscores
       a short header buried in a long question (verified: scored
       0.14 for a real case that should have matched confidently).
    """
    if not label_norm:
        return 0.0
    if label_norm in q_norm:
        return 1.0

    ratio_score = difflib.SequenceMatcher(None, q_norm, label_norm).ratio()

    label_words = [w for w in label_norm.split() if w not in _STOPWORDS and len(w) >= 2]
    word_overlap_score = 0.0
    if label_words:
        q_words = set(q_norm.split())
        matched_words = [w for w in label_words if w in q_words]
        word_overlap_score = len(matched_words) / len(label_words)
        # A partial-word overlap alone (e.g. only "revenue" matched out
        # of "q4 revenue", with "q4" itself NOT in the question) is
        # ambiguous — every quarter's revenue column would share that
        # word. Trust it fully in two cases: every significant word
        # matched, OR the matched word(s) include a digit-bearing token
        # (e.g. "q1", "q4", "2024") — a numbered identifier is
        # inherently far more discriminating than a shared generic noun
        # like "revenue"/"cost"/"total", and realistic phrasing very
        # often names the quarter/period without repeating the column's
        # full label ("...from Q1 to Q4?" naming a table column literally
        # called "Q1 Revenue" — verified as a real, otherwise-failing
        # case by direct execution). Anything else (partial, non-numeric
        # overlap only) is discarded rather than trusted at partial
        # credit, to avoid a new false-positive class.
        has_digit_match = any(any(ch.isdigit() for ch in w) for w in matched_words)
        if has_digit_match:
            # A numbered identifier (quarter/year/code) matching IS the
            # discriminating signal on its own — don't dilute it by the
            # fraction of OTHER, generic words that happened not to
            # match too (verified necessary by direct execution: without
            # this, "Q1"-only overlap on a "Q1 Revenue" header scored
            # 0.5, just under _MATCH_THRESHOLD=0.55, silently failing a
            # realistic percentage-change question).
            word_overlap_score = 1.0
        elif word_overlap_score < 1.0:
            word_overlap_score = 0.0

    return max(ratio_score, word_overlap_score)


def _best_match_index(question: str, labels: list[str]) -> int | None:
    """Which of `labels` (column headers, or row labels) the question is
    most plausibly asking about, via normalized substring + sequence
    -similarity matching — deterministic string comparison, never an LLM
    call (see module docstring). Returns `None` (not the "closest"
    label) when nothing clears `_MATCH_THRESHOLD`, so an unrelated
    question never silently latches onto the wrong column.
    """
    q_norm = _normalize(question)
    if not q_norm or not labels:
        return None

    best_index, best_score = None, 0.0
    for i, label in enumerate(labels):
        label_norm = _normalize(label)
        score = _match_score(q_norm, label_norm)
        if score > best_score:
            best_index, best_score = i, score

    if best_score < _MATCH_THRESHOLD:
        return None
    return best_index


def detect_operation(question: str) -> TableOperation | None:
    """Which deterministic operation (if any) the question is asking
    for. Order matters: percentage-change/compare are checked before the
    single-column aggregates since "compare the percentage change
    between Q1 and Q4" should not be misread as a plain MAX/MIN request.
    Returns `None` for a question that isn't asking for a table
    calculation at all (a plain "what is X" lookup) — callers should not
    invoke this module for those; see `app.rag.query_classifier`'s
    `AGGREGATION`/`TABLE_QUERY` routing.
    """
    q = question or ""
    if _PCT_CHANGE_WORDS.search(q):
        return TableOperation.PERCENTAGE_CHANGE
    if _COMPARE_WORDS.search(q):
        return TableOperation.COMPARE
    if _MAX_WORDS.search(q):
        return TableOperation.MAX
    if _MIN_WORDS.search(q):
        return TableOperation.MIN
    if _SUM_WORDS.search(q):
        return TableOperation.SUM
    if _AVG_WORDS.search(q):
        return TableOperation.AVERAGE
    return None


@dataclass
class TableOperationResult:
    operation: TableOperation
    value: float | None
    label: str                        # human-readable summary, e.g. "North America: $1.4M"
    row_label: str | None = None
    column_label: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    secondary_value: float | None = None    # the second operand for compare/percentage_change
    secondary_label: str | None = None
    explanation: str = ""


@dataclass
class _TableGrid:
    headers: list[str]
    row_labels: list[str]
    body: list[list[str]]              # data rows only (header row excluded), same column order as headers

    @classmethod
    def from_rows(cls, table_rows: list[list[str]]) -> "_TableGrid | None":
        if not table_rows or len(table_rows) < 2:
            return None
        headers = [str(h) for h in table_rows[0]]
        body = [list(r) for r in table_rows[1:]]
        row_labels = [row[0] if row else "" for row in body]
        return cls(headers=headers, row_labels=row_labels, body=body)

    def column_values(self, column_index: int) -> list[tuple[int, float]]:
        """`[(row_index, numeric_value), ...]` for every data row where
        that column parses as a number — rows with a non-numeric cell in
        this column (blank, "N/A", a label) are simply excluded, not
        treated as zero.
        """
        out = []
        for i, row in enumerate(self.body):
            if column_index >= len(row):
                continue
            value = _parse_numeric(row[column_index])
            if value is not None:
                out.append((i, value))
        return out

    def cell(self, row_index: int, column_index: int) -> str | None:
        if 0 <= row_index < len(self.body) and 0 <= column_index < len(self.body[row_index]):
            return self.body[row_index][column_index]
        return None


def compute_table_operation(
    table_rows: list[list[str]], question: str, operation: TableOperation | None = None
) -> TableOperationResult | None:
    """Pure function — no retrieval, no I/O, no LLM (see module
    docstring). Returns `None` whenever the operation or its target
    column/row can't be determined with reasonable confidence, rather
    than a best-effort guess.
    """
    grid = _TableGrid.from_rows(table_rows)
    if grid is None:
        return None
    op = operation or detect_operation(question)
    if op is None:
        return None

    if op in (TableOperation.MAX, TableOperation.MIN, TableOperation.SUM, TableOperation.AVERAGE):
        return _compute_single_column(grid, question, op)
    if op in (TableOperation.PERCENTAGE_CHANGE, TableOperation.COMPARE):
        return _compute_two_column(grid, question, op)
    return None


def _numeric_columns(grid: _TableGrid) -> list[int]:
    return [c for c in range(len(grid.headers)) if grid.column_values(c)]


def _compute_single_column(grid: _TableGrid, question: str, op: TableOperation) -> TableOperationResult | None:
    numeric_cols = _numeric_columns(grid)
    if not numeric_cols:
        return None

    # BUGFIX (found by direct execution against "Which region had the
    # highest Q4 revenue?" — grid: Region | Q1 Revenue | Q4 Revenue):
    # matching against ALL headers let the ROW-LABEL column ("Region")
    # win the match purely because the question's subject noun ("region")
    # happens to equal that column's header — even though the question is
    # asking about a DIFFERENT, numeric column ("Q4 Revenue"). This is not
    # a rare edge case: it's exactly the shape of the spec's own example
    # question ("Which department has the highest cost?" — "department"
    # is almost always the row-label header for this kind of table).
    # Fixed by only matching against NUMERIC column headers in the first
    # place — a MAX/MIN/SUM/AVERAGE target must be numeric anyway, so
    # restricting the candidate set removes the false positive entirely
    # rather than trying to out-clever it with a smarter scoring rule.
    numeric_headers = [grid.headers[c] for c in numeric_cols]
    matched = _best_match_index(question, numeric_headers)
    column_index = numeric_cols[matched] if matched is not None else None
    if column_index is None:
        # Unambiguous fallback: if there's exactly ONE numeric column in
        # the whole table, "highest value" can only mean that column,
        # even without an explicit header match.
        column_index = numeric_cols[0] if len(numeric_cols) == 1 else None
    if column_index is None:
        return None

    values = grid.column_values(column_index)
    if not values:
        return None

    header = grid.headers[column_index]
    if op == TableOperation.MAX:
        row_index, value = max(values, key=lambda rv: rv[1])
        row_label = grid.row_labels[row_index]
        return TableOperationResult(
            operation=op, value=value, row_label=row_label, column_label=header,
            row_index=row_index, column_index=column_index,
            label=f"{row_label}: {grid.cell(row_index, column_index)}",
            explanation=f"Highest value in column '{header}' is {value:g}, in row '{row_label}'.",
        )
    if op == TableOperation.MIN:
        row_index, value = min(values, key=lambda rv: rv[1])
        row_label = grid.row_labels[row_index]
        return TableOperationResult(
            operation=op, value=value, row_label=row_label, column_label=header,
            row_index=row_index, column_index=column_index,
            label=f"{row_label}: {grid.cell(row_index, column_index)}",
            explanation=f"Lowest value in column '{header}' is {value:g}, in row '{row_label}'.",
        )
    if op == TableOperation.SUM:
        total = sum(v for _, v in values)
        return TableOperationResult(
            operation=op, value=total, column_label=header, column_index=column_index,
            label=f"Total of '{header}': {total:g}",
            explanation=f"Sum of all {len(values)} numeric values in column '{header}' is {total:g}.",
        )
    if op == TableOperation.AVERAGE:
        avg = sum(v for _, v in values) / len(values)
        return TableOperationResult(
            operation=op, value=avg, column_label=header, column_index=column_index,
            label=f"Average of '{header}': {avg:g}",
            explanation=f"Average of {len(values)} numeric values in column '{header}' is {avg:g}.",
        )
    return None


def _compute_two_column(grid: _TableGrid, question: str, op: TableOperation) -> TableOperationResult | None:
    numeric_cols = _numeric_columns(grid)
    if len(numeric_cols) < 2:
        return None

    # Find up to two distinct header matches by scoring each numeric
    # column independently and taking the top two — rather than reusing
    # `_best_match_index` twice (which would just return the same best
    # match both times).
    scored = []
    q_norm = _normalize(question)
    for c in numeric_cols:
        label_norm = _normalize(grid.headers[c])
        score = _match_score(q_norm, label_norm)
        if score >= _MATCH_THRESHOLD:
            scored.append((score, c))
    scored.sort(key=lambda sc: sc[0], reverse=True)
    matched_cols = [c for _, c in scored[:2]]
    if len(matched_cols) < 2:
        return None
    col_a, col_b = sorted(matched_cols)  # keep original left-to-right column order (e.g. Q1 before Q4)

    # A row must be identified too (percentage change / comparison is
    # meaningless without knowing WHICH row's two values) — try a row
    # -label match first; if none matches AND the table has exactly one
    # data row, that row is unambiguous.
    row_index = _best_match_index(question, grid.row_labels)
    if row_index is None:
        if len(grid.body) == 1:
            row_index = 0
        else:
            return None

    value_a = _parse_numeric(grid.cell(row_index, col_a) or "")
    value_b = _parse_numeric(grid.cell(row_index, col_b) or "")
    if value_a is None or value_b is None:
        return None

    row_label = grid.row_labels[row_index]
    header_a, header_b = grid.headers[col_a], grid.headers[col_b]

    if op == TableOperation.PERCENTAGE_CHANGE:
        if value_a == 0:
            return None  # division by zero — cannot express as a percentage change
        pct = (value_b - value_a) / abs(value_a) * 100
        return TableOperationResult(
            operation=op, value=pct, row_label=row_label,
            column_label=f"{header_a} -> {header_b}", row_index=row_index,
            secondary_value=value_b, secondary_label=header_b,
            label=f"{row_label}: {pct:+.1f}% ({header_a} {value_a:g} -> {header_b} {value_b:g})",
            explanation=(
                f"Percentage change for '{row_label}' from '{header_a}' ({value_a:g}) to "
                f"'{header_b}' ({value_b:g}) is {pct:+.1f}%."
            ),
        )
    # COMPARE
    delta = value_b - value_a
    return TableOperationResult(
        operation=op, value=value_a, secondary_value=value_b, row_label=row_label,
        column_label=header_a, secondary_label=header_b, row_index=row_index,
        label=f"{row_label}: {header_a}={value_a:g}, {header_b}={value_b:g} (difference {delta:+g})",
        explanation=(
            f"For '{row_label}', '{header_a}' is {value_a:g} and '{header_b}' is {value_b:g} "
            f"— a difference of {delta:+g}."
        ),
    )


def try_compute_from_chunks(table_chunks: list[dict], question: str, operation: TableOperation | None = None) -> dict | None:
    """Given ALREADY-RETRIEVED table chunks (e.g. `state["chunks"]` inside
    `app.rag.graph`, already retrieved/reranked/graded by the time
    `_generate_answer` runs), try to compute a deterministic answer.

    Returns the same `{"answered": True, ...}` dict shape
    `answer_table_question` returns on success, or `None` (not a
    `{"answered": False, ...}` dict) when nothing was computable — `None`
    is the "try the next thing" signal for a caller iterating candidates,
    matching `compute_table_operation`'s own `None`-means-"can't compute"
    convention. Extracted out of `answer_table_question` (below) so the
    graph-embedded fast path (`app.rag.graph`, spec item 9) and the
    standalone/API-style entry point share this exact logic instead of
    two copies drifting apart — `answer_table_question` calls this too,
    right after its own retrieval.
    """
    op = operation if operation is not None else detect_operation(question)
    for chunk in table_chunks:
        if chunk.get("block_type") != "table" or not chunk.get("table_rows"):
            continue
        result = compute_table_operation(chunk["table_rows"], question, operation=op)
        if result is not None:
            citations = validate_citations([chunk["chunk_id"]], [chunk])
            return {
                "answered": True,
                "operation": result.operation.value,
                "value": result.value,
                "secondary_value": result.secondary_value,
                "label": result.label,
                "explanation": result.explanation,
                "row_label": result.row_label,
                "column_label": result.column_label,
                "secondary_label": result.secondary_label,
                "citations": citations,
            }
    return None


def answer_table_question(
    index,
    embeddings,
    settings,
    question: str,
    *,
    allowed_owner_ids: frozenset[str] | None = None,
    source_files: frozenset[str] | None = None,
) -> dict:
    """Retrieves the most relevant table chunk(s) for `question` and
    computes a deterministic answer from the best candidate.

    Raises `TableReasoningError` if no table evidence was retrieved at
    all. Returns `{"answered": False, ...}` (not an exception) if table
    evidence WAS found but no candidate could be confidently computed —
    that's a legitimate "this module can't help here" outcome, not a
    failure, and the caller (`app/api/reasoning.py`) should fall back to
    the normal `/ask` pipeline rather than surface an error for it.
    """
    breakdown = hybrid_retrieve_single(
        index, embeddings, settings, question, allowed_owner_ids=allowed_owner_ids, source_files=source_files
    )
    table_chunks = [c for c in breakdown["fused"] if c.get("block_type") == "table" and c.get("table_rows")]
    if not table_chunks:
        raise TableReasoningError("No table evidence was retrieved for this question.")

    op = detect_operation(question)
    result = try_compute_from_chunks(table_chunks, question, operation=op)
    if result is not None:
        return result

    return {
        "answered": False,
        "operation": op.value if op else None,
        "reason": (
            "Table evidence was found, but the referenced row/column could not be matched with "
            "confidence, or the requested calculation needs data this module could not locate."
        ),
        "citations": validate_citations([table_chunks[0]["chunk_id"]], [table_chunks[0]]),
    }