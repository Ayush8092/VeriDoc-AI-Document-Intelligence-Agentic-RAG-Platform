"""Figure/chart intelligence: query-time reasoning over already-extracted
visual objects (Phase 6 spec item 10).

Mirrors `app.rag.table_reasoning`'s design closely — same module
structure, same "operate on already-retrieved chunks, return `None`/
`answered=False` rather than a guess" fail-closed philosophy, same
citation handling via `app.rag.llm.validate_citations`. Two real
differences from table reasoning, both because the underlying data is
different in kind:

1. **Deterministic where the data supports it, LLM-assisted otherwise.**
   The spec's "prefer deterministic Python calculations over LLM
   arithmetic" rule is stated explicitly for TABLES (item 9); item 10
   (figures/charts) has no equivalent hard constraint — it asks for
   "chart understanding, extracting chart values, trend identification,
   figure summarization, answering questions about figures", which spans
   things a chart's structured `series`/`values` genuinely can answer
   exactly (`highest_value`, `lowest_value`, `trend`) and things that
   can't be computed at all from numbers alone (summarizing a photo,
   describing a diagram with no series data). So: `compute_chart_operation`
   is exactly as deterministic as `app.rag.table_reasoning`'s equivalent
   — zero LLM calls, real `min`/`max`/monotonicity checks on real
   extracted numbers — and is tried FIRST; `answer_figure_question` falls
   back to a grounded LLM call (through `app.rag.llm.generate_answer`,
   the same evidence-only, citation-validated path `/ask` itself uses —
   not a second, less-constrained prompt) only when the deterministic
   path can't answer, or the visual isn't a chart at all (a photo/
   diagram/screenshot has no series data to compute from).
2. **The structured data source is `chunk.chart_data`** (see
   `app/chunking.py`'s `Chunk.chart_data` and `app/vectorstore.py`'s
   `chart_data_json` field), populated end-to-end from
   `app.documents.figures.chart_understanding`'s extraction at ingestion
   time — `{"title", "x_axis_label", "y_axis_label", "legend", "series":
   [{"name", "values": [number, ...]}]}`. A chunk with no `chart_data`
   (chart extraction failed, low confidence, or this is a non-chart
   visual) simply can't be computed over deterministically — this module
   treats that exactly like `table_reasoning` treats a table with no
   parseable numeric column: fall through, don't guess.

Every claim this module returns preserves figure ID (`object_id`), page,
bounding box, source document, and confidence — via
`app.rag.llm.validate_citations`'s existing citation shape (already
Phase 4-aware; see its `object_id`/`bbox`/`visual_type`/`chart_type`/
`caption` fields), not a parallel, figure-specific citation format.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from app.rag.llm import generate_answer, validate_citations
from app.retrieval import hybrid_retrieve_single

_log = logging.getLogger(__name__)


class FigureReasoningError(ValueError):
    """No figure/chart evidence at all was retrieved for the question —
    distinct from `answer_figure_question` returning `answered=False`
    (evidence exists, but couldn't be confidently used; see module
    docstring).
    """


class ChartOperation(str, Enum):
    MAX = "max"
    MIN = "min"
    TREND = "trend"
    SUM = "sum"
    AVERAGE = "average"


_MAX_WORDS = re.compile(r"\b(highest|largest|maximum|greatest|\bmax\b|peak)\b", re.IGNORECASE)
_MIN_WORDS = re.compile(r"\b(lowest|smallest|minimum|least|\bmin\b|trough)\b", re.IGNORECASE)
_TREND_WORDS = re.compile(r"\b(trend|increasing|decreasing|rising|falling|growth|declin\w*|over time)\b", re.IGNORECASE)
_SUM_WORDS = re.compile(r"\b(total|sum(?:med)?|add up|combined)\b", re.IGNORECASE)
_AVG_WORDS = re.compile(r"\b(average|mean)\b", re.IGNORECASE)


def detect_chart_operation(question: str) -> ChartOperation | None:
    """Which deterministic chart operation (if any) `question` asks for.
    `None` means "not a chart-value question at all" — callers should
    not invoke `compute_chart_operation` for those; see
    `app.rag.query_classifier`'s `FIGURE_QUERY`/`CHART_QUERY` routing.
    """
    q = question or ""
    if _TREND_WORDS.search(q):
        return ChartOperation.TREND
    if _MAX_WORDS.search(q):
        return ChartOperation.MAX
    if _MIN_WORDS.search(q):
        return ChartOperation.MIN
    if _SUM_WORDS.search(q):
        return ChartOperation.SUM
    if _AVG_WORDS.search(q):
        return ChartOperation.AVERAGE
    return None


@dataclass
class ChartOperationResult:
    operation: ChartOperation
    value: float | None = None
    series_name: str | None = None
    trend_direction: str | None = None  # "increasing" | "decreasing" | "flat" | "mixed"
    label: str = ""
    explanation: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).strip()


_MATCH_THRESHOLD = 0.55


def _best_series(question: str, series: list[dict]) -> dict | None:
    """Which of `chart_data["series"]` the question is most plausibly
    about, by name — same normalized-substring-then-similarity matching
    `app.rag.table_reasoning._best_match_index` uses for table columns
    (deterministic string comparison, not an LLM call). A chart with
    exactly one series is unambiguous regardless of whether its name
    matches anything in the question — most extracted charts are
    single-series, and requiring an explicit name match would make this
    module fail on the common case for no real benefit.
    """
    named = [s for s in series if s.get("name") and s.get("values")]
    if len(series) == 1 and series[0].get("values"):
        return series[0]
    if not named:
        return series[0] if series and series[0].get("values") else None

    import difflib

    q_norm = _normalize(question)
    best, best_score = None, 0.0
    for s in named:
        name_norm = _normalize(s["name"])
        score = 1.0 if name_norm and name_norm in q_norm else difflib.SequenceMatcher(None, q_norm, name_norm).ratio()
        if score > best_score:
            best, best_score = s, score
    if best_score < _MATCH_THRESHOLD:
        return named[0]  # still just one series in practice for most extracted charts — best-effort, not a guess at VALUES
    return best


def _numeric_values(series: dict) -> list[float]:
    out = []
    for v in series.get("values") or []:
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            try:
                out.append(float(v.replace(",", "").replace("$", "").replace("%", "").strip()))
            except ValueError:
                continue
    return out


def compute_chart_operation(chart_data: dict, question: str, operation: ChartOperation | None = None) -> ChartOperationResult | None:
    """Deterministic computation over an already-extracted chart's
    `series` values — zero LLM calls (see module docstring). Returns
    `None` (never a guess) when `chart_data` has no usable series data,
    or `operation` couldn't be determined.
    """
    if not chart_data or not isinstance(chart_data, dict):
        return None
    series_list = chart_data.get("series") or []
    if not series_list:
        return None

    op = operation or detect_chart_operation(question)
    if op is None:
        return None

    series = _best_series(question, series_list)
    if series is None:
        return None
    values = _numeric_values(series)
    if not values:
        return None

    name = series.get("name") or chart_data.get("title") or "the series"

    if op == ChartOperation.MAX:
        v = max(values)
        return ChartOperationResult(
            operation=op, value=v, series_name=name,
            label=f"Highest value in {name}: {v:g}",
            explanation=f"The highest value in '{name}' is {v:g}, out of {len(values)} data points.",
        )
    if op == ChartOperation.MIN:
        v = min(values)
        return ChartOperationResult(
            operation=op, value=v, series_name=name,
            label=f"Lowest value in {name}: {v:g}",
            explanation=f"The lowest value in '{name}' is {v:g}, out of {len(values)} data points.",
        )
    if op == ChartOperation.SUM:
        v = sum(values)
        return ChartOperationResult(
            operation=op, value=v, series_name=name,
            label=f"Total of {name}: {v:g}",
            explanation=f"The sum of all {len(values)} values in '{name}' is {v:g}.",
        )
    if op == ChartOperation.AVERAGE:
        v = sum(values) / len(values)
        return ChartOperationResult(
            operation=op, value=v, series_name=name,
            label=f"Average of {name}: {v:g}",
            explanation=f"The average of {len(values)} values in '{name}' is {v:g}.",
        )
    if op == ChartOperation.TREND:
        if len(values) < 2:
            return None  # a trend needs at least two points — one value has no direction
        # Deterministic monotonicity check, not a fabricated narrative:
        # "increasing"/"decreasing" only when EVERY consecutive step
        # moves the same direction; "mixed" when it doesn't, rather than
        # picking the more dramatic-sounding label from noisy data.
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        if all(d > 0 for d in diffs):
            direction = "increasing"
        elif all(d < 0 for d in diffs):
            direction = "decreasing"
        elif all(d == 0 for d in diffs):
            direction = "flat"
        else:
            direction = "mixed"
        net_change = values[-1] - values[0]
        return ChartOperationResult(
            operation=op, value=net_change, series_name=name, trend_direction=direction,
            label=f"Trend in {name}: {direction} ({net_change:+g} overall)",
            explanation=(
                f"'{name}' is {direction} across its {len(values)} data points "
                f"(from {values[0]:g} to {values[-1]:g}, a net change of {net_change:+g})."
            ),
        )
    return None


def try_compute_from_chunks(visual_chunks: list[dict], question: str, operation: ChartOperation | None = None) -> dict | None:
    """Given ALREADY-RETRIEVED figure/chart chunks (e.g. `state["chunks"]`
    inside `app.rag.graph`), try to compute a deterministic chart answer.

    Returns the same `{"answered": True, "mode": "deterministic", ...}`
    shape `answer_figure_question` returns on success, or `None` when
    nothing was computable — mirrors
    `app.rag.table_reasoning.try_compute_from_chunks` exactly (see its
    docstring for why this split exists). Deliberately does NOT fall back
    to an LLM call here (unlike `answer_figure_question`) — the graph
    -embedded fast path only ever wants the free, deterministic answer;
    if that's unavailable it should fall through to the graph's own
    `generate_answer` node, not make a SECOND LLM call outside the
    graph's normal citation-validated flow.
    """
    op = operation if operation is not None else detect_chart_operation(question)
    for chunk in visual_chunks:
        if chunk.get("block_type") not in ("chart", "figure", "visual"):
            continue
        chart_data = chunk.get("chart_data")
        if not chart_data:
            continue
        result = compute_chart_operation(chart_data, question, operation=op)
        if result is not None:
            citations = validate_citations([chunk["chunk_id"]], [chunk])
            return {
                "answered": True,
                "mode": "deterministic",
                "operation": result.operation.value,
                "value": result.value,
                "series_name": result.series_name,
                "trend_direction": result.trend_direction,
                "label": result.label,
                "explanation": result.explanation,
                "citations": citations,
            }
    return None


def answer_figure_question(
    index,
    embeddings,
    chat,
    settings,
    question: str,
    *,
    allowed_owner_ids: frozenset[str] | None = None,
    source_files: frozenset[str] | None = None,
) -> dict:
    """Retrieve the most relevant figure/chart chunk(s) for `question` and
    answer it — deterministically when `compute_chart_operation` can
    (see module docstring), otherwise via a grounded LLM call restricted
    to the retrieved visual evidence (never free-form: the same
    `generate_answer`/citation-validation path `/ask` uses).

    Raises `FigureReasoningError` if no figure/chart evidence at all was
    retrieved. Returns `{"answered": False, ...}` (not an exception) if
    evidence exists but neither the deterministic nor the LLM path could
    produce a grounded answer — callers (`app/rag/graph.py`) should treat
    that as "let the normal /ask pipeline try", not a hard failure.
    """
    breakdown = hybrid_retrieve_single(
        index, embeddings, settings, question, allowed_owner_ids=allowed_owner_ids, source_files=source_files
    )
    visual_chunks = [c for c in breakdown["fused"] if c.get("block_type") in ("chart", "figure", "visual")]
    if not visual_chunks:
        raise FigureReasoningError("No figure/chart evidence was retrieved for this question.")

    op = detect_chart_operation(question)
    deterministic = try_compute_from_chunks(visual_chunks, question, operation=op)
    if deterministic is not None:
        return deterministic

    # No deterministic answer (not a value/trend question, or the
    # matched chart has no usable series data) — fall back to a grounded
    # LLM answer restricted to the retrieved visual evidence, same
    # evidence-only contract /ask's generate_answer already enforces
    # (prompt-injection-defended context wrapping, label-indirection
    # citation validation — see app/rag/llm.py).
    top_chunks = visual_chunks[:5]
    llm_result = generate_answer(chat, settings.answer_model, question, top_chunks, temperature=settings.answer_temperature)
    citations = validate_citations(llm_result["cited_chunk_ids"], top_chunks)
    if llm_result["found"]:
        return {
            "answered": True,
            "mode": "llm",
            "operation": op.value if op else None,
            "value": None,
            "label": llm_result["answer"],
            "explanation": llm_result["answer"],
            "citations": citations,
        }

    return {
        "answered": False,
        "operation": op.value if op else None,
        "reason": (
            "Figure/chart evidence was found, but no chart data or grounded answer could be "
            "confidently produced from it."
        ),
        "citations": validate_citations([visual_chunks[0]["chunk_id"]], [visual_chunks[0]]),
    }