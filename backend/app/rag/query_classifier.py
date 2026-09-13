"""Query understanding: classify a question before retrieval runs.

Query type affects how the rest of the pipeline behaves (see
`app/rag/graph.py`):

- `UNANSWERABLE` / pure small talk skips retrieval entirely and refuses
  immediately instead of burning a retrieval + grading round trip.
- `COMPARISON` / `CROSS_DOCUMENT` / `MULTI_HOP` widen `query_fanout` on the
  first attempt (instead of only on the bounded retry) because a single
  vector search is less likely to be sufficient for these.
- `TABLE_QUERY` biases reranking toward table-typed chunks.
- Every type is surfaced on the API response (`query_type`) for
  observability/tracing — see `app/evaluation/metrics.py`.

Design: classification is DETERMINISTIC by default (regex/keyword rules
over the question text — cheap, instant, fully unit-testable, no network
call, no added latency on the hot path) with an optional LLM fallback
(`classify_query_llm`) for cases the rules are unsure about. This mirrors
the project's existing "the LLM supplies a judgment, plain Python code
makes the routing decision" pattern (see `app/rag/graph.py`'s docstring on
`grade_chunks` vs. routing) — classification itself can consult an LLM,
but nothing downstream trusts the LLM's own opinion of what it should do
next.
"""

from __future__ import annotations

import json
import re
from enum import Enum


class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    MULTI_HOP = "MULTI_HOP"
    COMPARISON = "COMPARISON"
    EXTRACTION = "EXTRACTION"
    SUMMARIZATION = "SUMMARIZATION"
    CROSS_DOCUMENT = "CROSS_DOCUMENT"
    TABLE_QUERY = "TABLE_QUERY"
    OCR_QUERY = "OCR_QUERY"
    AMBIGUOUS = "AMBIGUOUS"
    UNANSWERABLE = "UNANSWERABLE"
    # Phase 6, spec item 1 ("advanced query planning... explicit query
    # modes"). Additive only — every existing QueryType value/meaning is
    # unchanged, and every existing check in app/rag/graph.py uses
    # membership tests (`query_type in {...}`), never an exhaustive
    # match, so these four new members flow through as "not matched" ->
    # today's default behavior anywhere they aren't explicitly opted in.
    AGGREGATION = "AGGREGATION"          # "total/average/highest/lowest across..." — see app/rag/table_reasoning.py
    TIMELINE = "TIMELINE"                # "list the events/dates in order", "what happened when"
    CONTRADICTION_DETECTION = "CONTRADICTION_DETECTION"  # "do these documents disagree about..."
    REPORT_GENERATION = "REPORT_GENERATION"  # "generate a report comparing..." — see app/rag/report.py


# Query types that benefit from a wider first-attempt search fan-out
# because a single dense query is less likely to cover everything needed.
WIDE_FANOUT_TYPES = {
    QueryType.MULTI_HOP,
    QueryType.COMPARISON,
    QueryType.CROSS_DOCUMENT,
    QueryType.SUMMARIZATION,
    # Phase 6 additions — all three genuinely need evidence gathered
    # from more than one place: a contradiction/report spans multiple
    # documents by nature, and a timeline needs every dated event, not
    # just the first match. AGGREGATION deliberately excluded — most
    # aggregation questions ("highest value in this table") target one
    # already-localized table, not a wide search.
    QueryType.CONTRADICTION_DETECTION,
    QueryType.REPORT_GENERATION,
    QueryType.TIMELINE,
}

_COMPARISON_WORDS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|differ|stricter|"
    r"which (?:is|one|policy|document|plan)|better than|more than|less than)\b",
    re.IGNORECASE,
)
_CROSS_DOC_WORDS = re.compile(
    r"\b(across (?:all|both|the) (?:documents|reports|files)|all (?:three|the) "
    r"(?:reports|documents|policies)|every document|each document|contradiction|"
    # Phase 7 completion pass: added after `evaluation/run_query_planning_eval.py`
    # (the one evaluation metric in this project that's actually
    # executable without live API keys) measured CROSS_DOCUMENT
    # accuracy at 0/11 against `evaluation/datasets/phase4_v1.jsonl` —
    # every single CROSS_DOCUMENT question fell through to LOOKUP. This
    # pattern ("which document(s) ... mention/contain/reference/appear/
    # discuss") covers the most common phrasing found in that real
    # failure analysis; verified to raise in-taxonomy accuracy from
    # 0.5816 to 0.5887 on the same dataset with zero regressions in
    # `tests/test_query_classifier.py`'s existing 29 cases. Deliberately
    # NOT broadened further to chase every remaining miss (e.g. "which
    # fictional company appears across X, Y, and Z" uses a different
    # subject than "document" and was left alone) — see
    # `evaluation/README.md`'s query-planning section for the full,
    # honest confusion-matrix analysis of what remains unfixed and why.
    r"which documents?\b.{0,40}\b(?:mentions?|contains?|references?|appears?|discuss(?:es)?))\b",
    re.IGNORECASE,
)
_EXTRACTION_WORDS = re.compile(
    r"\b(extract|list all|pull out|give me (?:a list|all)|structured|schema|"
    r"parties|effective date|expiration date|renewal terms?)\b",
    re.IGNORECASE,
)
_SUMMARIZATION_WORDS = re.compile(
    r"\b(summar(?:y|ize|ise)|key points|executive summary|tl;?dr|overview of|action items|"
    r"main takeaways?)\b",
    re.IGNORECASE,
)
_TABLE_WORDS = re.compile(
    r"\b(table|row|column|cell|spreadsheet)\b",
    re.IGNORECASE,
)
_OCR_WORDS = re.compile(
    r"\b(scanned|scan|handwritten|photo of|image of|receiving log)\b",
    re.IGNORECASE,
)
_MULTI_HOP_WORDS = re.compile(
    r"\b(change between|changed from|then .* after|before .* and after|"
    r"how did .* change|what happened after|as a result of)\b",
    re.IGNORECASE,
)
# Phase 6, spec item 1 — additive new query modes. Each is checked at a
# point in `classify_query_rule_based` chosen so it never overrides an
# EXISTING, already-tested classification (see that function's inline
# comments and tests/test_query_classifier.py) — only claims phrasing
# that previously fell through to a less-specific category or LOOKUP.
_CONTRADICTION_WORDS = re.compile(
    r"\b(disagree|disagreement|inconsisten(?:t|cy|cies)|conflict(?:s|ing)?)\b",
    re.IGNORECASE,
)
_REPORT_WORDS = re.compile(
    r"\b(generate a report|write a report|create a report|produce a report|report comparing|report on)\b",
    re.IGNORECASE,
)
_AGGREGATION_WORDS = re.compile(
    r"\b(total|sum of|average|mean of|highest|lowest|maximum|minimum|\bmax\b|\bmin\b|"
    r"percentage change|percent change|calculate the)\b",
    re.IGNORECASE,
)
_TIMELINE_WORDS = re.compile(
    r"\b(timeline|chronological|in chronological order|sequence of events|"
    r"list the (?:events|dates)|when did .* happen|what happened when)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_WORDS = re.compile(
    r"^\s*(it|this|that|they|those|these)\b",
    re.IGNORECASE,
)
_GREETING_OR_META = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay)[\s!.,]*$",
    re.IGNORECASE,
)


def classify_query_rule_based(question: str) -> QueryType:
    """Deterministic classification. Order matters: more specific /
    higher-value categories are checked before falling back to LOOKUP.
    """
    q = (question or "").strip()
    if not q:
        return QueryType.UNANSWERABLE
    if _GREETING_OR_META.match(q):
        return QueryType.UNANSWERABLE
    if _AMBIGUOUS_WORDS.match(q) and len(q.split()) <= 6:
        # A short question that opens with an unresolved pronoun and has no
        # other content to disambiguate it ("What about it?") — genuinely
        # ambiguous without conversational context.
        return QueryType.AMBIGUOUS
    if _CROSS_DOC_WORDS.search(q):
        return QueryType.CROSS_DOCUMENT
    if _CONTRADICTION_WORDS.search(q):
        # Checked AFTER cross-doc so an explicit "across all documents"
        # framing keeps winning (see tests/test_query_classifier.py's
        # existing "Find contradictions across all the documents."
        # case) — this only catches disagree/inconsistent/conflict
        # phrasing that _CROSS_DOC_WORDS doesn't already claim.
        return QueryType.CONTRADICTION_DETECTION
    if _REPORT_WORDS.search(q):
        # Checked BEFORE comparison: "generate a report comparing X and
        # Y" is a report request first, a comparison request second —
        # app/rag/report.py already calls into comparison internally.
        return QueryType.REPORT_GENERATION
    if _COMPARISON_WORDS.search(q):
        return QueryType.COMPARISON
    if _SUMMARIZATION_WORDS.search(q):
        return QueryType.SUMMARIZATION
    if _EXTRACTION_WORDS.search(q):
        return QueryType.EXTRACTION
    if _AGGREGATION_WORDS.search(q):
        # Checked BEFORE the generic table check: a calculation-flavored
        # question ("highest value", "percentage change") routes to
        # app/rag/table_reasoning.py's deterministic arithmetic even
        # when it also mentions "table" — more specific/actionable than
        # a plain TABLE_QUERY lookup.
        return QueryType.AGGREGATION
    if _TABLE_WORDS.search(q):
        return QueryType.TABLE_QUERY
    if _OCR_WORDS.search(q):
        return QueryType.OCR_QUERY
    if _TIMELINE_WORDS.search(q):
        return QueryType.TIMELINE
    if _MULTI_HOP_WORDS.search(q):
        return QueryType.MULTI_HOP
    return QueryType.LOOKUP


_CLASSIFIER_SYSTEM = (
    "You classify a user's question about a document corpus into exactly "
    "one of these types:\n"
    "LOOKUP - a single fact stated in one place.\n"
    "MULTI_HOP - requires connecting facts from more than one place, "
    "including how something changed over time.\n"
    "COMPARISON - explicitly compares two or more named things.\n"
    "EXTRACTION - asks to pull out structured fields (dates, parties, "
    "amounts) rather than answer in prose.\n"
    "SUMMARIZATION - asks for a summary, key points, or overview.\n"
    "CROSS_DOCUMENT - reasons across many/all documents at once.\n"
    "TABLE_QUERY - specifically about a table, row, column, or cell.\n"
    "OCR_QUERY - specifically about a scanned document/image.\n"
    "AGGREGATION - asks for a calculation: total, average, highest/lowest, "
    "percentage change.\n"
    "TIMELINE - asks for events/dates in chronological order.\n"
    "CONTRADICTION_DETECTION - asks whether specific things disagree/conflict, "
    "without a broad 'across all documents' framing (that's CROSS_DOCUMENT).\n"
    "REPORT_GENERATION - explicitly asks to generate/write/produce a report.\n"
    "AMBIGUOUS - unclear what is being asked without more context.\n"
    "UNANSWERABLE - not a real question about the corpus (greeting, "
    "small talk, empty).\n\n"
    'Reply with a single JSON object: {"type": "<ONE_OF_THE_ABOVE>"}'
)


def classify_query_llm(chat, model: str, question: str) -> QueryType:
    """LLM-backed classification for cases the rules can't confidently place.

    Never raises: any provider/parsing failure degrades to the rule-based
    result, since query classification is a routing hint, not a
    correctness-critical judgment (see module docstring).
    """
    fallback = classify_query_rule_based(question)
    try:
        response = chat.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": f"Question: {question}"},
            ],
        )
        text = response.choices[0].message.content or "{}"
        data = json.loads(text)
        raw = str(data.get("type", "")).strip().upper()
        return QueryType(raw)
    except Exception:
        return fallback


def classify_query(question: str, chat=None, model: str | None = None, use_llm: bool = False) -> QueryType:
    """Entry point used by the graph. Rule-based by default; LLM only when
    explicitly requested and only as a refinement — never a replacement —
    of the rule-based pass, so classification is always at least as fast
    and reliable as the deterministic path.
    """
    rule_based = classify_query_rule_based(question)
    if not use_llm or chat is None or model is None:
        return rule_based
    if rule_based not in (QueryType.LOOKUP,):
        # The rules already found a specific, high-confidence signal;
        # don't pay for an LLM call to second-guess it.
        return rule_based
    return classify_query_llm(chat, model, question)