"""Claim-level grounding (Phase 3A).

Document/chunk-level citation validation (`llm.validate_citations`) only
answers "did the model cite a chunk that was actually retrieved?" — it
says nothing about whether the cited chunk *actually supports* the
specific sentence it's attached to. A model can cite a real, retrieved
chunk that happens to be about the right document while still asserting
something that chunk doesn't say (a fabricated number, an inverted
condition, a detail from a different section entirely).

This module goes one level deeper:

    Answer
    ├── Claim 1 → checked against its cited evidence chunk(s) → supported?
    ├── Claim 2 → checked against its cited evidence chunk(s) → supported?
    └── Claim 3 → checked against its cited evidence chunk(s) → supported?

Two LLM-backed steps, both structured JSON (same `llm.chat_json` machinery
already used for grading/rewriting/answering — see `app/rag/llm.py`):

1. `extract_claims` — split the generated answer into short, checkable
   claims. A "claim" here is a single factual assertion, not a full
   sentence necessarily (a sentence can carry more than one checkable
   fact) — see the prompt.
2. `check_claim_support` — for each claim, given every chunk the answer
   actually cited (labeled EVIDENCE_1/EVIDENCE_2/... — the same
   label-indirection pattern `llm._evidence_context` already uses for
   generation, so the model is never asked to reproduce a real chunk_id),
   judge SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED *and* say which
   specific labeled evidence block(s) it relied on. This is a single
   combined call per claim — see "Cost, Phase 3A hardening" below for why
   that matters — and it directly produces the claim -> chunk_id
   association the module previously only approximated via a second,
   much more expensive pass.

Neither step can invent a citation or override `validate_citations` — this
module only ever grades claims against chunk_ids that already passed
code-level validation; it never changes which citations reach the API
response. Its output (`citation_precision`, `citation_recall`,
`grounded_answer_rate` contribution) is additive metadata for the
response and for evaluation (`app/evaluation/metrics.py`), not a gate on
whether an answer is returned — refusal is still decided upstream by
`grade_chunks` / `validate_citations` (see `app/rag/graph.py`).

Cost (Phase 3A hardening — problem #4): the original implementation made
1 (extract) + N_claims (label each claim against ALL cited text
concatenated) + N_claims * N_cited_chunks (a second full pass, re-checking
every claim against every individual chunk, just to approximate which
chunk "counted" for precision) LLM calls per answer — genuinely
combinatorial, and the second pass was pure waste: it re-derived
information the first pass's per-claim label already implied. This
version makes exactly 1 + N_claims calls: each per-claim call already
asks the model which specific evidence block(s) it used, so
`citation_precision` (Phase 3A hardening — problem #5: precise
claim/citation association, `{"claim", "support": [...], "status"}`) is
computed for free from data the single combined call already returned,
with no extra requests. `grounding_calls` and `grounding_latency_ms` are
returned by `ground_answer` so this cost is visible in traces/evaluation
rather than only asserted in this docstring (token-level accounting is
not wired — `llm.chat_json` doesn't currently surface the provider's
`usage` object — so it is intentionally left unreported rather than
fabricated; see metric-integrity principle in the project spec).
"""

from __future__ import annotations

import logging
import time

from app.rag.llm import chat_json

_log = logging.getLogger(__name__)

SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
_VALID_LABELS = {SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED}


_CLAIM_EXTRACTION_SYSTEM = (
    "You split an answer into short, independently checkable factual "
    "claims. Each claim should be a single self-contained factual "
    "assertion (a figure, a date, a condition, a definition, a named "
    "relationship). Do not include hedging language, meta-commentary, or "
    "claims about what is NOT stated — only positive factual assertions "
    "the answer makes. If the answer makes no checkable factual claims "
    "(e.g. it is a refusal), return an empty list.\n\n"
    "Reply with a single JSON object and nothing else:\n"
    '{"claims": ["<claim 1>", "<claim 2>", ...]}'
)

_CLAIM_SUPPORT_SYSTEM = (
    "You judge whether a claim is supported by a set of labeled evidence "
    "blocks (EVIDENCE_1, EVIDENCE_2, ...). Reply with exactly one label:\n"
    "SUPPORTED - at least one evidence block directly states or clearly "
    "entails the claim.\n"
    "PARTIALLY_SUPPORTED - the evidence is related and consistent with "
    "the claim but does not fully state every detail in it (this "
    "includes a claim whose parts are each stated by DIFFERENT evidence "
    "blocks, none of which alone states the whole claim).\n"
    "UNSUPPORTED - no evidence block states the claim, contradicts it, or "
    "the claim is not addressed by any evidence block at all.\n\n"
    "Also report, BY LABEL, every evidence block that actually supports "
    "the claim (partially or fully) — this is what a citation click-"
    "through would show a user, so include every block that is genuinely "
    "load-bearing, not just the single strongest one. Use an empty list "
    "if the label is UNSUPPORTED.\n\n"
    "Judge only from the literal content of the evidence blocks — never "
    "use outside knowledge.\n\n"
    "An evidence block may be a FIGURE, CHART, or other VISUAL object "
    "(its Content starts with a bracketed tag like '[chart]'). A "
    "caption or title ALONE (e.g. 'Figure 3: Quarterly revenue') is "
    "sufficient support only for a claim that the figure/chart EXISTS "
    "or is broadly ABOUT that topic — it is NOT sufficient support for "
    "a claim about a SPECIFIC value, trend, or comparison (e.g. "
    "'Revenue peaks in Q4') unless that specific value/trend is itself "
    "stated in the block's Content (extracted chart title/axes/legend/"
    "series/values). A specific-value claim backed only by a caption, "
    "with no matching extracted data in the Content, is UNSUPPORTED.\n\n"
    "Reply with a single JSON object and nothing else:\n"
    '{"label": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED", '
    '"supporting_evidence": ["EVIDENCE_1", ...], '
    '"reason": "<one short sentence>"}'
)


def extract_claims(chat, model: str, answer: str) -> list[str]:
    """Split `answer` into short factual claims. Never raises; degrades to []."""
    if not answer or not answer.strip():
        return []
    data = chat_json(chat, model, _CLAIM_EXTRACTION_SYSTEM, f"Answer:\n{answer}", purpose="grounding_extract")
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        return []
    return [str(c).strip() for c in claims if str(c).strip()]


def _evidence_context(chunks: list[dict]) -> tuple[str, dict[str, str]]:
    """Build an EVIDENCE_N-labeled context block + label -> chunk_id map.

    Duplicated (rather than imported) from `app.rag.llm._evidence_context`:
    that helper is private to `llm.py` and this module intentionally keeps
    its own copy so a change to the generation-prompt's evidence
    formatting doesn't silently change what the grounding check sees, and
    vice versa — the two use the same label-indirection *pattern* for the
    same reason (never let the model see/return a real chunk_id) but are
    independent prompts that can evolve separately.
    """
    labels: dict[str, str] = {}
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        label = f"EVIDENCE_{i}"
        labels[label] = c.get("chunk_id", "")
        text = c.get("text", "") or c.get("snippet", "")
        blocks.append(f"{label}\nSource: {c.get('source_file', '')}\nContent:\n{text}")
    context = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    return context, labels


def check_claim_support(chat, model: str, claim: str, chunks: list[dict]) -> dict:
    """Judge one claim against every cited chunk in a SINGLE LLM call.

    Returns:
        {
          "label": SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED,
          "reason": str,
          "supporting_chunk_ids": [chunk_id, ...],  # resolved from the
              model's EVIDENCE_N labels; never a label the model wasn't
              actually shown, and never invented — see `_evidence_context`.
        }

    An empty/unavailable evidence set, or any parsing failure, is treated
    as UNSUPPORTED — silence is not evidence.
    """
    real_chunks = [c for c in chunks if (c.get("text") or c.get("snippet"))]
    if not real_chunks:
        return {"label": UNSUPPORTED, "reason": "no evidence text available", "supporting_chunk_ids": []}

    context, label_to_chunk_id = _evidence_context(real_chunks)
    user_text = f"Claim: {claim}\n\nEvidence:\n{context}"
    data = chat_json(chat, model, _CLAIM_SUPPORT_SYSTEM, user_text, purpose="grounding_check")

    label = str(data.get("label", "")).strip().upper()
    if label not in _VALID_LABELS:
        label = UNSUPPORTED

    raw_labels = data.get("supporting_evidence", [])
    supporting_chunk_ids: list[str] = []
    if isinstance(raw_labels, list):
        for raw in raw_labels:
            chunk_id = label_to_chunk_id.get(str(raw).strip())
            if chunk_id and chunk_id not in supporting_chunk_ids:
                supporting_chunk_ids.append(chunk_id)
    # A label the model claims as SUPPORTED/PARTIALLY_SUPPORTED but with no
    # resolvable supporting_evidence is inconsistent output — treat it as
    # UNSUPPORTED rather than trust an unexplained label (fail closed).
    if label != UNSUPPORTED and not supporting_chunk_ids:
        label = UNSUPPORTED

    return {
        "label": label,
        "reason": str(data.get("reason", "")).strip(),
        "supporting_chunk_ids": supporting_chunk_ids,
    }


def ground_answer(
    chat,
    model: str,
    answer: str,
    cited_chunks: list[dict],
    max_claims: int = 8,
) -> dict:
    """Run claim extraction + per-claim support checking for one answer.

    Makes exactly `1 + len(claims)` LLM calls (see module docstring, "Cost"
    — down from the previous `1 + N + N*M`): one call to extract claims,
    then one combined call per claim that both grades it AND identifies
    which specific cited chunk(s) support it.

    `max_claims` bounds the number of LLM calls this makes per request —
    an answer honestly shouldn't have dozens of independently-checkable
    claims, and bounding this keeps grounding latency predictable (see
    `app/evaluation/metrics.py`'s latency breakdown, Phase 3A requirement
    #40).

    Returns:
        {
          "claims": [
              {"claim": str, "label": str, "status": str, "reason": str,
               "support": [{"chunk_id": str, "page": int|None}, ...]},
              ...
          ],
          "grounded_claim_rate": float in [0, 1] (SUPPORTED / total, or
              1.0 when there are no claims to check — a refusal or a
              claim-free answer is vacuously fully grounded),
          "citation_precision": float in [0, 1] — fraction of cited
              chunks that supported at least one claim (a citation that
              supports nothing the answer actually says is a false
              citation), now read directly off each claim's resolved
              `supporting_chunk_ids` rather than a separate LLM pass,
          "grounding_calls": int — total LLM calls this invocation made,
          "grounding_latency_ms": float — wall-clock time spent in this
              function's LLM calls,
        }
    """
    started = time.monotonic()
    calls = 0

    claims = extract_claims(chat, model, answer)[:max_claims]
    calls += 1 if answer and answer.strip() else 0
    if not claims:
        return {
            "claims": [],
            "grounded_claim_rate": 1.0,
            "citation_precision": 1.0,
            "grounding_calls": calls,
            "grounding_latency_ms": round((time.monotonic() - started) * 1000, 2),
        }

    chunk_by_id = {c["chunk_id"]: c for c in cited_chunks if c.get("chunk_id")}

    graded = []
    supported_count = 0
    chunks_supporting_something: set[str] = set()
    for claim in claims:
        result = check_claim_support(chat, model, claim, cited_chunks)
        calls += 1
        if result["label"] == SUPPORTED:
            supported_count += 1
        chunks_supporting_something.update(result["supporting_chunk_ids"])
        # Phase 4.1(4) fix: `support` used to carry only chunk_id/page,
        # so a claim grounded in a chart/figure lost exactly the
        # provenance (object_id, chart_type, bbox, ...) the doc's own
        # worked example ("Claim: 'Revenue peaks in Q4.' -> Chart object
        # ID = abc123, Page = 4, Chart type = bar") calls for. Every
        # field here is read straight off the already-retrieved chunk
        # (`chunk_by_id`, built from `cited_chunks` above) — never from
        # the model's `check_claim_support` output, which only ever
        # supplies WHICH chunk_id(s) support the claim, not any of the
        # chunk's own metadata. A claim cannot cause a fabricated
        # object_id/bbox/chart_type to appear here: the model has no
        # channel to provide those values, only to select among chunks
        # that already carry them.
        support = []
        for cid in result["supporting_chunk_ids"]:
            source_chunk = chunk_by_id.get(cid, {})
            support.append(
                {
                    "chunk_id": cid,
                    "page": source_chunk.get("page_start"),
                    "block_type": source_chunk.get("block_type", "text"),
                    "object_id": source_chunk.get("object_id"),
                    "visual_type": source_chunk.get("visual_type"),
                    "chart_type": source_chunk.get("chart_type"),
                    "caption": source_chunk.get("caption"),
                    "bbox": source_chunk.get("bbox"),
                    "coordinate_space": source_chunk.get("coordinate_space"),
                }
            )
        graded.append(
            {
                "claim": claim,
                "label": result["label"],
                "status": result["label"],
                "reason": result["reason"],
                "support": support,
            }
        )

    grounded_claim_rate = supported_count / len(claims) if claims else 1.0

    # Citation precision at the chunk level: does each cited chunk support
    # at least one claim? Read directly from the union of every claim's
    # resolved `supporting_chunk_ids` above — no extra LLM calls.
    if not cited_chunks:
        citation_precision = 0.0 if claims else 1.0
    else:
        citation_precision = len(chunks_supporting_something) / len(cited_chunks)

    return {
        "claims": graded,
        "grounded_claim_rate": round(grounded_claim_rate, 4),
        "citation_precision": round(citation_precision, 4),
        "grounding_calls": calls,
        "grounding_latency_ms": round((time.monotonic() - started) * 1000, 2),
    }


def citation_recall(cited_chunk_ids: list[str], required_chunk_ids: list[str]) -> float | None:
    """Fraction of `required_chunk_ids` (ground-truth evidence for a benchmark
    question) that were actually cited. Returns `None` when there's no
    ground truth to compare against (live traffic, not a benchmark run) —
    the caller should omit the field rather than report a misleading 0/0.
    """
    if not required_chunk_ids:
        return None
    required = set(required_chunk_ids)
    cited = set(cited_chunk_ids)
    return round(len(required & cited) / len(required), 4)