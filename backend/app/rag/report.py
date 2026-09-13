"""Grounded report generation (Phase 6, spec item 6).

A report is an ASSEMBLY of already-grounded pieces this module doesn't
duplicate the grounding logic for:

    - executive_summary / key findings -> `app.rag.summarization`
      (EXECUTIVE mode, run across every selected document together)
    - detailed_comparison / contradictions -> `app.rag.comparison`
      (`compare_documents` for exactly 2 documents, `compare_multiple`
      for more) — only run when 2+ documents are selected; a
      single-document report has no comparison section.
    - risks -> `app.rag.summarization` (KEY_RISKS mode)

Every section's content is exactly what those modules already returned
— citations included — assembled into one structured report object; no
new LLM call re-states or paraphrases what they found (spec: "the
report must never invent information", most directly satisfied by not
giving the model a second chance to rephrase already-grounded content).

**Recommendations are the one section that's explicitly model reasoning,
not a document restatement** — spec item 6: "If a recommendation is
model-generated reasoning rather than directly stated in documents,
explicitly label it as an inference/recommendation." `_generate_recommendations`
is the only LLM call in this module that is allowed to reason beyond
what's literally stated (e.g. "given both policies require 90 days
notice, plan renewal timelines accordingly") — and every recommendation
in the response carries `basis: "inference"` (as opposed to
`"evidence"` for the grounded sections) so the frontend/reader can never
mistake one for the other. Recommendations still cite the findings they
were reasoned FROM (not fabricated evidence), and a report with zero
comparison points / summary content produces zero recommendations
rather than the model inventing something to say.
"""

from __future__ import annotations

import logging

from app.rag.comparison import compare_documents, compare_multiple
from app.rag.llm import chat_json
from app.rag.multi_doc import ResolvedDocument
from app.rag.summarization import SummaryMode, summarize_documents

_log = logging.getLogger(__name__)

#: Recommendations returned — bounded for the same reason every other
#: LLM-decided-length list in Phase 6 is (app.rag.extraction.MAX_SCHEMA_FIELDS,
#: app.rag.comparison.MAX_COMPARISON_POINTS).
MAX_RECOMMENDATIONS = 10

_RECOMMENDATION_SYSTEM = (
    "You are given the FINDINGS section of a report (a list of finding_id -> text, already "
    "grounded in source documents). Based ONLY on these findings, produce a short list of "
    "practical recommendations or things to watch out for — reasoning that follows from the "
    "findings but is not itself something a document explicitly stated.\n\n"
    "For each recommendation, output:\n"
    "- text: the recommendation.\n"
    "- based_on_finding_ids: the finding_id(s) this recommendation follows from.\n\n"
    "If the findings don't support any meaningful recommendation, return an empty list — never "
    "invent a recommendation not grounded in the given findings.\n\n"
    'Reply with a single JSON object: {"recommendations": [{"text": ..., "based_on_finding_ids": [...]}]}'
)


def _generate_recommendations(chat, model: str, findings: list[dict]) -> list[dict]:
    if not findings:
        return []
    id_to_finding = {str(i): f for i, f in enumerate(findings)}
    listing = "\n".join(f"finding_id {fid}: {f['text']}" for fid, f in id_to_finding.items())
    data = chat_json(chat, model, _RECOMMENDATION_SYSTEM, listing, purpose="report_recommendations")
    raw = data.get("recommendations", []) if isinstance(data.get("recommendations"), list) else []

    out = []
    for item in raw[:MAX_RECOMMENDATIONS]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        based_on = item.get("based_on_finding_ids", [])
        finding_refs = []
        if isinstance(based_on, list):
            for fid in based_on:
                f = id_to_finding.get(str(fid).strip())
                if f is not None:
                    finding_refs.append({"text": f["text"], "citations": f.get("citations", [])})
        out.append(
            {
                "text": text,
                "basis": "inference",  # spec item 6: always explicitly labeled, never confused with evidence
                "based_on_findings": finding_refs,
            }
        )
    return out


def generate_report(
    index, embeddings, chat, settings, title: str, resolved: list[ResolvedDocument], allowed_owner_ids
) -> dict:
    """Assemble a full report over `resolved` documents.

    Returns:
        {
          "title": str,
          "documents": [{"id", "title", "filename"}, ...],
          "executive_summary": str,
          "key_findings": [{"text", "citations"}, ...],
          "detailed_comparison": {...} | None,   # compare_documents/compare_multiple's own shape, or None for a single document
          "risks": [{"text", "citations"}, ...],
          "recommendations": [{"text", "basis": "inference", "based_on_findings": [...]}, ...],
          "appendix": {"total_documents", "total_chunks_considered"},
        }
    """
    if not resolved:
        raise ValueError("At least one document must be selected for a report.")

    exec_summary_result = summarize_documents(index, embeddings, chat, settings, resolved, SummaryMode.EXECUTIVE, allowed_owner_ids)
    executive_summary = exec_summary_result["combined_summary"] or (
        next(iter(exec_summary_result["per_document"].values()))["summary"] if len(resolved) == 1 else ""
    )
    key_findings = exec_summary_result["combined_points"] or (
        next(iter(exec_summary_result["per_document"].values()))["points"] if len(resolved) == 1 else []
    )

    risks_result = summarize_documents(index, embeddings, chat, settings, resolved, SummaryMode.KEY_RISKS, allowed_owner_ids)
    risks = risks_result["combined_points"] or (
        next(iter(risks_result["per_document"].values()))["points"] if len(resolved) == 1 else []
    )

    detailed_comparison = None
    if len(resolved) == 2:
        detailed_comparison = compare_documents(
            index, embeddings, chat, settings, title, resolved, allowed_owner_ids
        )
    elif len(resolved) > 2:
        detailed_comparison = compare_multiple(
            index, embeddings, chat, settings, title, resolved, allowed_owner_ids
        )

    # Recommendations reason over findings + (if present) comparison
    # points that flag a real difference/conflict — the parts of the
    # report most likely to actually warrant a recommendation.
    findings_for_recommendations = list(key_findings)
    if detailed_comparison:
        for point in detailed_comparison.get("points", []):
            relation = point.get("relation") or point.get("consensus")
            if relation in ("different", "conflicting", "disagree"):
                text = point.get("summary_a") or point.get("aspect", "")
                citations = point.get("evidence_a", []) + point.get("evidence_b", [])
                if not citations and "evidence_by_document" in point:
                    for cites in point["evidence_by_document"].values():
                        citations.extend(cites)
                if text:
                    findings_for_recommendations.append({"text": f"{point.get('aspect', '')}: {text}", "citations": citations})

    recommendations = _generate_recommendations(chat, settings.answer_model, findings_for_recommendations)

    total_chunks = sum(pd.get("chunk_count", 0) for pd in exec_summary_result["per_document"].values())

    return {
        "title": title,
        "documents": [{"id": r.id, "title": r.title, "filename": r.filename} for r in resolved],
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "detailed_comparison": detailed_comparison,
        "risks": risks,
        "recommendations": recommendations,
        "appendix": {"total_documents": len(resolved), "total_chunks_considered": total_chunks},
    }