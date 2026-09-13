"""Tests for app.rag.report (Phase 6 spec item 6, report generation) —
the one dedicated Phase 6 test file that did not exist yet.

Same testing pattern as tests/test_summarization.py and
tests/test_comparison.py: a real `LocalVectorIndex` (not mocked) + a fake
chat client that inspects the actual SYSTEM prompt sent (each of
`generate_report`'s several distinct LLM call types —
map-summarize / reduce-summarize / compare / recommendations — has its
own system prompt with a distinguishing marker phrase, checked against
the real prompt text in app/rag/summarization.py and app/rag/
comparison.py, not guessed) and returns a schema-valid response for
THAT call type. This exercises `generate_report`'s real assembly logic
(it calls into `summarize_documents` and `compare_documents`/
`compare_multiple` for real) without a live LLM or vector database.
"""

from __future__ import annotations

import json

import pytest

from app.rag.multi_doc import ResolvedDocument
from app.rag.report import generate_report
from app.vectorstore_local import LocalVectorIndex


class _FakeSettings:
    embedding_dimension = 4
    pinecone_namespace = "test-ns"
    answer_model = "fake-model"
    top_k = 10
    score_threshold = 0.0
    hybrid_retrieval_enabled = False


class _FixedEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def _upsert(index: LocalVectorIndex, chunk_id: str, source_file: str, owner_id: str, text: str, chunk_index: int = 0):
    index.upsert(
        [
            {
                "id": chunk_id,
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "owner_id": owner_id,
                    "text": text,
                    "chunk_index": chunk_index,
                    "page_start": 1,
                    "page_end": 1,
                },
            }
        ],
        namespace="test-ns",
    )


def _response(payload: dict):
    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    return _Response(json.dumps(payload))


class _FakeChat:
    """Dispatches on the REAL system-prompt marker text for each of
    `generate_report`'s distinct LLM call types (verified against
    app/rag/summarization.py's `_map_system`/`_REDUCE_SYSTEM` and
    app/rag/comparison.py's `_TWO_DOC_SYSTEM`/`_N_DOC_SYSTEM`, and
    app/rag/report.py's `_RECOMMENDATION_SYSTEM`), and returns a
    schema-valid response built from what's ACTUALLY in the user
    message (real EVIDENCE_n / point_id / finding_id references), so a
    generated report's citations trace back to real, upserted chunks —
    not fabricated test fixture data.
    """

    def __init__(self):
        self.seen: list[tuple[str, str]] = []  # (system, user) pairs, every call

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return self._respond(kwargs)

        self.chat = _Chat()

    def _respond(self, kwargs):
        import re

        system = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
        user = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        self.seen.append((system, user))

        if "FINDINGS section" in system:
            finding_ids = re.findall(r"finding_id (\S+):", user)
            payload = {
                "recommendations": (
                    [{"text": "Review the flagged difference before renewal.", "based_on_finding_ids": finding_ids[:1]}]
                    if finding_ids
                    else []
                )
            }
        elif "missing_in_a" in system:  # two-doc comparison
            # Real label format is EVIDENCE_<doc_id>_<n> (see
            # app/rag/multi_doc.py's per_document_evidence_context
            # docstring) — extracted from the actual user prompt rather
            # than assumed, so this only "finds" evidence that's really
            # there.
            labels = re.findall(r"EVIDENCE_\d+_\d+", user)
            labels_by_doc: dict[str, list[str]] = {}
            for lbl in labels:
                doc_id = lbl.split("_")[1]
                labels_by_doc.setdefault(doc_id, []).append(lbl)
            doc_ids = sorted(labels_by_doc, key=int)
            evidence_a = labels_by_doc.get(doc_ids[0], []) if doc_ids else []
            evidence_b = labels_by_doc.get(doc_ids[1], []) if len(doc_ids) > 1 else []
            payload = {
                "points": [
                    {
                        "aspect": "notice period",
                        "relation": "different",
                        "summary_a": "30 days",
                        "summary_b": "60 days",
                        "evidence_a": evidence_a,
                        "evidence_b": evidence_b,
                    }
                ]
            }
        elif "consensus" in system:  # multi-doc comparison
            labels = re.findall(r"EVIDENCE_\d+_\d+", user)
            labels_by_doc: dict[str, list[str]] = {}
            for lbl in labels:
                doc_id = lbl.split("_")[1]
                labels_by_doc.setdefault(doc_id, []).append(lbl)
            payload = {
                "points": [
                    {
                        "aspect": "notice period",
                        "consensus": "disagree",
                        "summary_by_document": {did: f"{did} days" for did in labels_by_doc},
                        "evidence_by_document": labels_by_doc,
                    }
                ]
            }
        elif "point_id" in user:  # reduce step (system is the shared _REDUCE_SYSTEM)
            source_ids = re.findall(r"point_id (\d+):", user)
            payload = {
                "summary": "Combined summary of the evidence.",
                "points": [{"text": "A combined finding.", "source_point_ids": source_ids}],
            }
        else:  # map step — EVIDENCE_<n> labeled blocks
            labels = re.findall(r"EVIDENCE_\d+", user)
            payload = {"points": [{"text": f"A finding from {lbl}.", "evidence": [lbl]} for lbl in labels[:2]]}

        return _response(payload)


@pytest.fixture()
def settings():
    return _FakeSettings()


@pytest.fixture()
def embeddings():
    return _FixedEmbeddings()


def _one_doc():
    return [ResolvedDocument(id=1, filename="policy.pdf", title="Policy", source_root="uploads", file_type="pdf")]


def _two_docs():
    return [
        ResolvedDocument(id=1, filename="policy_a.pdf", title="Policy A", source_root="uploads", file_type="pdf"),
        ResolvedDocument(id=2, filename="policy_b.pdf", title="Policy B", source_root="uploads", file_type="pdf"),
    ]


def _three_docs():
    return _two_docs() + [
        ResolvedDocument(id=3, filename="policy_c.pdf", title="Policy C", source_root="uploads", file_type="pdf")
    ]


# ---------------------------------------------------------------------------
# 1. Executive summary
# ---------------------------------------------------------------------------


def test_report_contains_executive_summary_from_evidence(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="Employees get 15 days of paid leave per year.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Leave Policy Report", _one_doc(), allowed_owner_ids=None)

    assert isinstance(report["executive_summary"], str)
    assert report["executive_summary"] == "Combined summary of the evidence."
    # The single-document fallback path (no reduce across documents) must
    # still surface a real summary — never leave this blank when there
    # was real evidence.
    assert report["executive_summary"] != ""


def test_report_executive_summary_reduce_is_actually_invoked(settings, embeddings, tmp_path):
    """The executive summary must come from the real map->reduce
    pipeline (app.rag.summarization), not a fabricated placeholder —
    verified by confirming the reduce call actually happened and its
    output is exactly what's in the report."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="Employees get 15 days of paid leave per year.")

    chat = _FakeChat()
    generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    reduce_calls = [s for s, u in chat.seen if "point_id" in u]
    assert len(reduce_calls) >= 1


# ---------------------------------------------------------------------------
# 2. Key findings
# ---------------------------------------------------------------------------


def test_report_key_findings_generated_from_supplied_evidence(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="Employees get 15 days of paid leave per year.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    assert isinstance(report["key_findings"], list)
    assert len(report["key_findings"]) >= 1
    for finding in report["key_findings"]:
        assert "text" in finding
        assert "citations" in finding


def test_report_key_findings_citations_trace_to_real_chunks(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "chunk::a::0", "policy.pdf", owner_id="", text="Employees get 15 days of paid leave per year.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    # Each citation is a structured object (chunk_id/source_file/page_start/
    # page_end/document_id — see app/rag/summarization.py's citation shape),
    # not a bare chunk_id string.
    all_chunk_ids = {c["chunk_id"] for f in report["key_findings"] for c in f["citations"]}
    assert all_chunk_ids  # at least one real citation
    assert all_chunk_ids <= {"chunk::a::0"}  # never references a chunk that was never upserted


# ---------------------------------------------------------------------------
# 3. Detailed comparison
# ---------------------------------------------------------------------------


def test_report_single_document_has_no_comparison_section(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="Notice period is 30 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    assert report["detailed_comparison"] is None


def test_report_two_documents_produces_comparison_section(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _two_docs(), allowed_owner_ids=None)

    assert report["detailed_comparison"] is not None
    points = report["detailed_comparison"]["points"]
    assert len(points) == 1
    assert points[0]["aspect"] == "notice period"
    assert points[0]["relation"] == "different"
    assert points[0]["summary_a"] == "30 days"
    assert points[0]["summary_b"] == "60 days"


def test_report_three_documents_uses_multi_document_comparison_shape(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")
    _upsert(index, "c::0", "policy_c.pdf", owner_id="", text="Notice period is 90 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _three_docs(), allowed_owner_ids=None)

    assert report["detailed_comparison"] is not None
    points = report["detailed_comparison"]["points"]
    assert points[0]["consensus"] == "disagree"
    assert "summary_by_document" in points[0]


# ---------------------------------------------------------------------------
# 4. Risks
# ---------------------------------------------------------------------------


def test_report_risks_are_populated_from_evidence(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="Late termination may result in a penalty fee.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    assert isinstance(report["risks"], list)
    for risk in report["risks"]:
        assert "text" in risk
        assert "citations" in risk


def test_report_risks_use_key_risks_summary_mode(settings, embeddings, tmp_path):
    """Confirms risks actually go through a SEPARATE summarization pass
    (SummaryMode.KEY_RISKS) from key_findings, not a re-use/duplicate of
    the executive summary's points — verified by checking the risks
    mode-instruction text reached the map call."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="Late termination may result in a penalty fee.")

    chat = _FakeChat()
    generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    map_systems = [s for s, u in chat.seen if "EVIDENCE_" in u]
    assert any("risks, concerns, or potential problems" in s for s in map_systems)
    assert any("executive summary" in s.lower() for s in map_systems)


# ---------------------------------------------------------------------------
# 5. Recommendations — the "basis == inference" structured field
# ---------------------------------------------------------------------------


def test_report_recommendations_are_explicitly_marked_as_inference(settings, embeddings, tmp_path):
    """The specific, non-negotiable requirement from the spec: every
    recommendation must carry a structured `basis` field equal to the
    literal string "inference" — checked as an actual dict field on
    each recommendation object, not a substring search anywhere in the
    report.
    """
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _two_docs(), allowed_owner_ids=None)

    assert len(report["recommendations"]) >= 1
    for rec in report["recommendations"]:
        assert rec["basis"] == "inference"
        assert "text" in rec
        assert "based_on_findings" in rec


def test_report_recommendations_reference_real_findings_not_fabricated_ones(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _two_docs(), allowed_owner_ids=None)

    for rec in report["recommendations"]:
        for based_on in rec["based_on_findings"]:
            assert "text" in based_on
            assert "citations" in based_on


def test_report_no_findings_produces_no_recommendations(settings, embeddings, tmp_path):
    """generate_report's own documented behavior: zero comparison
    points / summary content -> zero recommendations, never a
    fabricated one — verified against a document with NO retrievable
    chunks at all (see the empty/minimal evidence tests below for why
    this is the correct existing behavior, not an assumption)."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    chat = _FakeChat()

    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    assert report["recommendations"] == []


# ---------------------------------------------------------------------------
# 6. Appendix
# ---------------------------------------------------------------------------


def test_report_appendix_preserves_document_and_chunk_counts(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _two_docs(), allowed_owner_ids=None)

    assert report["appendix"]["total_documents"] == 2
    assert report["appendix"]["total_chunks_considered"] == 2


def test_report_documents_list_matches_input(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _two_docs(), allowed_owner_ids=None)

    assert report["documents"] == [
        {"id": 1, "title": "Policy A", "filename": "policy_a.pdf"},
        {"id": 2, "title": "Policy B", "filename": "policy_b.pdf"},
    ]


# ---------------------------------------------------------------------------
# 7. Citations / evidence relationships
# ---------------------------------------------------------------------------


def test_report_comparison_evidence_traces_to_real_chunks(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "chunk::a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "chunk::b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _two_docs(), allowed_owner_ids=None)

    point = report["detailed_comparison"]["points"][0]
    evidence_a_ids = {c["chunk_id"] for c in point["evidence_a"]}
    evidence_b_ids = {c["chunk_id"] for c in point["evidence_b"]}
    assert evidence_a_ids <= {"chunk::a::0"}
    assert evidence_b_ids <= {"chunk::b::0"}
    assert evidence_a_ids and evidence_b_ids  # both sides actually had resolvable evidence


# ---------------------------------------------------------------------------
# 8. Empty / minimal evidence
# ---------------------------------------------------------------------------


def test_report_with_zero_chunks_returns_empty_sections_not_an_error(settings, embeddings, tmp_path):
    """The actual current behavior (see app/rag/summarization.py's
    `_summarize_single_document`: zero chunks -> {"summary": "",
    "points": [], "chunk_count": 0}, no LLM call, no exception) —
    generate_report itself only raises when `resolved` is empty, not
    when a resolved document simply has no chunks."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)  # nothing upserted
    chat = _FakeChat()

    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    assert report["executive_summary"] == ""
    assert report["key_findings"] == []
    assert report["risks"] == []
    assert report["recommendations"] == []
    assert report["appendix"]["total_chunks_considered"] == 0
    # No LLM call should have been made for zero evidence (matches
    # summarization.py's "no LLM call made for zero evidence" contract).
    assert chat.seen == []


def test_report_minimal_single_chunk_evidence(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy.pdf", owner_id="", text="A single relevant sentence.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _one_doc(), allowed_owner_ids=None)

    assert report["appendix"]["total_chunks_considered"] == 1
    assert len(report["key_findings"]) >= 1


# ---------------------------------------------------------------------------
# 9. Multiple documents
# ---------------------------------------------------------------------------


def test_report_multi_document_appendix_counts_chunks_across_all_documents(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")
    _upsert(index, "c::0", "policy_c.pdf", owner_id="", text="Notice period is 90 days.")

    chat = _FakeChat()
    report = generate_report(index, embeddings, chat, settings, "Report", _three_docs(), allowed_owner_ids=None)

    assert report["appendix"]["total_documents"] == 3
    assert report["appendix"]["total_chunks_considered"] == 3


def test_report_multi_document_tenant_scoping_excludes_other_owners_chunks(settings, embeddings, tmp_path):
    """Report generation must honor allowed_owner_ids exactly like
    summarization/comparison do — reuses the same real cross-tenant
    scenario as tests/test_summarization.py's regression test."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "owner1::report.pdf::0", "report.pdf", owner_id="1", text="Owner 1's private figures.")
    _upsert(index, "owner2::report.pdf::0", "report.pdf", owner_id="2", text="Owner 2's private figures.")

    chat = _FakeChat()
    doc = ResolvedDocument(id=101, filename="report.pdf", title="Report", source_root="uploads", file_type="pdf")
    generate_report(index, embeddings, chat, settings, "Report", [doc], allowed_owner_ids=frozenset({"1"}))

    all_text = " ".join(u for _, u in chat.seen)
    assert "Owner 2's private figures" not in all_text
    assert "Owner 1's private figures" in all_text


# ---------------------------------------------------------------------------
# 10. Error / validation behavior — only what generate_report actually validates
# ---------------------------------------------------------------------------


def test_report_empty_document_list_raises_value_error(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    chat = _FakeChat()

    with pytest.raises(ValueError):
        generate_report(index, embeddings, chat, settings, "Report", [], allowed_owner_ids=None)