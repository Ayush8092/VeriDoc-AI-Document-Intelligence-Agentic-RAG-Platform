"""Tests for app/api/reasoning.py — POST /compare, /extract, /summarize,
/report (Phase 6 spec item 14, "wire multi-document reasoning behind
clean REST endpoints").

Same TestClient + real-SQLite-DB pattern as tests/test_auth.py and
tests/test_conversations_smoke.py: real HTTP requests through the actual
app, a real (per-test) database for Document/User rows, real bcrypt/JWT
for authenticated requests. The one thing mocked is
`app.api.reasoning.get_service` (same mocking point
tests/test_conversations_smoke.py already established for
`app.api.conversations.get_service` — these endpoints share the exact
same `DocumentQAService` singleton via `app.api.ask.get_service`) — set
to a fake service whose `.index` is a real `LocalVectorIndex` with real
upserted test chunks and whose `.chat` is a fake chat client returning
scripted, schema-valid JSON. This exercises the real
resolve_documents -> retrieval -> app.rag.* -> Pydantic-response control
flow end-to-end over real HTTP, not just the underlying app.rag.*
functions directly (already covered by tests/test_comparison.py,
tests/test_extraction.py, tests/test_summarization.py,
tests/test_report.py).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.session import get_session_factory, reset_engine_for_tests
from app.vectorstore_local import LocalVectorIndex


# --- shared fixtures -----------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()
    reset_engine_for_tests()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_engine_for_tests()


def _register(client: TestClient, email: str) -> str:
    resp = client.post("/auth/register", json={"email": email, "password": "hunter2pass"})
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_document(filename: str, owner_id: int | None = None, title: str | None = None) -> int:
    """Insert a Document row directly (no upload endpoint involved — these
    tests are about /compare //extract//summarize//report, not
    ingestion) using the SAME settings/engine the just-created TestClient
    app is bound to.
    """
    from app.db.models import Document

    settings = get_settings()
    factory = get_session_factory(settings)
    session = factory()
    try:
        doc = Document(source_root="uploads", filename=filename, file_type="pdf", title=title or filename, owner_id=owner_id)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        return doc.id
    finally:
        session.close()


def _fixed_settings() -> Settings:
    return Settings(
        _env_file=None,
        hybrid_retrieval_enabled=False,
        embedding_dimension=4,
        pinecone_namespace="test-ns",
        score_threshold=0.0,
    )


class _FixedEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def _upsert(index: LocalVectorIndex, chunk_id: str, source_file: str, owner_id: str, text: str):
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
                    "chunk_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                },
            }
        ],
        namespace="test-ns",
    )


def _json_response(payload: dict):
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


class _ScriptedChat:
    """Returns `payload` (a fixed, schema-valid JSON dict) for every LLM
    call this request makes, regardless of which of comparison/
    extraction/summarization/report's several distinct prompt types
    issued it — good enough for these API-boundary tests, whose job is
    to check request/response wiring, auth, and tenant isolation, not
    reason about report generation's multi-call internals (already
    covered thoroughly by tests/test_report.py).
    """

    def __init__(self, payload: dict):
        self._payload = payload
        self.call_count = 0

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    self.call_count += 1
                    return _json_response(self._payload)

        self.chat = _Chat()


def _install_fake_service(monkeypatch, index, chat, settings=None):
    from app.api import reasoning as reasoning_module

    class _FakeService:
        def __init__(self):
            self.index = index
            self.embeddings = _FixedEmbeddings()
            self.chat = chat
            self.settings = settings or _fixed_settings()

    monkeypatch.setattr(reasoning_module, "get_service", lambda: _FakeService())


# ===========================================================================
# POST /compare
# ===========================================================================


def test_compare_success_returns_two_doc_shape_with_evidence(client, monkeypatch, tmp_path):
    doc_a = _make_document("policy_a.pdf", title="Policy A")
    doc_b = _make_document("policy_b.pdf", title="Policy B")

    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Notice period is 30 days.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Notice period is 60 days.")

    payload = {
        "points": [
            {
                "aspect": "notice period",
                "relation": "different",
                "summary_a": "30 days",
                "summary_b": "60 days",
                "evidence_a": ["EVIDENCE_%d_1" % doc_a],
                "evidence_b": ["EVIDENCE_%d_1" % doc_b],
            }
        ]
    }
    _install_fake_service(monkeypatch, index, _ScriptedChat(payload))

    resp = client.post("/compare", json={"topic": "notice period", "document_ids": [doc_a, doc_b]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["topic"] == "notice period"
    assert body["document_a"]["id"] == doc_a
    assert body["document_b"]["id"] == doc_b
    assert body["points"][0]["relation"] == "different"
    assert body["points"][0]["evidence_a"][0]["source_file"] == "policy_a.pdf"
    assert body["points"][0]["evidence_b"][0]["source_file"] == "policy_b.pdf"


def test_compare_success_multi_doc_shape_for_three_documents(client, monkeypatch, tmp_path):
    doc_a = _make_document("a.pdf")
    doc_b = _make_document("b.pdf")
    doc_c = _make_document("c.pdf")

    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        _upsert(index, f"{name}::0", name, owner_id="", text=f"Content of {name}.")

    payload = {
        "points": [
            {
                "aspect": "topic",
                "consensus": "agree",
                "summary_by_document": {str(doc_a): "x", str(doc_b): "x", str(doc_c): "x"},
                "evidence_by_document": {
                    str(doc_a): [f"EVIDENCE_{doc_a}_1"],
                    str(doc_b): [f"EVIDENCE_{doc_b}_1"],
                    str(doc_c): [f"EVIDENCE_{doc_c}_1"],
                },
            }
        ]
    }
    _install_fake_service(monkeypatch, index, _ScriptedChat(payload))

    resp = client.post("/compare", json={"topic": "topic", "document_ids": [doc_a, doc_b, doc_c]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["documents"]) == 3
    assert body["points"][0]["consensus"] == "agree"


def test_compare_validation_requires_at_least_two_document_ids(client, monkeypatch, tmp_path):
    """Pydantic-level validation (CompareRequest.document_ids min_length=2)
    — must reject BEFORE resolve_documents/get_service are ever reached.
    """
    resp = client.post("/compare", json={"topic": "x", "document_ids": [1]})
    assert resp.status_code == 422


def test_compare_validation_rejects_empty_topic(client):
    resp = client.post("/compare", json={"topic": "", "document_ids": [1, 2]})
    assert resp.status_code == 422


def test_compare_validation_malformed_request_missing_fields(client):
    resp = client.post("/compare", json={"document_ids": [1, 2]})  # missing topic
    assert resp.status_code == 422


def test_compare_missing_document_returns_404(client, monkeypatch, tmp_path):
    doc_a = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({"points": []}))

    resp = client.post("/compare", json={"topic": "x", "document_ids": [doc_a, 999999]})
    assert resp.status_code == 404


def test_compare_invalid_document_id_type_returns_422(client):
    resp = client.post("/compare", json={"topic": "x", "document_ids": ["not-an-int", 2]})
    assert resp.status_code == 422


def test_compare_cross_tenant_document_access_returns_404_not_leaked(client, monkeypatch, tmp_path):
    """The core Phase 5/6 security invariant: a user cannot compare
    another tenant's private document by guessing its id — 404 (same as
    'does not exist'), never a 403 that would confirm the id is valid
    but someone else's (see app.rag.multi_doc's module docstring
    convention).
    """
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")

    settings = get_settings()
    from app.security.auth import decode_access_token

    alice_claims = decode_access_token(alice_token, settings)
    alice_id = int(alice_claims["sub"])

    private_doc = _make_document("private.pdf", owner_id=alice_id)
    public_doc = _make_document("public.pdf", owner_id=None)

    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({"points": []}))

    resp = client.post(
        "/compare",
        json={"topic": "x", "document_ids": [private_doc, public_doc]},
        headers=_auth(bob_token),
    )
    assert resp.status_code == 404

    # Sanity: the OWNER can access their own document fine.
    resp_owner = client.post(
        "/compare",
        json={"topic": "x", "document_ids": [private_doc, public_doc]},
        headers=_auth(alice_token),
    )
    assert resp_owner.status_code == 200


def test_compare_unauthenticated_can_still_access_public_documents(client, monkeypatch, tmp_path):
    """Auth is OPTIONAL for these endpoints (get_current_user_optional) —
    an anonymous caller must still be able to compare public-corpus
    documents, matching /ask's own no-auth-required default.
    """
    doc_a = _make_document("a.pdf", owner_id=None)
    doc_b = _make_document("b.pdf", owner_id=None)
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({"points": []}))

    resp = client.post("/compare", json={"topic": "x", "document_ids": [doc_a, doc_b]})
    assert resp.status_code == 200


def test_compare_comparison_error_translates_to_422(client, monkeypatch, tmp_path):
    """ComparisonError (e.g. empty topic reaching the module layer some
    other way) must translate to 422, not a 500 — error-translation
    contract from app/api/reasoning.py's _handle_upstream_errors /
    explicit except ComparisonError clause.
    """
    doc_a = _make_document("a.pdf")
    doc_b = _make_document("b.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module
    from app.rag.comparison import ComparisonError

    def _boom(*a, **k):
        raise ComparisonError("forced failure for testing")

    monkeypatch.setattr(reasoning_module, "compare_documents", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({"points": []}))

    resp = client.post("/compare", json={"topic": "x", "document_ids": [doc_a, doc_b]})
    assert resp.status_code == 422
    assert "forced failure" in resp.text


def test_compare_upstream_provider_error_translates_to_503(client, monkeypatch, tmp_path):
    doc_a = _make_document("a.pdf")
    doc_b = _make_document("b.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module
    from app.rag.llm import LLMProviderError

    def _boom(*a, **k):
        raise LLMProviderError("provider unavailable")

    monkeypatch.setattr(reasoning_module, "compare_documents", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({"points": []}))

    resp = client.post("/compare", json={"topic": "x", "document_ids": [doc_a, doc_b]})
    assert resp.status_code == 503


def test_compare_unexpected_exception_translates_to_500(client, monkeypatch, tmp_path):
    doc_a = _make_document("a.pdf")
    doc_b = _make_document("b.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module

    def _boom(*a, **k):
        raise KeyError("something truly unexpected")

    monkeypatch.setattr(reasoning_module, "compare_documents", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({"points": []}))

    resp = client.post("/compare", json={"topic": "x", "document_ids": [doc_a, doc_b]})
    assert resp.status_code == 500


# ===========================================================================
# POST /extract
# ===========================================================================


def test_extract_success_returns_fields_with_citations(client, monkeypatch, tmp_path):
    doc_id = _make_document("contract.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c::0", "contract.pdf", owner_id="", text="The contract start date is January 1, 2024.")

    payload = {
        "start_date": {"value": "2024-01-01", "status": "found", "confidence": 0.9, "evidence": [f"EVIDENCE_{doc_id}_1"]}
    }
    chat = _ScriptedChat(payload)
    _install_fake_service(monkeypatch, index, chat)

    resp = client.post("/extract", json={"document_ids": [doc_id], "schema": {"start_date": "date"}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "start_date" in body["fields"]
    assert body["documents"][0]["id"] == doc_id


def test_extract_unsupported_information_returns_null(client, monkeypatch, tmp_path):
    doc_id = _make_document("contract.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c::0", "contract.pdf", owner_id="", text="This document says nothing about a signing bonus.")

    payload = {"signing_bonus": {"value": None, "status": "not_found", "confidence": 0.0, "evidence": []}}
    _install_fake_service(monkeypatch, index, _ScriptedChat(payload))

    resp = client.post("/extract", json={"document_ids": [doc_id], "schema": {"signing_bonus": "number"}})
    assert resp.status_code == 200, resp.text
    field = resp.json()["fields"]["signing_bonus"]
    assert field["value"] is None
    assert field["status"] == "not_found"


def test_extract_validation_invalid_schema_type_returns_422(client, monkeypatch, tmp_path):
    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/extract", json={"document_ids": [doc_id], "schema": {"x": "not_a_real_type"}})
    assert resp.status_code == 422


def test_extract_validation_exceeds_max_field_limit_returns_422(client, monkeypatch, tmp_path):
    from app.rag.extraction import MAX_SCHEMA_FIELDS

    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    schema = {f"field_{i}": "string" for i in range(MAX_SCHEMA_FIELDS + 1)}
    resp = client.post("/extract", json={"document_ids": [doc_id], "schema": schema})
    assert resp.status_code == 422


def test_extract_validation_empty_document_ids_returns_422(client):
    resp = client.post("/extract", json={"document_ids": [], "schema": {"x": "string"}})
    assert resp.status_code == 422


def test_extract_missing_document_returns_404(client, monkeypatch, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/extract", json={"document_ids": [999999], "schema": {"x": "string"}})
    assert resp.status_code == 404


def test_extract_cross_tenant_document_access_returns_404(client, monkeypatch, tmp_path):
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")
    settings = get_settings()
    from app.security.auth import decode_access_token

    alice_id = int(decode_access_token(alice_token, settings)["sub"])
    private_doc = _make_document("private.pdf", owner_id=alice_id)

    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post(
        "/extract", json={"document_ids": [private_doc], "schema": {"x": "string"}}, headers=_auth(bob_token)
    )
    assert resp.status_code == 404


def test_extract_schema_validation_error_translates_to_422(client, monkeypatch, tmp_path):
    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module
    from app.rag.extraction import SchemaValidationError

    def _boom(*a, **k):
        raise SchemaValidationError("bad schema for testing")

    monkeypatch.setattr(reasoning_module, "extract_structured", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/extract", json={"document_ids": [doc_id], "schema": {"x": "string"}})
    assert resp.status_code == 422
    assert "bad schema" in resp.text


# ===========================================================================
# POST /summarize
# ===========================================================================


def test_summarize_success_single_document(client, monkeypatch, tmp_path):
    doc_id = _make_document("policy.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "p::0", "policy.pdf", owner_id="", text="Employees get 15 days of paid leave per year.")

    payload = {"summary": "15 days of paid leave.", "points": [{"text": "15 days", "evidence": ["EVIDENCE_1"]}]}
    _install_fake_service(monkeypatch, index, _ScriptedChat(payload))

    resp = client.post("/summarize", json={"document_ids": [doc_id], "mode": "document"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "document"
    assert str(doc_id) in body["per_document"]


def test_summarize_success_multi_document_has_combined_summary(client, monkeypatch, tmp_path):
    doc_a = _make_document("a.pdf")
    doc_b = _make_document("b.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "a.pdf", owner_id="", text="Doc A content.")
    _upsert(index, "b::0", "b.pdf", owner_id="", text="Doc B content.")

    def _respond(kwargs):
        import re

        user = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        if "point_id" in user:
            ids = re.findall(r"point_id (\d+):", user)
            payload = {"summary": "combined", "points": [{"text": "combined point", "source_point_ids": ids}]}
        else:
            labels = re.findall(r"EVIDENCE_\d+", user)
            payload = {"points": [{"text": f"from {lbl}", "evidence": [lbl]} for lbl in labels[:1]]}
        return _json_response(payload)

    class _Chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return _respond(kwargs)

    class _MultiCallChat:
        chat = _Chat()

    _install_fake_service(monkeypatch, index, _MultiCallChat())

    resp = client.post("/summarize", json={"document_ids": [doc_a, doc_b], "mode": "document"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["per_document"].keys()) == {str(doc_a), str(doc_b)}
    assert body["combined_summary"]


def test_summarize_validation_invalid_mode_returns_422(client, monkeypatch, tmp_path):
    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/summarize", json={"document_ids": [doc_id], "mode": "not_a_real_mode"})
    assert resp.status_code == 422
    assert "Supported" in resp.text


def test_summarize_validation_empty_document_ids_returns_422(client):
    resp = client.post("/summarize", json={"document_ids": [], "mode": "document"})
    assert resp.status_code == 422


def test_summarize_missing_document_returns_404(client, monkeypatch, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/summarize", json={"document_ids": [999999], "mode": "document"})
    assert resp.status_code == 404


def test_summarize_cross_tenant_document_access_returns_404(client, monkeypatch, tmp_path):
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")
    settings = get_settings()
    from app.security.auth import decode_access_token

    alice_id = int(decode_access_token(alice_token, settings)["sub"])
    private_doc = _make_document("private.pdf", owner_id=alice_id)

    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post(
        "/summarize", json={"document_ids": [private_doc], "mode": "document"}, headers=_auth(bob_token)
    )
    assert resp.status_code == 404


def test_summarize_summarization_error_translates_to_422(client, monkeypatch, tmp_path):
    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module
    from app.rag.summarization import SummarizationError

    def _boom(*a, **k):
        raise SummarizationError("forced failure")

    monkeypatch.setattr(reasoning_module, "summarize_documents", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/summarize", json={"document_ids": [doc_id], "mode": "document"})
    assert resp.status_code == 422


# ===========================================================================
# POST /report
# ===========================================================================


def test_report_success_single_document(client, monkeypatch, tmp_path):
    doc_id = _make_document("policy.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "p::0", "policy.pdf", owner_id="", text="Employees get 15 days of paid leave per year.")

    def _respond(kwargs):
        import re

        system = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
        user = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        if "FINDINGS section" in system:
            finding_ids = re.findall(r"finding_id (\S+):", user)
            payload = {
                "recommendations": [{"text": "Review this.", "based_on_finding_ids": finding_ids[:1]}]
                if finding_ids
                else []
            }
        elif "point_id" in user:
            ids = re.findall(r"point_id (\d+):", user)
            payload = {"summary": "combined summary", "points": [{"text": "combined", "source_point_ids": ids}]}
        else:
            labels = re.findall(r"EVIDENCE_\d+", user)
            payload = {"points": [{"text": f"finding from {lbl}", "evidence": [lbl]} for lbl in labels[:1]]}
        return _json_response(payload)

    class _Chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return _respond(kwargs)

    class _ReportChat:
        chat = _Chat()

    _install_fake_service(monkeypatch, index, _ReportChat())

    resp = client.post("/report", json={"title": "Leave Policy Report", "document_ids": [doc_id]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["title"] == "Leave Policy Report"
    assert isinstance(body["executive_summary"], str)
    assert isinstance(body["key_findings"], list)
    assert body["detailed_comparison"] is None  # single document -> no comparison section
    assert isinstance(body["risks"], list)
    for rec in body["recommendations"]:
        assert rec["basis"] == "inference"
    assert body["appendix"]["total_documents"] == 1


def test_report_validation_empty_title_returns_422(client):
    resp = client.post("/report", json={"title": "", "document_ids": [1]})
    assert resp.status_code == 422


def test_report_validation_empty_document_ids_returns_422(client):
    resp = client.post("/report", json={"title": "Report", "document_ids": []})
    assert resp.status_code == 422


def test_report_missing_document_returns_404(client, monkeypatch, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/report", json={"title": "Report", "document_ids": [999999]})
    assert resp.status_code == 404


def test_report_cross_tenant_document_access_returns_404(client, monkeypatch, tmp_path):
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")
    settings = get_settings()
    from app.security.auth import decode_access_token

    alice_id = int(decode_access_token(alice_token, settings)["sub"])
    private_doc = _make_document("private.pdf", owner_id=alice_id)

    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post(
        "/report", json={"title": "Report", "document_ids": [private_doc]}, headers=_auth(bob_token)
    )
    assert resp.status_code == 404

    # Owner themself can still generate the report.
    resp_owner = client.post(
        "/report", json={"title": "Report", "document_ids": [private_doc]}, headers=_auth(alice_token)
    )
    assert resp_owner.status_code == 200


def test_report_value_error_translates_to_422(client, monkeypatch, tmp_path):
    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module

    def _boom(*a, **k):
        raise ValueError("forced failure")

    monkeypatch.setattr(reasoning_module, "generate_report", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/report", json={"title": "Report", "document_ids": [doc_id]})
    assert resp.status_code == 422


def test_report_upstream_provider_error_translates_to_503(client, monkeypatch, tmp_path):
    doc_id = _make_document("a.pdf")
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)

    from app.api import reasoning as reasoning_module
    from app.rag.llm import LLMProviderError

    def _boom(*a, **k):
        raise LLMProviderError("upstream down")

    monkeypatch.setattr(reasoning_module, "generate_report", _boom)
    _install_fake_service(monkeypatch, index, _ScriptedChat({}))

    resp = client.post("/report", json={"title": "Report", "document_ids": [doc_id]})
    assert resp.status_code == 503