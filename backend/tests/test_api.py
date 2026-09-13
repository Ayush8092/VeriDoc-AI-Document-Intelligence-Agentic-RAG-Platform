import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.session import reset_engine_for_tests


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


def test_health_never_touches_external_services(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_not_ready_without_provider_keys(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_ask_validation_error_on_empty_question(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_returns_503_when_service_unavailable(client, monkeypatch):
    from app.api import ask as ask_module

    def _boom():
        raise RuntimeError("Pinecone index 'x' does not exist yet.")

    monkeypatch.setattr(ask_module, "get_service", _boom)

    resp = client.post("/ask", json={"question": "What is the free trial length?"})
    assert resp.status_code == 503


def test_ask_with_trace_flag_returns_stage_latencies(client, monkeypatch):
    """Phase 7 trace viewer (item 18): `?trace=true` must surface the
    real per-node timing breakdown DocumentQAService.ask() computes
    (app/rag/graph.py's `_timed_trace`/`stage_latencies`) -- previously
    computed internally but silently discarded before reaching the API
    response (found during the Phase 7 audit)."""
    from app.api import ask as ask_module

    class _FakeService:
        def ask(self, question: str, allowed_owner_ids=None) -> dict:
            return {
                "answer": "The trial lasts 14 days.",
                "found": True,
                "citations": [],
                "trace": ["classify_query: query_type=LOOKUP", "retrieve: 3 chunks"],
                "query_type": "LOOKUP",
                "claims": [],
                "grounded_claim_rate": None,
                "citation_precision": None,
                "citation_recall": None,
                "stage_latencies": [
                    {"stage": "classify_query", "elapsed_ms": 1.2, "message": "query_type=LOOKUP"},
                    {"stage": "retrieve", "elapsed_ms": 45.6, "message": "3 chunks"},
                ],
            }

    monkeypatch.setattr(ask_module, "get_service", lambda: _FakeService())

    resp = client.post("/ask?trace=true", json={"question": "How long is the trial?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage_latencies"] == [
        {"stage": "classify_query", "elapsed_ms": 1.2, "message": "query_type=LOOKUP"},
        {"stage": "retrieve", "elapsed_ms": 45.6, "message": "3 chunks"},
    ]


def test_ask_without_trace_flag_omits_stage_latencies(client, monkeypatch):
    from app.api import ask as ask_module

    class _FakeService:
        def ask(self, question: str, allowed_owner_ids=None) -> dict:
            return {
                "answer": "The trial lasts 14 days.",
                "found": True,
                "citations": [],
                "trace": ["classify_query: query_type=LOOKUP"],
                "query_type": "LOOKUP",
                "claims": [],
                "grounded_claim_rate": None,
                "citation_precision": None,
                "citation_recall": None,
                "stage_latencies": [{"stage": "classify_query", "elapsed_ms": 1.2, "message": "x"}],
            }

    monkeypatch.setattr(ask_module, "get_service", lambda: _FakeService())

    resp = client.post("/ask", json={"question": "How long is the trial?"})
    assert resp.status_code == 200
    # response_model_exclude_none=True on /ask omits None fields entirely
    # rather than serializing them as `null` -- absence is the correct
    # "not included" signal here, not a literal null in the JSON body.
    assert resp.json().get("stage_latencies") is None


def test_documents_list_empty_before_ingestion(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == {"documents": []}


def test_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/documents/upload",
        files={"files": ("bad.exe", b"not a real file", "application/octet-stream")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["rejected"][0]["filename"] == "bad.exe"


def test_upload_rejects_empty_file(client):
    resp = client.post("/documents/upload", files={"files": ("a.txt", b"", "text/plain")})
    body = resp.json()
    assert body["status"] == "error"
    assert "empty" in body["rejected"][0]["error"]


def test_frontend_placeholder_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Veridoc" in resp.content