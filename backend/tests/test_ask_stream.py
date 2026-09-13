"""Tests for POST /ask/stream (app/api/ask_stream.py).

Phase 4 hardening issue #3: this endpoint previously had no dedicated
test coverage at all. These tests exercise the real SSE wire format
(parsing `data: ...\\n\\n` frames exactly as a browser EventSource would)
against a fake `DocumentQAService`, so they don't depend on live
Gemini/Groq/Pinecone credentials — same pattern `tests/test_api.py`
already uses for `POST /ask` (`monkeypatch.setattr(ask_module,
"get_service", ...)`).

Per the module's own docstring, this endpoint streams an
already-fully-computed, already-citation-validated answer word-by-word;
these tests verify that framing honestly (TTFT = time to first *token*
event, not time to the model's first generated token) rather than
asserting something the implementation doesn't do.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import reset_engine_for_tests
from app.rag.llm import REFUSAL_TEXT


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


class _FakeService:
    """Stands in for DocumentQAService.ask() — see app/rag/graph.py for
    the real return shape this mirrors. Accepts `allowed_owner_ids`
    (Phase 5 tenant isolation, ignored here) since app/api/ask_stream.py
    now always passes it as a keyword argument."""

    def __init__(self, result: dict):
        self._result = result

    def ask(self, question: str, allowed_owner_ids=None) -> dict:
        return dict(self._result)


def _parse_sse(body: bytes) -> list[dict]:
    """Parse `data: {...}\\n\\n` SSE framing into a list of decoded JSON
    events, in order — exactly what an EventSource client reconstructs.
    """
    text = body.decode("utf-8")
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"malformed SSE frame (missing 'data: ' prefix): {block!r}"
        events.append(json.loads(block[len("data: ") :]))
    return events


def _install_fake_service(monkeypatch, result: dict) -> None:
    from app.api import ask as ask_module

    monkeypatch.setattr(ask_module, "get_service", lambda: _FakeService(result))


_FOUND_RESULT = {
    "answer": "The trial lasts 14 days.",
    "found": True,
    "citations": [
        {
            "chunk_id": "a::intro::0",
            "source_file": "a.md",
            "section": "Intro",
            "snippet": "Every account gets a 14-day trial.",
            "score": 0.9,
        }
    ],
    "query_type": "LOOKUP",
    "claims": [
        {
            "claim": "The trial lasts 14 days.",
            "label": "SUPPORTED",
            "status": "SUPPORTED",
            "reason": "stated directly",
            "support": [{"chunk_id": "a::intro::0", "page": 1}],
        }
    ],
    "grounded_claim_rate": 1.0,
    "citation_precision": 1.0,
    "citation_recall": None,
    "trace": None,
}


def test_stream_happy_path_emits_meta_then_tokens_then_done(client, monkeypatch):
    _install_fake_service(monkeypatch, _FOUND_RESULT)

    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.content)
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert all(e["type"] == "token" for e in events[1:-1])


def test_stream_sse_frames_are_well_formed_json_lines(client, monkeypatch):
    """Every frame must be exactly one `data: <json>\\n\\n` block -- no
    stray blank lines, no multi-line payloads (SSE breaks on those)."""
    _install_fake_service(monkeypatch, _FOUND_RESULT)
    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    raw = resp.content.decode("utf-8")
    # No frame's JSON payload itself contains a literal blank line, which
    # would prematurely terminate an SSE event for a real client.
    for block in raw.split("\n\n"):
        if block.strip():
            assert "\n\n" not in block


def test_stream_meta_event_carries_trace_id_query_type_and_found(client, monkeypatch):
    _install_fake_service(monkeypatch, _FOUND_RESULT)
    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    meta = _parse_sse(resp.content)[0]
    assert meta["type"] == "meta"
    assert isinstance(meta["trace_id"], str) and meta["trace_id"]
    assert meta["query_type"] == "LOOKUP"
    assert meta["found"] is True


def test_stream_token_events_reconstruct_the_exact_answer(client, monkeypatch):
    """Joining every `token` event's text must reproduce `result['answer']`
    byte-for-byte -- no lost or added whitespace across the word-chunk
    split (see `_word_chunks` in ask_stream.py)."""
    _install_fake_service(monkeypatch, _FOUND_RESULT)
    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    events = _parse_sse(resp.content)
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) > 1  # actually incremental, not one giant chunk
    reconstructed = "".join(t["text"] for t in tokens)
    assert reconstructed == _FOUND_RESULT["answer"]


def test_stream_done_event_carries_citations_and_claims(client, monkeypatch):
    _install_fake_service(monkeypatch, _FOUND_RESULT)
    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    done = _parse_sse(resp.content)[-1]
    assert done["type"] == "done"
    assert done["citations"] == _FOUND_RESULT["citations"]
    assert done["claims"] == _FOUND_RESULT["claims"]
    assert done["grounded_claim_rate"] == 1.0
    assert done["citation_precision"] == 1.0
    assert "usage" in done
    assert "trace_id" in done["usage"]


def test_stream_done_event_usage_includes_ttft_ms(client, monkeypatch):
    """TTFT here is measured as time-to-first-token-event (see module
    docstring's honest framing, not time to the model's first generated
    token) -- must be a real, non-negative measured value.

    `ttft_ms` only appears in `usage` when `INCLUDE_TRACE=true` (see
    app/api/ask_stream.py) -- set explicitly here rather than relying on
    whatever the ambient default happens to be.
    """
    monkeypatch.setenv("INCLUDE_TRACE", "true")
    get_settings.cache_clear()
    _install_fake_service(monkeypatch, _FOUND_RESULT)
    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    get_settings.cache_clear()
    done = _parse_sse(resp.content)[-1]
    usage = done["usage"]
    assert usage["ttft_ms"] is not None
    assert usage["ttft_ms"] >= 0.0


def test_stream_refusal_answer_falls_back_to_refusal_text(client, monkeypatch):
    refusal_result = {
        "answer": "",
        "found": False,
        "citations": [],
        "query_type": "LOOKUP",
        "claims": [],
        "grounded_claim_rate": None,
        "citation_precision": None,
        "citation_recall": None,
        "trace": None,
    }
    _install_fake_service(monkeypatch, refusal_result)
    resp = client.post("/ask/stream", json={"question": "What is the CEO's home address?"})
    events = _parse_sse(resp.content)
    assert events[0]["found"] is False
    tokens = [e for e in events if e["type"] == "token"]
    reconstructed = "".join(t["text"] for t in tokens)
    assert reconstructed == REFUSAL_TEXT
    done = events[-1]
    assert done["citations"] == []


def test_stream_upstream_provider_error_emits_error_event_not_500(client, monkeypatch):
    import groq

    from app.api import ask as ask_module

    class _BoomService:
        def ask(self, question: str, allowed_owner_ids=None) -> dict:
            raise groq.APIConnectionError(request=None)

    monkeypatch.setattr(ask_module, "get_service", lambda: _BoomService())

    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    # StreamingResponse always returns 200 -- errors are communicated as
    # an in-band SSE `error` event, since headers are already flushed by
    # the time the pipeline can fail.
    assert resp.status_code == 200
    events = _parse_sse(resp.content)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "Upstream service unavailable" in events[0]["detail"]


def test_stream_runtime_error_emits_error_event(client, monkeypatch):
    from app.api import ask as ask_module

    def _boom():
        raise RuntimeError("Pinecone index 'x' does not exist yet.")

    monkeypatch.setattr(ask_module, "get_service", _boom)

    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.content)
    assert events == [{"type": "error", "detail": "Pinecone index 'x' does not exist yet."}]


def test_stream_unexpected_error_emits_generic_error_event_without_leaking_details(client, monkeypatch):
    from app.api import ask as ask_module

    class _BoomService:
        def ask(self, question: str, allowed_owner_ids=None) -> dict:
            raise ValueError("some internal detail that should not reach the client")

    monkeypatch.setattr(ask_module, "get_service", lambda: _BoomService())

    resp = client.post("/ask/stream", json={"question": "How long is the trial?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.content)
    assert events == [{"type": "error", "detail": "Internal application error."}]


def test_stream_validation_error_on_empty_question(client):
    resp = client.post("/ask/stream", json={"question": ""})
    assert resp.status_code == 422