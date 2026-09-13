"""One-off smoke test proving app/api/conversations.py's create/list/get/
delete flow actually works end-to-end through TestClient (not just
imports) — real DB, real tenant-scoping logic, mocked LLM call only.
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from app.core.config import get_settings
    from app.db.session import reset_engine_for_tests
    get_settings.cache_clear()
    reset_engine_for_tests()
    from app.main import app
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    reset_engine_for_tests()


def test_create_list_get_delete_conversation_anonymous(client):
    resp = client.post("/conversations", json={"title": "test convo", "document_ids": []})
    assert resp.status_code == 201, resp.text
    convo = resp.json()
    convo_id = convo["id"]
    assert convo["owner_id"] is None
    assert convo["anonymous_session_id"] is not None
    anon_id = convo["anonymous_session_id"]

    resp = client.get("/conversations", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["conversations"]) == 1

    resp = client.get(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 200

    # Different anonymous session cannot see it
    resp = client.get(f"/conversations/{convo_id}", params={"anonymous_session_id": "someone-else"})
    assert resp.status_code == 404

    resp = client.delete(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 204

    resp = client.get(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 404


def test_send_message_persists_turn(client, monkeypatch):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo = resp.json()
    anon_id = convo["anonymous_session_id"]

    from app.api import conversations as conv_module

    class FakeService:
        chat = object()
        def ask(self, question, allowed_owner_ids=None, source_files=None):
            return {"answer": "42", "citations": [], "query_type": "LOOKUP"}

    monkeypatch.setattr(conv_module, "get_service", lambda: FakeService())

    resp = client.post(
        f"/conversations/{convo['id']}/messages",
        json={"question": "what is it?", "anonymous_session_id": anon_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assistant_message"]["content"] == "42"
    assert body["user_message"]["content"] == "what is it?"

    resp = client.get(f"/conversations/{convo['id']}/messages", params={"anonymous_session_id": anon_id})
    assert len(resp.json()) == 2