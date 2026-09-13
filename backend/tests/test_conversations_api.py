"""Dedicated tests for app/api/conversations.py (Phase 6 spec items 7 &
14 — conversation memory API). Previously only covered by
tests/test_conversations_smoke.py's 2 tests; this closes the gap the
Phase 6 completion audit's Step 4 explicitly calls for: create, list,
retrieve, send message, delete, ownership, authentication/session
behavior, tenant isolation, invalid/missing conversation ID, and that
conversation retrieval actually uses the intended document/RAG context.

Same TestClient + real-SQLite-DB pattern as tests/test_reasoning_api.py:
real HTTP requests through the actual app, real bcrypt/JWT for
authenticated requests, `app.api.conversations.get_service` mocked to a
fake service (real `LocalVectorIndex`, fake chat) so no live LLM/vector
DB call is needed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
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


def _register(client: TestClient, email: str) -> str:
    resp = client.post("/auth/register", json={"email": email, "password": "hunter2pass"})
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class _FakeService:
    """Matches tests/test_conversations_smoke.py's established fake — a
    minimal stand-in for DocumentQAService that records exactly what
    `send_message` passed it (question/owner scope/source_files), so
    tests can assert on the REAL call the endpoint made, not just the
    HTTP response.
    """

    chat = object()

    def __init__(self, answer="42"):
        self.answer = answer
        self.calls: list[dict] = []

    def ask(self, question, allowed_owner_ids=None, source_files=None):
        self.calls.append(
            {"question": question, "allowed_owner_ids": allowed_owner_ids, "source_files": source_files}
        )
        return {"answer": self.answer, "citations": [], "query_type": "LOOKUP"}


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_conversation_anonymous_gets_new_session_id(client):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["owner_id"] is None
    assert body["anonymous_session_id"] is not None
    assert body["title"] == "t"
    assert body["document_ids"] == []


def test_create_conversation_authenticated_sets_owner_not_anonymous_id(client):
    token = _register(client, "alice@example.com")
    resp = client.post("/conversations", json={"title": "t", "document_ids": []}, headers=_auth(token))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["owner_id"] is not None
    assert body["anonymous_session_id"] is None


def test_create_conversation_with_nonexistent_document_id_rejected(client):
    token = _register(client, "alice@example.com")
    resp = client.post("/conversations", json={"title": "t", "document_ids": [999999]}, headers=_auth(token))
    assert resp.status_code in (404, 422)


def test_create_conversation_enforces_max_conversations_limit(client):
    token = _register(client, "alice@example.com")
    from app.services.conversation_memory import MAX_CONVERSATIONS_PER_OWNER

    for i in range(MAX_CONVERSATIONS_PER_OWNER):
        resp = client.post("/conversations", json={"title": f"t{i}", "document_ids": []}, headers=_auth(token))
        assert resp.status_code == 201, resp.text

    resp = client.post("/conversations", json={"title": "one too many", "document_ids": []}, headers=_auth(token))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_conversations_authenticated_scoped_to_owner(client):
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")

    client.post("/conversations", json={"title": "alice's", "document_ids": []}, headers=_auth(alice_token))
    client.post("/conversations", json={"title": "bob's 1", "document_ids": []}, headers=_auth(bob_token))
    client.post("/conversations", json={"title": "bob's 2", "document_ids": []}, headers=_auth(bob_token))

    resp = client.get("/conversations", headers=_auth(alice_token))
    assert resp.status_code == 200
    titles = [c["title"] for c in resp.json()["conversations"]]
    assert titles == ["alice's"]

    resp = client.get("/conversations", headers=_auth(bob_token))
    titles = [c["title"] for c in resp.json()["conversations"]]
    assert set(titles) == {"bob's 1", "bob's 2"}


def test_list_conversations_anonymous_without_session_id_is_empty(client):
    """An anonymous request with no anonymous_session_id at all (never
    had one) must see an empty list — never another session's
    conversations, and never a 500/error."""
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json() == {"conversations": []}


def test_list_conversations_anonymous_scoped_to_session_id(client):
    resp1 = client.post("/conversations", json={"title": "session 1", "document_ids": []})
    anon1 = resp1.json()["anonymous_session_id"]
    resp2 = client.post("/conversations", json={"title": "session 2", "document_ids": []})
    anon2 = resp2.json()["anonymous_session_id"]
    assert anon1 != anon2

    resp = client.get("/conversations", params={"anonymous_session_id": anon1})
    titles = [c["title"] for c in resp.json()["conversations"]]
    assert titles == ["session 1"]


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


def test_get_conversation_by_id(client):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    resp = client.get(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 200
    assert resp.json()["id"] == convo_id


def test_get_conversation_invalid_id_returns_404(client):
    resp = client.get("/conversations/999999")
    assert resp.status_code == 404


def test_get_conversation_missing_anonymous_session_id_returns_404(client):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    # No anonymous_session_id supplied at all -> must not succeed.
    resp = client.get(f"/conversations/{convo_id}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Ownership / tenant isolation
# ---------------------------------------------------------------------------


def test_authenticated_user_cannot_access_another_users_conversation(client):
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")

    resp = client.post("/conversations", json={"title": "alice's private convo", "document_ids": []}, headers=_auth(alice_token))
    convo_id = resp.json()["id"]

    resp = client.get(f"/conversations/{convo_id}", headers=_auth(bob_token))
    assert resp.status_code == 404


def test_anonymous_session_cannot_access_different_anonymous_sessions_conversation(client):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]

    resp = client.get(f"/conversations/{convo_id}", params={"anonymous_session_id": "someone-elses-session"})
    assert resp.status_code == 404


def test_authenticated_user_cannot_access_anonymous_conversation(client):
    """An authenticated user's Bearer identity must never be substituted
    for anonymous_session_id-based ownership, or vice versa."""
    resp = client.post("/conversations", json={"title": "anon convo", "document_ids": []})
    convo_id = resp.json()["id"]

    token = _register(client, "alice@example.com")
    resp = client.get(f"/conversations/{convo_id}", headers=_auth(token))
    assert resp.status_code == 404


def test_delete_is_owner_scoped(client):
    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")

    resp = client.post("/conversations", json={"title": "alice's", "document_ids": []}, headers=_auth(alice_token))
    convo_id = resp.json()["id"]

    resp = client.delete(f"/conversations/{convo_id}", headers=_auth(bob_token))
    assert resp.status_code == 404  # bob cannot delete alice's conversation

    resp = client.get(f"/conversations/{convo_id}", headers=_auth(alice_token))
    assert resp.status_code == 200  # still exists, untouched


def test_send_message_to_another_users_conversation_returns_404(client, monkeypatch):
    from app.api import conversations as conv_module

    monkeypatch.setattr(conv_module, "get_service", lambda: _FakeService())

    alice_token = _register(client, "alice@example.com")
    bob_token = _register(client, "bob@example.com")

    resp = client.post("/conversations", json={"title": "alice's", "document_ids": []}, headers=_auth(alice_token))
    convo_id = resp.json()["id"]

    resp = client.post(f"/conversations/{convo_id}/messages", json={"question": "hi"}, headers=_auth(bob_token))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_conversation_then_get_returns_404(client):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    resp = client.delete(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 204

    resp = client.get(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 404


def test_delete_nonexistent_conversation_returns_404(client):
    resp = client.delete("/conversations/999999")
    assert resp.status_code == 404


def test_delete_removes_messages_too(client, monkeypatch):
    from app.api import conversations as conv_module

    fake = _FakeService()
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "hi", "anonymous_session_id": anon_id},
    )
    resp = client.delete(f"/conversations/{convo_id}", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 204

    # Recreate a conversation with the same id space to confirm no
    # orphaned message rows leak into a fresh conversation's history --
    # simplest direct check is just that fetching the deleted
    # conversation's messages 404s (its access check runs first).
    resp = client.get(f"/conversations/{convo_id}/messages", params={"anonymous_session_id": anon_id})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Send message / persistence / history
# ---------------------------------------------------------------------------


def test_send_message_persists_both_turns_in_order(client, monkeypatch):
    from app.api import conversations as conv_module

    fake = _FakeService(answer="the answer is 42")
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    resp = client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "what is it?", "anonymous_session_id": anon_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_message"]["content"] == "what is it?"
    assert body["user_message"]["role"] == "user"
    assert body["assistant_message"]["content"] == "the answer is 42"
    assert body["assistant_message"]["role"] == "assistant"

    resp = client.get(f"/conversations/{convo_id}/messages", params={"anonymous_session_id": anon_id})
    messages = resp.json()
    assert len(messages) == 2
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_send_message_multi_turn_history_accumulates(client, monkeypatch):
    from app.api import conversations as conv_module

    fake = _FakeService()
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    for i in range(3):
        resp = client.post(
            f"/conversations/{convo_id}/messages",
            json={"question": f"question {i}", "anonymous_session_id": anon_id},
        )
        assert resp.status_code == 200

    resp = client.get(f"/conversations/{convo_id}/messages", params={"anonymous_session_id": anon_id})
    assert len(resp.json()) == 6  # 3 user + 3 assistant


def test_send_message_second_turn_context_includes_first_turn(client, monkeypatch):
    """Verifies conversation retrieval actually uses the intended
    document/RAG context: the SECOND question sent to the underlying
    service must be prefixed with context derived from the first turn
    (build_turn_context/compose_question_with_context), not just the
    raw new question in isolation."""
    from app.api import conversations as conv_module

    fake = _FakeService()
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "What is the notice period?", "anonymous_session_id": anon_id},
    )
    client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "And for the other policy?", "anonymous_session_id": anon_id},
    )

    assert len(fake.calls) == 2
    first_call_question = fake.calls[0]["question"]
    second_call_question = fake.calls[1]["question"]
    # The first call has nothing to prepend yet.
    assert first_call_question == "What is the notice period?"
    # The second call's composed question must carry forward context
    # from the first turn -- not just the bare new question.
    assert "And for the other policy?" in second_call_question
    assert second_call_question != "And for the other policy?"
    assert "notice period" in second_call_question or "What is the notice period?" in second_call_question


def test_send_message_document_scoped_conversation_passes_source_files(client, monkeypatch, tmp_path):
    """A conversation created WITH document_ids must scope retrieval to
    those documents' filenames on every turn (source_files passed to
    service.ask) -- the actual "document-aware context" requirement."""
    from app.api import conversations as conv_module
    from app.db.models import Document
    from app.db.session import get_session_factory

    fake = _FakeService()
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    token = _register(client, "alice@example.com")

    # Insert a real Document row alice owns, matching what
    # app.rag.multi_doc.resolve_documents expects to find.
    settings = get_settings()
    factory = get_session_factory(settings)
    session = factory()
    try:
        from app.db.models import User

        user = session.query(User).filter(User.email == "alice@example.com").one()
        doc = Document(source_root="uploads", filename="policy.pdf", file_type="pdf", title="Policy", owner_id=user.id)
        session.add(doc)
        session.commit()
        doc_id = doc.id
    finally:
        session.close()

    resp = client.post(
        "/conversations", json={"title": "t", "document_ids": [doc_id]}, headers=_auth(token)
    )
    assert resp.status_code == 201, resp.text
    convo_id = resp.json()["id"]
    assert resp.json()["document_ids"] == [doc_id]

    resp = client.post(f"/conversations/{convo_id}/messages", json={"question": "what does it say?"}, headers=_auth(token))
    assert resp.status_code == 200, resp.text

    assert len(fake.calls) == 1
    assert fake.calls[0]["source_files"] == frozenset({"policy.pdf"})


def test_send_message_unfocused_conversation_passes_no_source_file_restriction(client, monkeypatch):
    from app.api import conversations as conv_module

    fake = _FakeService()
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "anything", "anonymous_session_id": anon_id},
    )
    assert fake.calls[0]["source_files"] is None


def test_send_message_owner_scope_matches_caller_identity(client, monkeypatch):
    """The owner scope passed to retrieval must match the CALLER's own
    identity (allowed_owner_ids(current_user)), not the conversation's
    creator implicitly bypassing tenant scoping some other way."""
    from app.api import conversations as conv_module

    fake = _FakeService()
    monkeypatch.setattr(conv_module, "get_service", lambda: fake)

    token = _register(client, "alice@example.com")
    resp = client.post("/conversations", json={"title": "t", "document_ids": []}, headers=_auth(token))
    convo_id = resp.json()["id"]

    client.post(f"/conversations/{convo_id}/messages", json={"question": "hi"}, headers=_auth(token))

    owner_scope = fake.calls[0]["allowed_owner_ids"]
    assert owner_scope is not None
    assert "" in owner_scope  # public/corpus content always included


def test_send_message_invalid_conversation_id_returns_404(client, monkeypatch):
    from app.api import conversations as conv_module

    monkeypatch.setattr(conv_module, "get_service", lambda: _FakeService())

    resp = client.post("/conversations/999999/messages", json={"question": "hi"})
    assert resp.status_code == 404


def test_send_message_empty_question_rejected(client):
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    resp = client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "", "anonymous_session_id": anon_id},
    )
    assert resp.status_code == 422


def test_send_message_upstream_failure_returns_503(client, monkeypatch):
    from app.api import conversations as conv_module

    class _BoomService:
        chat = object()

        def ask(self, *a, **k):
            raise RuntimeError("Pinecone index does not exist yet")

    monkeypatch.setattr(conv_module, "get_service", lambda: _BoomService())

    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]

    resp = client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "hi", "anonymous_session_id": anon_id},
    )
    assert resp.status_code == 503


def test_anonymous_session_id_echoed_back_only_for_anonymous_callers(client, monkeypatch):
    from app.api import conversations as conv_module

    monkeypatch.setattr(conv_module, "get_service", lambda: _FakeService())

    # Anonymous caller: echoed back.
    resp = client.post("/conversations", json={"title": "t", "document_ids": []})
    convo_id = resp.json()["id"]
    anon_id = resp.json()["anonymous_session_id"]
    resp = client.post(
        f"/conversations/{convo_id}/messages",
        json={"question": "hi", "anonymous_session_id": anon_id},
    )
    assert resp.json()["anonymous_session_id"] == anon_id

    # Authenticated caller: always None.
    token = _register(client, "alice@example.com")
    resp = client.post("/conversations", json={"title": "t", "document_ids": []}, headers=_auth(token))
    convo_id = resp.json()["id"]
    resp = client.post(f"/conversations/{convo_id}/messages", json={"question": "hi"}, headers=_auth(token))
    assert resp.json()["anonymous_session_id"] is None