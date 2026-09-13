"""Tests for app/services/conversation_memory.py (Phase 6, spec item 7).

Split into two groups:
  - Pure-function tests (new_anonymous_session_id, resolve_identity,
    compose_question_with_context) — no DB, no mocks. These were
    directly executed (against a stand-alone import harness stubbing out
    fastapi/sqlalchemy/app.rag.llm) during development, since this
    module's own import chain pulls in pydantic-dependent packages not
    installed in the environment this was authored in — every assertion
    below matches what that harness actually observed, not a guess.
  - DB-backed tests (create/list/get/delete conversation, add/list
    messages, rolling-summary refresh, turn-context building) — use the
    same `settings`/`session_scope` fixture pattern as
    `tests/test_db_models.py`, a real (temp, SQLite) database. Written
    against the actual current model/service code, matching every field
    name and function signature precisely, but NOT executed in the
    environment this was authored in (sqlalchemy itself isn't installed
    there — see the repo's own test-running notes). Treat these as
    ready to run, not as already-confirmed-passing.

**Tenant isolation is the single most safety-critical thing this module
does** (spec: "one user's memory can never be accessed by another
user") — `test_get_conversation_or_404_*` below is the direct regression
coverage for that guarantee, at the service layer (independent of
`tests/test_conversations_smoke.py`'s API-layer coverage of the same
property).
"""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.db.models import Conversation, ConversationMessage, Document, User
from app.db.session import init_db, reset_engine_for_tests, session_scope
from app.services import conversation_memory as cm


@pytest.fixture()
def settings(tmp_path):
    reset_engine_for_tests()
    s = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    init_db(s)
    yield s
    reset_engine_for_tests()


# =============================================================================
# Pure functions — no DB. Directly executed during development (see
# module docstring); assertions match observed behavior exactly.
# =============================================================================


def test_new_anonymous_session_id_is_long_and_unique():
    id1 = cm.new_anonymous_session_id()
    id2 = cm.new_anonymous_session_id()
    assert len(id1) > 20
    assert id1 != id2


class _FakeUser:
    id = 42


def test_resolve_identity_authenticated_user_wins_no_merge_with_leftover_anon_token():
    """A logged-in request that also happens to carry an old anonymous
    session id must NOT be merged into that anonymous identity — see
    resolve_identity's docstring for exactly why."""
    owner_id, anon = cm.resolve_identity(_FakeUser(), "leftover-anon-token-from-before-login")
    assert owner_id == 42
    assert anon is None


def test_resolve_identity_anonymous_session_preserved_when_provided():
    owner_id, anon = cm.resolve_identity(None, "existing-session-id")
    assert owner_id is None
    assert anon == "existing-session-id"


def test_resolve_identity_generates_new_anonymous_session_when_neither_given():
    owner_id, anon = cm.resolve_identity(None, None)
    assert owner_id is None
    assert anon is not None and len(anon) > 20


def test_compose_question_with_context_empty_prefix_is_passthrough():
    assert cm.compose_question_with_context("", "What is the price?") == "What is the price?"


def test_compose_question_with_context_includes_prefix_and_question():
    result = cm.compose_question_with_context("Earlier: discussed pricing.", "What about the Pro plan?")
    assert "Earlier: discussed pricing." in result
    assert "Current question: What about the Pro plan?" in result


# =============================================================================
# DB-backed — create/list/get/delete conversation
# =============================================================================


def test_create_conversation_authenticated(settings):
    with session_scope(settings) as session:
        user = User(email="a@example.com", hashed_password="x")
        session.add(user)
        session.flush()

        convo = cm.create_conversation(
            session, current_user=user, anonymous_session_id=None, title="Test", document_ids=[]
        )
        assert convo.owner_id == user.id
        assert convo.anonymous_session_id is None
        assert convo.title == "Test"
        assert json.loads(convo.document_ids) == []


def test_create_conversation_anonymous(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(
            session, current_user=None, anonymous_session_id=None, title="Anon convo", document_ids=[]
        )
        assert convo.owner_id is None
        assert convo.anonymous_session_id is not None


def test_create_conversation_enforces_max_per_owner(settings, monkeypatch):
    monkeypatch.setattr(cm, "MAX_CONVERSATIONS_PER_OWNER", 2)
    with session_scope(settings) as session:
        user = User(email="a@example.com", hashed_password="x")
        session.add(user)
        session.flush()

        cm.create_conversation(session, current_user=user, anonymous_session_id=None, title="1", document_ids=[])
        cm.create_conversation(session, current_user=user, anonymous_session_id=None, title="2", document_ids=[])

        with pytest.raises(Exception) as exc_info:
            cm.create_conversation(session, current_user=user, anonymous_session_id=None, title="3", document_ids=[])
        assert exc_info.value.status_code == 422


def test_create_conversation_validates_document_ids_are_visible_to_caller(settings):
    """document_ids must go through the same tenant-visibility check
    every other multi-document feature uses (app.rag.multi_doc.
    resolve_documents) — a conversation "focused" on a document the
    caller can't see must fail loudly, not silently scope to nothing."""
    with session_scope(settings) as session:
        other_user = User(email="other@example.com", hashed_password="x")
        session.add(other_user)
        session.flush()
        private_doc = Document(source_root="uploads", filename="private.pdf", file_type="pdf", owner_id=other_user.id)
        session.add(private_doc)
        session.flush()

        caller = User(email="caller@example.com", hashed_password="x")
        session.add(caller)
        session.flush()

        with pytest.raises(Exception) as exc_info:
            cm.create_conversation(
                session, current_user=caller, anonymous_session_id=None, title="t", document_ids=[private_doc.id]
            )
        assert exc_info.value.status_code == 404


def test_get_conversation_or_404_owner_can_access_own_conversation(settings):
    with session_scope(settings) as session:
        user = User(email="a@example.com", hashed_password="x")
        session.add(user)
        session.flush()
        convo = cm.create_conversation(session, current_user=user, anonymous_session_id=None, title="t", document_ids=[])
        convo_id = convo.id
        user_id = user.id

    with session_scope(settings) as session:
        user_obj = session.get(User, user_id)
        result = cm.get_conversation_or_404(session, convo_id, user_obj, None)
        assert result.id == convo_id


def test_get_conversation_or_404_different_owner_gets_404_not_403(settings):
    """The critical tenant-isolation regression test: another authenticated
    user, even with a perfectly valid session, must be refused access —
    and refused as 404 (not found) rather than 403 (forbidden), so the
    conversation's existence isn't confirmable to a user who shouldn't
    see it (matches app.api.source._get_document_or_404's convention)."""
    with session_scope(settings) as session:
        owner = User(email="owner@example.com", hashed_password="x")
        intruder = User(email="intruder@example.com", hashed_password="x")
        session.add_all([owner, intruder])
        session.flush()
        convo = cm.create_conversation(session, current_user=owner, anonymous_session_id=None, title="t", document_ids=[])
        convo_id = convo.id
        intruder_id = intruder.id

    with session_scope(settings) as session:
        intruder_obj = session.get(User, intruder_id)
        with pytest.raises(Exception) as exc_info:
            cm.get_conversation_or_404(session, convo_id, intruder_obj, None)
        assert exc_info.value.status_code == 404


def test_get_conversation_or_404_wrong_anonymous_session_gets_404(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(
            session, current_user=None, anonymous_session_id="session-a", title="t", document_ids=[]
        )
        convo_id = convo.id

    with session_scope(settings) as session:
        with pytest.raises(Exception) as exc_info:
            cm.get_conversation_or_404(session, convo_id, None, "session-b-different")
        assert exc_info.value.status_code == 404


def test_get_conversation_or_404_correct_anonymous_session_succeeds(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(
            session, current_user=None, anonymous_session_id="session-a", title="t", document_ids=[]
        )
        convo_id = convo.id

    with session_scope(settings) as session:
        result = cm.get_conversation_or_404(session, convo_id, None, "session-a")
        assert result.id == convo_id


def test_get_conversation_or_404_authenticated_user_cannot_use_anonymous_session_to_access(settings):
    """An authenticated caller is checked against owner_id ONLY — an
    anonymous_session_id they happen to pass is irrelevant/ignored for
    an authenticated request, matching resolve_identity's "authenticated
    always wins" rule."""
    with session_scope(settings) as session:
        convo = cm.create_conversation(
            session, current_user=None, anonymous_session_id="session-a", title="t", document_ids=[]
        )
        convo_id = convo.id
        user = User(email="a@example.com", hashed_password="x")
        session.add(user)
        session.flush()

    with session_scope(settings) as session:
        user_obj = session.query(User).filter(User.email == "a@example.com").one()
        with pytest.raises(Exception) as exc_info:
            cm.get_conversation_or_404(session, convo_id, user_obj, "session-a")
        assert exc_info.value.status_code == 404


def test_list_conversations_scoped_to_owner(settings):
    with session_scope(settings) as session:
        user_a = User(email="a@example.com", hashed_password="x")
        user_b = User(email="b@example.com", hashed_password="x")
        session.add_all([user_a, user_b])
        session.flush()
        cm.create_conversation(session, current_user=user_a, anonymous_session_id=None, title="a1", document_ids=[])
        cm.create_conversation(session, current_user=user_a, anonymous_session_id=None, title="a2", document_ids=[])
        cm.create_conversation(session, current_user=user_b, anonymous_session_id=None, title="b1", document_ids=[])
        user_a_id = user_a.id

    with session_scope(settings) as session:
        user_a_obj = session.get(User, user_a_id)
        results = cm.list_conversations(session, user_a_obj, None)
        assert {c.title for c in results} == {"a1", "a2"}


def test_list_conversations_anonymous_with_no_session_id_returns_empty():
    """No identity at all (no user, no anonymous session) legitimately
    has zero conversations — must not raise, must not return everyone's."""
    # No DB fixture needed — this path returns [] before ever querying.
    class _NoSession:
        def execute(self, *a, **k):
            raise AssertionError("must not query the DB when there is no identity at all")

    assert cm.list_conversations(_NoSession(), None, None) == []


def test_delete_conversation_cascades_to_messages(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        cm.add_message(session, convo, role="user", content="hi")
        cm.add_message(session, convo, role="assistant", content="hello")
        convo_id = convo.id

    with session_scope(settings) as session:
        convo = session.get(Conversation, convo_id)
        cm.delete_conversation(session, convo)

    with session_scope(settings) as session:
        assert session.get(Conversation, convo_id) is None
        remaining = session.query(ConversationMessage).filter(ConversationMessage.conversation_id == convo_id).all()
        assert remaining == []


# =============================================================================
# DB-backed — messages, rolling summary, turn context
# =============================================================================


def test_add_message_and_list_messages_round_trip(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        cm.add_message(session, convo, role="user", content="What is the price?")
        cm.add_message(
            session,
            convo,
            role="assistant",
            content="$5/month",
            citations=[{"chunk_id": "a::b::0"}],
            retrieved_chunk_ids=["a::b::0"],
            query_type="LOOKUP",
        )
        convo_id = convo.id

    with session_scope(settings) as session:
        convo = session.get(Conversation, convo_id)
        messages = cm.list_messages(session, convo)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].content == "$5/month"
        assert json.loads(messages[1].citations) == [{"chunk_id": "a::b::0"}]
        assert json.loads(messages[1].retrieved_chunk_ids) == ["a::b::0"]
        assert messages[1].query_type == "LOOKUP"


def test_maybe_refresh_rolling_summary_noop_when_within_window(settings):
    calls = []

    def fake_chat_json(*a, **k):
        calls.append(1)
        return {"summary": "should not be called"}

    import app.services.conversation_memory as cm_module

    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        for i in range(cm.RECENT_MESSAGE_WINDOW):  # exactly at the window, not over it
            cm.add_message(session, convo, role="user", content=f"msg {i}")
        convo_id = convo.id

    with session_scope(settings) as session:
        convo = session.get(Conversation, convo_id)
        cm_module.maybe_refresh_rolling_summary(session, convo, chat=None, model="fake-model")
        assert convo.rolling_summary == ""
    assert calls == []


def test_maybe_refresh_rolling_summary_triggers_and_updates_when_over_window(settings, monkeypatch):
    import app.services.conversation_memory as cm_module

    monkeypatch.setattr(cm_module, "chat_json", lambda *a, **k: {"summary": "User asked about pricing."})

    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        for i in range(cm.RECENT_MESSAGE_WINDOW + 4):  # comfortably over the window
            cm.add_message(session, convo, role="user", content=f"msg {i}")
        convo_id = convo.id

    with session_scope(settings) as session:
        convo = session.get(Conversation, convo_id)
        cm_module.maybe_refresh_rolling_summary(session, convo, chat=object(), model="fake-model")
        assert convo.rolling_summary == "User asked about pricing."


def test_maybe_refresh_rolling_summary_never_raises_on_chat_failure(settings, monkeypatch):
    import app.services.conversation_memory as cm_module

    def failing_chat_json(*a, **k):
        raise RuntimeError("provider outage")

    monkeypatch.setattr(cm_module, "chat_json", failing_chat_json)

    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        for i in range(cm.RECENT_MESSAGE_WINDOW + 2):
            cm.add_message(session, convo, role="user", content=f"msg {i}")
        convo_id = convo.id

    with session_scope(settings) as session:
        convo = session.get(Conversation, convo_id)
        # Must NOT raise — a failed summary refresh keeps the old
        # (empty) summary and the turn proceeds normally.
        cm_module.maybe_refresh_rolling_summary(session, convo, chat=object(), model="fake-model")
        assert convo.rolling_summary == ""


def test_build_turn_context_empty_conversation_returns_empty_prefix_and_no_source_files(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        prefix, source_files = cm.build_turn_context(session, convo)
        assert prefix == ""
        assert source_files is None


def test_build_turn_context_includes_rolling_summary_and_recent_messages(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        convo.rolling_summary = "Discussed pricing earlier."
        cm.add_message(session, convo, role="user", content="What about support?")
        cm.add_message(session, convo, role="assistant", content="24/7 support is included.")

        prefix, source_files = cm.build_turn_context(session, convo)
        assert "Discussed pricing earlier." in prefix
        assert "What about support?" in prefix
        assert "24/7 support is included." in prefix


def test_build_turn_context_only_includes_last_window_messages(settings):
    with session_scope(settings) as session:
        convo = cm.create_conversation(session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[])
        for i in range(cm.RECENT_MESSAGE_WINDOW + 4):
            cm.add_message(session, convo, role="user", content=f"message-number-{i}")

        prefix, _ = cm.build_turn_context(session, convo)
        # The oldest messages must NOT appear verbatim in the prefix —
        # only the last RECENT_MESSAGE_WINDOW do (this is the actual
        # "bounded, not unlimited history" guarantee item 3 asks about).
        assert "message-number-0" not in prefix
        assert f"message-number-{cm.RECENT_MESSAGE_WINDOW + 3}" in prefix


def test_build_turn_context_resolves_document_ids_to_source_files(settings):
    with session_scope(settings) as session:
        doc = Document(source_root="corpus", filename="policy.md", file_type="md")
        session.add(doc)
        session.flush()
        convo = cm.create_conversation(
            session, current_user=None, anonymous_session_id="s1", title="t", document_ids=[doc.id]
        )
        prefix, source_files = cm.build_turn_context(session, convo)
        assert source_files == frozenset({"policy.md"})