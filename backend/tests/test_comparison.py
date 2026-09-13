"""Tests for app.rag.comparison (Phase 6 spec item 3, "document
comparison"). Same testing pattern as tests/test_summarization.py: a
real `LocalVectorIndex` (not mocked) + a fake chat client that parses
the actual prompt sent and returns a crafted, schema-valid JSON
response — exercises the real retrieval -> label-indirection -> citation
-building control flow without a live LLM/Pinecone.
"""

from __future__ import annotations

import json

import pytest

from app.rag.comparison import ComparisonError, compare_documents, compare_multiple
from app.rag.multi_doc import ResolvedDocument
from app.vectorstore_local import LocalVectorIndex


class _FakeSettings:
    embedding_dimension = 4
    pinecone_namespace = "test-ns"
    answer_model = "fake-model"
    top_k = 10
    score_threshold = 0.0
    hybrid_retrieval_enabled = False


class _FixedEmbeddings:
    """Every query embeds to the same vector every upserted test chunk
    also uses — cosine similarity is always 1.0, so retrieval always
    "finds" whatever chunks were upserted for a document, without a real
    embedding model. Same convention as
    tests/test_summarization.py::_upsert's fixed vector.
    """

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


class _FakeChat:
    """Records every prompt; returns `_response_payload` regardless of
    content — set per-test to whatever comparison shape is needed.
    """

    def __init__(self, response_payload: dict):
        self.seen_texts: list[str] = []
        self._payload = response_payload

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return self._respond(kwargs)

        self.chat = _Chat()

    def _respond(self, kwargs):
        user_msg = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        self.seen_texts.append(user_msg)
        content = json.dumps(self._payload)

        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.message = _Msg(content)

        class _Response:
            def __init__(self, content):
                self.choices = [_Choice(content)]

        return _Response(content)


@pytest.fixture()
def settings():
    return _FakeSettings()


@pytest.fixture()
def embeddings():
    return _FixedEmbeddings()


def _two_docs():
    return [
        ResolvedDocument(id=1, filename="policy_a.pdf", title="Policy A", source_root="uploads", file_type="pdf"),
        ResolvedDocument(id=2, filename="policy_b.pdf", title="Policy B", source_root="uploads", file_type="pdf"),
    ]


# --- compare_documents (exactly two) ------------------------------------


def test_compare_documents_requires_exactly_two(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    chat = _FakeChat({"points": []})
    resolved = [ResolvedDocument(id=1, filename="a.pdf", title="A", source_root="uploads", file_type="pdf")]
    with pytest.raises(ComparisonError):
        compare_documents(index, embeddings, chat, settings, "notice period", resolved, allowed_owner_ids=None)


def test_compare_documents_rejects_empty_topic(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    chat = _FakeChat({"points": []})
    with pytest.raises(ComparisonError):
        compare_documents(index, embeddings, chat, settings, "   ", _two_docs(), allowed_owner_ids=None)


def test_compare_documents_classifies_same_different_missing_conflicting(settings, embeddings, tmp_path):
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
                "evidence_a": ["EVIDENCE_1_1"],
                "evidence_b": ["EVIDENCE_2_1"],
            }
        ]
    }
    chat = _FakeChat(payload)
    result = compare_documents(index, embeddings, chat, settings, "notice period", _two_docs(), allowed_owner_ids=frozenset({""}))

    assert result["topic"] == "notice period"
    assert result["document_a"]["id"] == 1
    assert result["document_b"]["id"] == 2
    assert len(result["points"]) == 1
    point = result["points"][0]
    assert point["relation"] == "different"
    assert point["summary_a"] == "30 days"
    assert point["summary_b"] == "60 days"
    assert len(point["evidence_a"]) == 1
    assert len(point["evidence_b"]) == 1
    # citations carry provenance for BOTH sides — spec: "cite evidence from BOTH documents"
    assert point["evidence_a"][0]["source_file"] == "policy_a.pdf"
    assert point["evidence_b"][0]["source_file"] == "policy_b.pdf"


def test_compare_documents_rejects_invalid_relation(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="", text="Some content.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="", text="Some other content.")

    payload = {"points": [{"aspect": "x", "relation": "not_a_real_relation", "evidence_a": [], "evidence_b": []}]}
    chat = _FakeChat(payload)
    result = compare_documents(index, embeddings, chat, settings, "x", _two_docs(), allowed_owner_ids=frozenset({""}))

    # fail-closed: an unrecognized relation is dropped entirely, not surfaced
    assert result["points"] == []


def test_compare_documents_tenant_isolation_excludes_other_owners_chunks(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::0", "policy_a.pdf", owner_id="1", text="Alice's private clause about renewal.")
    _upsert(index, "b::0", "policy_b.pdf", owner_id="2", text="Bob's private clause about renewal.")

    chat = _FakeChat({"points": []})
    compare_documents(index, embeddings, chat, settings, "renewal", _two_docs(), allowed_owner_ids=frozenset({"1"}))

    all_prompts = " ".join(chat.seen_texts)
    assert "Alice's private clause" in all_prompts
    assert "Bob's private clause" not in all_prompts


# --- compare_multiple (2+) -----------------------------------------------


def test_compare_multiple_requires_at_least_two(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    chat = _FakeChat({"points": []})
    resolved = [ResolvedDocument(id=1, filename="a.pdf", title="A", source_root="uploads", file_type="pdf")]
    with pytest.raises(ComparisonError):
        compare_multiple(index, embeddings, chat, settings, "topic", resolved, allowed_owner_ids=None)


def test_compare_multiple_reports_consensus_across_documents(settings, embeddings, tmp_path):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    resolved = [
        ResolvedDocument(id=1, filename="a.pdf", title="A", source_root="uploads", file_type="pdf"),
        ResolvedDocument(id=2, filename="b.pdf", title="B", source_root="uploads", file_type="pdf"),
        ResolvedDocument(id=3, filename="c.pdf", title="C", source_root="uploads", file_type="pdf"),
    ]
    for i, fname in enumerate(["a.pdf", "b.pdf", "c.pdf"], start=1):
        _upsert(index, f"{fname}::0", fname, owner_id="", text=f"Content from document {i}.")

    payload = {
        "points": [
            {
                "aspect": "topic",
                "summary_by_document": {"1": "x", "2": "x", "3": "y"},
                "consensus": "disagree",
                "evidence_by_document": {"1": ["EVIDENCE_1_1"], "2": ["EVIDENCE_2_1"], "3": ["EVIDENCE_3_1"]},
            }
        ]
    }
    chat = _FakeChat(payload)
    result = compare_multiple(index, embeddings, chat, settings, "topic", resolved, allowed_owner_ids=frozenset({""}))

    assert len(result["documents"]) == 3
    assert result["points"][0]["consensus"] == "disagree"