"""Offline unit tests for app.rag.llm — no network call, no API key.

The real Groq client is replaced with a tiny fake mimicking
`groq.Groq().chat.completions.create(...)`'s shape.
"""

import json

import groq
import pytest

from app.rag.llm import (
    REFUSAL_TEXT,
    LLMProviderError,
    generate_answer,
    grade_chunks,
    rewrite_query,
    validate_citations,
)

MODEL = "llama-3.1-8b-instant"

CHUNKS = [
    {
        "chunk_id": "01_product_overview.md::free-trial::0",
        "source_file": "01_product_overview.md",
        "section": "Free Trial",
        "text": "Every new account starts with a 14-day free trial of the Pro plan.",
        "score": 0.83,
    }
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, reply="{}", raise_exc=None):
        self._reply = reply
        self._raise = raise_exc

    def create(self, **kwargs):
        if self._raise:
            raise self._raise
        return _FakeCompletion(self._reply)


class _FakeChat:
    def __init__(self, reply="{}", raise_exc=None):
        self.chat = type("_C", (), {})()
        self.chat.completions = _FakeCompletions(reply, raise_exc)


class _FakeProviderError(groq.APIError):
    def __init__(self, message):
        Exception.__init__(self, message)


def test_grade_chunks_maps_relevant_labels_back_to_chunk_ids():
    chat = _FakeChat(reply=json.dumps({"relevant_labels": ["CANDIDATE_1"], "reason": "on topic"}))
    result = grade_chunks(chat, MODEL, "How long is the free trial?", CHUNKS)
    assert result["sufficient"] is True
    assert result["relevant_chunk_ids"] == ["01_product_overview.md::free-trial::0"]


def test_grade_chunks_empty_relevant_labels_is_insufficient():
    chat = _FakeChat(reply=json.dumps({"relevant_labels": [], "reason": "nothing relevant"}))
    result = grade_chunks(chat, MODEL, "unrelated question", CHUNKS)
    assert result["sufficient"] is False
    assert result["relevant_chunk_ids"] == []


def test_grade_chunks_drops_hallucinated_labels():
    chat = _FakeChat(reply=json.dumps({"relevant_labels": ["CANDIDATE_99"], "reason": "x"}))
    result = grade_chunks(chat, MODEL, "q", CHUNKS)
    assert result["sufficient"] is False


def test_grade_chunks_no_chunks_short_circuits_without_a_call():
    chat = _FakeChat(reply="should never be read")
    result = grade_chunks(chat, MODEL, "q", [])
    assert result == {"sufficient": False, "relevant_chunk_ids": [], "reason": "no chunks retrieved"}


def test_grade_chunks_malformed_json_is_insufficient():
    chat = _FakeChat(reply="not json")
    result = grade_chunks(chat, MODEL, "q", CHUNKS)
    assert result["sufficient"] is False


def test_grade_chunks_provider_error_propagates():
    chat = _FakeChat(raise_exc=_FakeProviderError("rate limited"))
    with pytest.raises(LLMProviderError):
        grade_chunks(chat, MODEL, "q", CHUNKS)


def test_rewrite_query_dedupes_and_truncates_to_fanout():
    chat = _FakeChat(reply=json.dumps({"queries": ["a", "A", "b", "c", "d"]}))
    result = rewrite_query(chat, MODEL, "question", "previous", fanout=3)
    assert result == ["a", "b", "c"]


def test_rewrite_query_falls_back_to_question_on_empty_result():
    chat = _FakeChat(reply=json.dumps({"queries": []}))
    result = rewrite_query(chat, MODEL, "question", "previous", fanout=3)
    assert result == ["question"]


def test_generate_answer_maps_evidence_refs_to_chunk_ids():
    chat = _FakeChat(
        reply=json.dumps({"found": True, "answer": "14 days.", "evidence_refs": ["EVIDENCE_1"]})
    )
    result = generate_answer(chat, MODEL, "How long is the trial?", CHUNKS)
    assert result["found"] is True
    assert result["cited_chunk_ids"] == ["01_product_overview.md::free-trial::0"]


def test_generate_answer_defaults_to_refusal_on_unparseable_reply():
    chat = _FakeChat(reply="not json")
    result = generate_answer(chat, MODEL, "q", CHUNKS)
    assert result["found"] is False
    assert result["answer"] == REFUSAL_TEXT
    assert result["cited_chunk_ids"] == []


def test_generate_answer_drops_hallucinated_evidence_refs():
    chat = _FakeChat(reply=json.dumps({"found": True, "answer": "x", "evidence_refs": ["EVIDENCE_99"]}))
    result = generate_answer(chat, MODEL, "q", CHUNKS)
    assert result["cited_chunk_ids"] == []


def test_validate_citations_only_accepts_chunks_actually_retrieved():
    retrieved = [{**CHUNKS[0], "score": 0.9, "block_type": "text", "confidence": 1.0}]
    valid = validate_citations(["01_product_overview.md::free-trial::0", "made-up::id::0"], retrieved)
    assert len(valid) == 1
    assert valid[0]["chunk_id"] == "01_product_overview.md::free-trial::0"


def test_validate_citations_dedupes_repeated_ids():
    retrieved = [{**CHUNKS[0], "score": 0.9, "block_type": "text", "confidence": 1.0}]
    cid = "01_product_overview.md::free-trial::0"
    valid = validate_citations([cid, cid], retrieved)
    assert len(valid) == 1


def test_validate_citations_carries_table_metadata_through():
    table_chunk = {
        "chunk_id": "x::pricing::0",
        "source_file": "x.md",
        "section": "Pricing",
        "text": "| Plan | Price |\n| --- | --- |\n| Basic | $5 |",
        "score": 0.7,
        "block_type": "table",
        "page_start": 1,
        "page_end": 1,
        "source": "native",
        "confidence": 1.0,
        "table_rows": [["Plan", "Price"], ["Basic", "$5"]],
    }
    valid = validate_citations(["x::pricing::0"], [table_chunk])
    assert valid[0]["block_type"] == "table"
    assert valid[0]["table_rows"] == [["Plan", "Price"], ["Basic", "$5"]]
