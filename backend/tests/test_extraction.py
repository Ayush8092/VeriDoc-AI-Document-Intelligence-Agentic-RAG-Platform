"""Tests for app.rag.extraction (Phase 6 spec item 4).

Uses the REAL `LocalVectorIndex` (no live Pinecone needed — see
`tests/test_summarization.py`'s precedent) plus a fake chat client that
returns scripted JSON, so these exercise the actual retrieval ->
label-indirection -> schema-validation pipeline, not a mocked-out
shortcut.
"""

from __future__ import annotations

import json

import pytest

from app.rag.extraction import (
    MAX_SCHEMA_FIELDS,
    SUPPORTED_FIELD_TYPES,
    SchemaValidationError,
    extract_structured,
    validate_schema,
)
from app.rag.multi_doc import ResolvedDocument
from app.vectorstore_local import LocalVectorIndex


class _FakeSettings:
    embedding_dimension = 4
    pinecone_namespace = "test-ns"
    answer_model = "fake-model"
    top_k = 10
    bm25_top_k = 10
    hybrid_retrieval_enabled = False
    score_threshold = 0.0
    rrf_k = 60


class _FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _ScriptedChat:
    """Returns a fixed JSON payload regardless of prompt content --
    enough to drive extract_structured's real validation/citation
    -resolution logic without a real LLM."""

    def __init__(self, payload: dict):
        self._payload = payload

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return self._respond()

        self.chat = _Chat()

    def _respond(self):
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


def _upsert(index: LocalVectorIndex, chunk_id: str, source_file: str, text: str):
    index.upsert(
        [
            {
                "id": chunk_id,
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "source_root": "uploads",
                    "document_title": source_file,
                    "section": "body",
                    "chunk_index": 0,
                    "text": text,
                    "block_type": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "source": "native",
                    "confidence": 1.0,
                    "owner_id": "",
                },
            }
        ],
        namespace="test-ns",
    )


@pytest.fixture()
def settings():
    return _FakeSettings()


@pytest.fixture()
def embeddings():
    return _FakeEmbeddings()


# --- schema validation ------------------------------------------------


def test_validate_schema_empty_raises():
    with pytest.raises(SchemaValidationError):
        validate_schema({})


def test_validate_schema_too_many_fields_raises():
    schema = {f"field_{i}": "string" for i in range(MAX_SCHEMA_FIELDS + 1)}
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_schema(schema)
    assert str(MAX_SCHEMA_FIELDS) in str(exc_info.value)


def test_validate_schema_exactly_max_fields_is_allowed():
    schema = {f"field_{i}": "string" for i in range(MAX_SCHEMA_FIELDS)}
    validate_schema(schema)  # must not raise


def test_validate_schema_invalid_field_name_raises():
    with pytest.raises(SchemaValidationError):
        validate_schema({"not a valid identifier!": "string"})


def test_validate_schema_unsupported_type_raises():
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_schema({"amount": "currency"})
    assert "currency" in str(exc_info.value)


@pytest.mark.parametrize("field_type", sorted(SUPPORTED_FIELD_TYPES))
def test_validate_schema_accepts_every_supported_type(field_type):
    validate_schema({"f": field_type})  # must not raise


# --- extraction: each supported type, found case -----------------------


def test_extract_string_field_found(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "The party of the first part is Acme Corp.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat(
        {"fields": {"party_name": {"value": "Acme Corp", "status": "found", "confidence": 0.9, "supporting_evidence": ["EVIDENCE_1_1"]}}}
    )
    result = extract_structured(index, embeddings, chat, settings, {"party_name": "string"}, resolved, allowed_owner_ids=None)
    field = result["fields"]["party_name"]
    assert field["value"] == "Acme Corp"
    assert field["status"] == "found"
    assert field["confidence"] == 0.9
    assert len(field["citations"]) == 1
    assert field["citations"][0]["chunk_id"] == "c1"
    assert field["citations"][0]["document_id"] == 1


def test_extract_number_field_found(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "invoice.pdf", "Total amount due: $4200.")
    resolved = [ResolvedDocument(id=1, filename="invoice.pdf", title="Invoice", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat(
        {"fields": {"total_amount": {"value": 4200, "status": "found", "confidence": 0.95, "supporting_evidence": ["EVIDENCE_1_1"]}}}
    )
    result = extract_structured(index, embeddings, chat, settings, {"total_amount": "number"}, resolved, allowed_owner_ids=None)
    assert result["fields"]["total_amount"]["value"] == 4200


def test_extract_date_field_found(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "This agreement is effective as of 2026-01-15.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat(
        {"fields": {"effective_date": {"value": "2026-01-15", "status": "found", "confidence": 0.9, "supporting_evidence": ["EVIDENCE_1_1"]}}}
    )
    result = extract_structured(index, embeddings, chat, settings, {"effective_date": "date"}, resolved, allowed_owner_ids=None)
    assert result["fields"]["effective_date"]["value"] == "2026-01-15"


def test_extract_boolean_field_found(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "This contract includes an auto-renewal clause.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat(
        {"fields": {"has_auto_renewal": {"value": True, "status": "found", "confidence": 0.9, "supporting_evidence": ["EVIDENCE_1_1"]}}}
    )
    result = extract_structured(index, embeddings, chat, settings, {"has_auto_renewal": "boolean"}, resolved, allowed_owner_ids=None)
    assert result["fields"]["has_auto_renewal"]["value"] is True


def test_extract_list_string_field_found(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "The signatories are Alice Smith and Bob Jones.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat(
        {
            "fields": {
                "signatories": {
                    "value": ["Alice Smith", "Bob Jones"],
                    "status": "found",
                    "confidence": 0.9,
                    "supporting_evidence": ["EVIDENCE_1_1"],
                }
            }
        }
    )
    result = extract_structured(index, embeddings, chat, settings, {"signatories": "list[string]"}, resolved, allowed_owner_ids=None)
    assert result["fields"]["signatories"]["value"] == ["Alice Smith", "Bob Jones"]


# --- unsupported information -> null / not_found -----------------------


def test_extract_unsupported_information_returns_null_not_found(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "This document says nothing about termination.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat({"fields": {"termination_date": {"value": None, "status": "not_found", "confidence": 0.0, "supporting_evidence": []}}})
    result = extract_structured(index, embeddings, chat, settings, {"termination_date": "date"}, resolved, allowed_owner_ids=None)
    field = result["fields"]["termination_date"]
    assert field["value"] is None
    assert field["status"] == "not_found"
    assert field["citations"] == []


def test_extract_model_omits_field_still_returns_null_not_found(tmp_path, settings, embeddings):
    """Every requested schema field must appear in the response, even if
    the model's raw output silently omitted it (module docstring, step 3)."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "Some unrelated content.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat({"fields": {}})  # model returned nothing at all
    result = extract_structured(
        index, embeddings, chat, settings, {"party_name": "string", "amount": "number"}, resolved, allowed_owner_ids=None
    )
    assert set(result["fields"].keys()) == {"party_name", "amount"}
    assert result["fields"]["party_name"]["status"] == "not_found"
    assert result["fields"]["amount"]["status"] == "not_found"


def test_extract_found_status_with_no_resolvable_citation_fails_closed(tmp_path, settings, embeddings):
    """A 'found' status with no resolvable citation is inconsistent model
    output -- must fail closed to not_found rather than trust an
    unexplained value (module docstring / extract_structured's own
    fail-closed handling)."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "c1", "contract.pdf", "Some content.")
    resolved = [ResolvedDocument(id=1, filename="contract.pdf", title="Contract", source_root="uploads", file_type="pdf")]

    chat = _ScriptedChat(
        {"fields": {"party_name": {"value": "Acme Corp", "status": "found", "confidence": 0.9, "supporting_evidence": []}}}
    )
    result = extract_structured(index, embeddings, chat, settings, {"party_name": "string"}, resolved, allowed_owner_ids=None)
    field = result["fields"]["party_name"]
    assert field["status"] == "not_found"
    assert field["value"] is None


# --- document isolation (tenant safety, same regression class as ---------
# --- test_summarization.py's identically-named-file leak test) -----------


def test_extract_scoped_to_selected_document_ignores_other_owners_identically_named_file(tmp_path, settings, embeddings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    index.upsert(
        [
            {
                "id": "owner1::report.pdf::0",
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {
                    "chunk_id": "owner1::report.pdf::0",
                    "source_file": "report.pdf",
                    "source_root": "uploads",
                    "document_title": "report.pdf",
                    "section": "body",
                    "chunk_index": 0,
                    "text": "Owner 1's private contract value is $1000.",
                    "block_type": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "source": "native",
                    "confidence": 1.0,
                    "owner_id": "1",
                },
            },
            {
                "id": "owner2::report.pdf::0",
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {
                    "chunk_id": "owner2::report.pdf::0",
                    "source_file": "report.pdf",
                    "source_root": "uploads",
                    "document_title": "report.pdf",
                    "section": "body",
                    "chunk_index": 0,
                    "text": "Owner 2's private contract value is $9999.",
                    "block_type": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "source": "native",
                    "confidence": 1.0,
                    "owner_id": "2",
                },
            },
        ],
        namespace="test-ns",
    )
    doc = ResolvedDocument(id=101, filename="report.pdf", title="Report", source_root="uploads", file_type="pdf")

    chat = _ScriptedChat(
        {"fields": {"value": {"value": "$1000", "status": "found", "confidence": 0.9, "supporting_evidence": ["EVIDENCE_101_1"]}}}
    )
    result = extract_structured(
        index, embeddings, chat, settings, {"value": "string"}, [doc], allowed_owner_ids=frozenset({"1"})
    )
    citation = result["fields"]["value"]["citations"][0]
    assert citation["chunk_id"] == "owner1::report.pdf::0"