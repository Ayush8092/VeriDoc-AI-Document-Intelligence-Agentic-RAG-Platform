"""Tests for app.rag.summarization (Phase 6 spec item 5).

Most important test in this file: `test_summarize_single_document_never
_leaks_another_owners_identically_named_file` — a direct regression test
for a real, severe cross-tenant data leak found and fixed this session.
`_summarize_single_document` used to filter chunks by `source_file`
ALONE, which is only the original filename — not unique across users.
Two different users each privately uploading a file named e.g.
"report.pdf" would have had their chunks silently merged into one
"summary" the moment either user asked to summarize their own
`report.pdf`. Uses the REAL `LocalVectorIndex` (see
`tests/test_vectorstore_local.py`'s precedent for testing this backend
directly rather than mocking it), not a hand-rolled fake, so this
actually exercises the real metadata-filter code path.
"""

from __future__ import annotations

import pytest

from app.rag.multi_doc import ResolvedDocument
from app.rag.summarization import SummarizationError, SummaryMode, _summarize_single_document, summarize_documents
from app.vectorstore_local import LocalVectorIndex


class _FakeSettings:
    embedding_dimension = 4
    pinecone_namespace = "test-ns"
    answer_model = "fake-model"


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


class _FakeChat:
    """Records every prompt it's given and returns a trivial, valid
    map/reduce JSON response referencing whatever EVIDENCE_N labels
    appear in the prompt — good enough to drive
    `_summarize_single_document`'s real map-reduce control flow without
    needing a real LLM.
    """

    def __init__(self):
        self.seen_texts: list[str] = []

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return self._respond(kwargs)

        self.chat = _Chat()

    def _respond(self, kwargs):
        import json
        import re

        user_msg = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        self.seen_texts.append(user_msg)
        labels = re.findall(r"EVIDENCE_\d+", user_msg)

        if "POINTS" in "".join(m["content"] for m in kwargs["messages"] if m["role"] == "system").upper() or "points" in kwargs["messages"][0]["content"]:
            pass  # both map and reduce prompts ask for "points" — same response shape works for both

        payload = {
            "summary": "A trivial test summary.",
            "points": [{"text": f"Point referencing {lbl}", "evidence": [lbl]} for lbl in labels[:1]],
        }
        content = json.dumps(payload)

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


def test_summarize_single_document_never_leaks_another_owners_identically_named_file(tmp_path, settings):
    """The core regression test: two different owners each have a chunk
    from a file named "report.pdf". Summarizing owner 1's document must
    NEVER include owner 2's chunk, even though `source_file` alone would
    match both.
    """
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "owner1::report.pdf::0", "report.pdf", owner_id="1", text="Owner 1's private revenue figures.")
    _upsert(index, "owner2::report.pdf::0", "report.pdf", owner_id="2", text="Owner 2's private revenue figures.")

    chat = _FakeChat()
    doc = ResolvedDocument(id=101, filename="report.pdf", title="Report", source_root="uploads", file_type="pdf")

    result = _summarize_single_document(
        index, settings, chat, doc, SummaryMode.DOCUMENT, allowed_owner_ids=frozenset({"1"})
    )

    # The critical assertion: owner 2's content must never have reached
    # the LLM prompt at all — not just "wasn't cited", but never sent.
    all_prompts = " ".join(chat.seen_texts)
    assert "Owner 2's private revenue figures" not in all_prompts
    assert "Owner 1's private revenue figures" in all_prompts
    assert result["chunk_count"] == 1


def test_summarize_single_document_public_corpus_scoping_still_works(tmp_path, settings):
    """Sanity check the fix doesn't break the ordinary shared/public-corpus
    case: `owner_id=""` chunks are visible when `allowed_owner_ids`
    includes the public sentinel `""`."""
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "corpus::policy.md::0", "policy.md", owner_id="", text="Public policy content.")

    chat = _FakeChat()
    doc = ResolvedDocument(id=1, filename="policy.md", title="Policy", source_root="corpus", file_type="md")

    result = _summarize_single_document(
        index, settings, chat, doc, SummaryMode.DOCUMENT, allowed_owner_ids=frozenset({""})
    )

    assert result["chunk_count"] == 1
    assert "Public policy content" in " ".join(chat.seen_texts)


def test_summarize_single_document_no_owner_scope_means_unrestricted():
    """`allowed_owner_ids=None` must mean 'no restriction' (matching
    every other tenant-scoped call site's convention throughout this
    project), NOT 'restrict to nothing' — an anonymous/no-auth deployment
    (AUTH not required) must keep working exactly as before this fix."""
    from app.retrieval import _combine_filters

    assert _combine_filters(None, frozenset({"report.pdf"})) == {"source_file": {"$in": ["report.pdf"]}}


def test_summarize_documents_empty_selection_raises():
    settings = _FakeSettings()
    with pytest.raises(SummarizationError):
        summarize_documents(None, None, _FakeChat(), settings, [], SummaryMode.DOCUMENT, allowed_owner_ids=None)


def test_summarize_single_document_with_no_chunks_returns_empty_summary(tmp_path, settings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    chat = _FakeChat()
    doc = ResolvedDocument(id=1, filename="nonexistent.pdf", title="X", source_root="uploads", file_type="pdf")

    result = _summarize_single_document(index, settings, chat, doc, SummaryMode.DOCUMENT, allowed_owner_ids=None)

    assert result == {"summary": "", "points": [], "chunk_count": 0}
    assert chat.seen_texts == []  # no LLM call made for zero evidence


def test_summarize_documents_multi_document_combined_summary_present(tmp_path, settings):
    index = LocalVectorIndex(tmp_path / "idx.json", dimension=4)
    _upsert(index, "a::doc1::0", "doc1.pdf", owner_id="1", text="Doc 1 content about revenue.")
    _upsert(index, "a::doc2::0", "doc2.pdf", owner_id="1", text="Doc 2 content about expenses.")
    chat = _FakeChat()

    resolved = [
        ResolvedDocument(id=1, filename="doc1.pdf", title="Doc 1", source_root="uploads", file_type="pdf"),
        ResolvedDocument(id=2, filename="doc2.pdf", title="Doc 2", source_root="uploads", file_type="pdf"),
    ]

    result = summarize_documents(index, None, chat, settings, resolved, SummaryMode.DOCUMENT, frozenset({"1"}))

    assert set(result["per_document"].keys()) == {"1", "2"}
    assert len(result["documents"]) == 2