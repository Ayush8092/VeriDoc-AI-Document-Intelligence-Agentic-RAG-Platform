"""Regression test for a real cross-tenant data leak found and fixed in
`app.rag.summarization`: `_summarize_single_document` filtered chunks by
`source_file` alone, but filenames are NOT unique across users — two
different users each privately uploading a file named "report.pdf"
would have had their content silently merged the moment either one
summarized their own file. Every other Phase 6 reasoning module
(`app.rag.comparison`, `app.rag.extraction`) already scoped retrieval by
`allowed_owner_ids` in addition to the selected document(s);
`summarization.py` was the one place that didn't.

This test builds two documents with the SAME filename but different
owners directly in a local vector index (bypassing live LLM calls,
which this test doesn't need — it only exercises the retrieval/filter
step) and proves `_summarize_single_document`'s chunk fetch for one
owner never includes the other owner's chunks.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.chunking import Chunk
from app.clients import ensure_index, get_pinecone
from app.core.config import Settings
from app.rag.multi_doc import ResolvedDocument
from app.rag.summarization import _summarize_single_document
from app.vectorstore import upsert_chunks


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        vectorstore_backend="local",
        local_vector_index_path=str(tmp_path / "vector_index"),
        pinecone_namespace="ns",
        embedding_dimension=4,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _chunk(**overrides) -> Chunk:
    base = dict(
        chunk_id="uploads::report.pdf::body::0",
        source_root="uploads",
        source_file="report.pdf",
        document_title="Report",
        section="body",
        chunk_index=0,
        text="alice's private content",
        content_hash="hash1",
        block_type="text",
        page_start=1,
        page_end=1,
        source="native",
        confidence=1.0,
        table_rows=None,
        bbox=None,
        owner_id=None,
    )
    base.update(overrides)
    return Chunk(**base)


def _unit(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype="float32")
    return (arr / np.linalg.norm(arr)).tolist()


def test_summarize_single_document_never_leaks_another_owners_same_named_file(tmp_path, monkeypatch):
    """Alice and Bob each upload a file named 'report.pdf' with
    different content. Summarizing Alice's document must retrieve ONLY
    Alice's chunks — never Bob's, even though the filename collides.
    """
    settings = _settings(tmp_path)
    pc = get_pinecone(settings)
    index = ensure_index(pc, settings)

    alice_chunk = _chunk(
        chunk_id="uploads::report.pdf::body::0",
        text="alice's private salary figures",
        owner_id="1",
    )
    bob_chunk = _chunk(
        chunk_id="uploads::report.pdf::body::0::bob",
        text="bob's private medical notes",
        owner_id="2",
    )
    upsert_chunks(index, settings, [alice_chunk, bob_chunk], [_unit([1.0, 0.0, 0.0, 0.0])] * 2)

    alice_doc = ResolvedDocument(id=1, filename="report.pdf", title="Report", source_root="uploads", file_type="pdf")

    # Stub the LLM calls _summarize_single_document makes after
    # retrieval — this test only needs to prove the RETRIEVAL step is
    # tenant-scoped, not exercise real summarization quality.
    from app.rag import summarization as summarization_module

    monkeypatch.setattr(summarization_module, "_map_batch", lambda chat, model, mode, chunks: [
        {"text": c["text"], "chunk_ids": [c["chunk_id"]]} for c in chunks
    ])
    monkeypatch.setattr(
        summarization_module,
        "_reduce_points",
        lambda chat, model, points: ("stub summary", points),
    )

    result = _summarize_single_document(
        index, settings, chat=object(), doc=alice_doc, mode=summarization_module.SummaryMode.DOCUMENT,
        allowed_owner_ids=frozenset({"1"}),
    )

    all_texts = " ".join(p["text"] for p in result["points"])
    assert "alice" in all_texts
    assert "bob" not in all_texts, "cross-tenant leak: Bob's chunk was included in Alice's summary"
    assert result["chunk_count"] == 1


def test_summarize_single_document_requires_owner_scoping_argument():
    """Signature guard: allowed_owner_ids has no default — a caller
    cannot accidentally omit tenant scoping (the exact mistake that
    caused the original bug)."""
    import inspect

    sig = inspect.signature(_summarize_single_document)
    assert "allowed_owner_ids" in sig.parameters
    assert sig.parameters["allowed_owner_ids"].default is inspect.Parameter.empty
