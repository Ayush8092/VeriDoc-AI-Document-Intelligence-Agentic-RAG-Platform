"""Unit tests for app.lexical_index — pure Python + tmp filesystem, no network."""

import json
from dataclasses import dataclass

from app.lexical_index import LexicalIndex, chunk_to_record, get_lexical_index, save_snapshot


@dataclass
class _FakeChunk:
    chunk_id: str
    source_root: str
    source_file: str
    document_title: str
    section: str
    chunk_index: int
    text: str
    content_hash: str
    block_type: str
    page_start: int | None
    page_end: int | None
    source: str
    confidence: float | None
    table_rows: list | None
    bbox: dict | None = None


def _chunk(chunk_id, text):
    return _FakeChunk(
        chunk_id=chunk_id,
        source_root="corpus",
        source_file="doc.md",
        document_title="Doc",
        section="Intro",
        chunk_index=0,
        text=text,
        content_hash="abc123",
        block_type="text",
        page_start=None,
        page_end=None,
        source="native",
        confidence=None,
        table_rows=None,
    )


def test_chunk_to_record_serializes_expected_fields():
    record = chunk_to_record(_chunk("doc::intro::0", "hello world"))
    assert record["chunk_id"] == "doc::intro::0"
    assert record["text"] == "hello world"
    assert record["block_type"] == "text"


def test_save_snapshot_writes_json_and_returns_count(tmp_path):
    chunks = [_chunk("a::x::0", "notice period is 60 days"), _chunk("b::x::0", "rent is $2000")]
    path = tmp_path / "lexical.json"

    count = save_snapshot(chunks, path)

    assert count == 2
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 2
    assert {r["chunk_id"] for r in data} == {"a::x::0", "b::x::0"}


def test_lexical_index_search_ranks_by_bm25(tmp_path):
    chunks = [
        _chunk("a::x::0", "completely unrelated content about fruit"),
        _chunk("b::x::0", "the notice period for termination is 60 days"),
    ]
    path = tmp_path / "lexical.json"
    save_snapshot(chunks, path)

    index = LexicalIndex.from_path(path)
    results = index.search("notice period termination", top_k=5)

    assert results[0]["chunk_id"] == "b::x::0"
    assert results[0]["_retriever"] == "bm25"
    assert "score" in results[0]


def _chunk_with_owner(chunk_id, text, owner_id):
    c = _chunk(chunk_id, text)
    # `_FakeChunk` (a plain, non-frozen dataclass) has no `owner_id`
    # field declared -- attach it dynamically; `chunk_to_record`'s
    # `getattr(chunk, field_name, None)` already tolerates a chunk
    # object that doesn't define every serialized field.
    c.owner_id = owner_id
    return c


def test_lexical_index_search_without_filter_sees_every_owner(tmp_path):
    chunks = [
        _chunk_with_owner("pub::x::0", "public notice period is 30 days", None),
        _chunk_with_owner("a::x::0", "alice notice period is 45 days", "1"),
        _chunk_with_owner("b::x::0", "bob notice period is 60 days", "2"),
    ]
    path = tmp_path / "lexical.json"
    save_snapshot(chunks, path)
    index = LexicalIndex.from_path(path)

    results = index.search("notice period", top_k=10)  # allowed_owner_ids=None (default)
    assert {r["chunk_id"] for r in results} == {"pub::x::0", "a::x::0", "b::x::0"}


def test_lexical_index_search_filters_by_allowed_owner_ids(tmp_path):
    """Phase 5 tenant isolation: a user's search must never surface
    another user's private chunks, only the shared/public ones plus
    their own."""
    chunks = [
        _chunk_with_owner("pub::x::0", "public notice period is 30 days", None),
        _chunk_with_owner("a::x::0", "alice notice period is 45 days", "1"),
        _chunk_with_owner("b::x::0", "bob notice period is 60 days", "2"),
    ]
    path = tmp_path / "lexical.json"
    save_snapshot(chunks, path)
    index = LexicalIndex.from_path(path)

    # User 1's allowed scope: public ("") + their own ("1") -- NOT "2".
    results = index.search("notice period", top_k=10, allowed_owner_ids=frozenset({"", "1"}))
    chunk_ids = {r["chunk_id"] for r in results}
    assert chunk_ids == {"pub::x::0", "a::x::0"}
    assert "b::x::0" not in chunk_ids


def test_lexical_index_search_anonymous_scope_sees_only_public(tmp_path):
    chunks = [
        _chunk_with_owner("pub::x::0", "public notice period is 30 days", None),
        _chunk_with_owner("a::x::0", "alice notice period is 45 days", "1"),
    ]
    path = tmp_path / "lexical.json"
    save_snapshot(chunks, path)
    index = LexicalIndex.from_path(path)

    results = index.search("notice period", top_k=10, allowed_owner_ids=frozenset({""}))
    assert {r["chunk_id"] for r in results} == {"pub::x::0"}


def test_lexical_index_search_filter_never_returns_fewer_than_topk_when_enough_allowed(tmp_path):
    """Filtering must happen BEFORE truncating to top_k, not after --
    otherwise a user could get fewer than top_k results even though
    enough allowed candidates exist (see LexicalIndex.search's
    docstring)."""
    chunks = [_chunk_with_owner(f"a::{i}::0", f"alice document number {i} about rent", "1") for i in range(5)]
    chunks += [_chunk_with_owner(f"b::{i}::0", f"bob document number {i} about rent", "2") for i in range(5)]
    path = tmp_path / "lexical.json"
    save_snapshot(chunks, path)
    index = LexicalIndex.from_path(path)

    results = index.search("rent document", top_k=3, allowed_owner_ids=frozenset({"1"}))
    assert len(results) == 3
    assert all(r["chunk_id"].startswith("a::") for r in results)


def test_lexical_index_missing_file_is_empty(tmp_path):
    index = LexicalIndex.from_path(tmp_path / "does_not_exist.json")
    assert len(index) == 0
    assert index.search("anything", top_k=5) == []


def test_lexical_index_corrupt_file_degrades_to_empty(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    index = LexicalIndex.from_path(path)
    assert len(index) == 0


def test_get_lexical_index_reflects_updated_snapshot_via_mtime_cache(tmp_path):
    path = tmp_path / "lexical.json"
    save_snapshot([_chunk("a::x::0", "first version content")], path)

    first = get_lexical_index(path, use_cache=True)
    assert len(first) == 1

    # Overwrite with different content — mtime changes, cache must refresh.
    import time

    time.sleep(0.01)
    save_snapshot(
        [_chunk("a::x::0", "first version content"), _chunk("b::x::0", "second chunk added")], path
    )

    second = get_lexical_index(path, use_cache=True)
    assert len(second) == 2