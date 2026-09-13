from types import SimpleNamespace

from app.chunking import Chunk
from app.core.config import Settings
from app.vectorstore import query, upsert_chunks


def _settings():
    return Settings(_env_file=None, pinecone_namespace="ns")


def _chunk(**overrides):
    base = dict(
        chunk_id="corpus::a.md::intro::0",
        source_root="corpus",
        source_file="a.md",
        document_title="A",
        section="intro",
        chunk_index=0,
        text="hello",
        content_hash="hash1",
        block_type="text",
        page_start=1,
        page_end=1,
        source="native",
        confidence=1.0,
        table_rows=None,
        bbox=None,
    )
    base.update(overrides)
    return Chunk(**base)


class _FakeIndex:
    def __init__(self):
        self.upserted = []

    def upsert(self, vectors, namespace):
        self.upserted.extend(vectors)


def test_upsert_chunks_embeds_table_metadata_as_json():
    index = _FakeIndex()
    settings = _settings()
    chunk = _chunk(block_type="table", table_rows=(("A", "B"), ("1", "2")), bbox={"x0": 1, "y0": 2, "x1": 3, "y1": 4})

    upsert_chunks(index, settings, [chunk], [[0.1, 0.2]])

    record = index.upserted[0]
    assert record["metadata"]["block_type"] == "table"
    assert '"A"' in record["metadata"]["table_json"]
    assert '"x0": 1' in record["metadata"]["bbox_json"]


def test_query_maps_metadata_back_to_dict_with_table_fields():
    match = {
        "id": "corpus::a.md::intro::0",
        "score": 0.91,
        "metadata": {
            "chunk_id": "corpus::a.md::intro::0",
            "source_root": "corpus",
            "source_file": "a.md",
            "document_title": "A",
            "section": "intro",
            "text": "hello",
            "block_type": "table",
            "page_start": 2,
            "page_end": 2,
            "source": "ocr",
            "confidence": 0.87,
            "table_json": '[["A", "B"], ["1", "2"]]',
            "bbox_json": '{"x0": 1, "y0": 2, "x1": 3, "y1": 4}',
        },
    }

    class _FakeResult:
        matches = [SimpleNamespace(**{**match, "metadata": match["metadata"]})]

    # exercise the dict-shaped branch (matches Pinecone's REST/dict response style)
    class _FakeIndexQuery:
        def query(self, vector, top_k, namespace, include_metadata):
            return {"matches": [match]}

    settings = _settings()
    results = query(_FakeIndexQuery(), settings, [0.1], top_k=5)

    assert len(results) == 1
    r = results[0]
    assert r["block_type"] == "table"
    assert r["page_start"] == 2
    assert r["source"] == "ocr"
    assert r["confidence"] == 0.87
    assert r["table_rows"] == [["A", "B"], ["1", "2"]]
    assert r["bbox"] == {"x0": 1, "y0": 2, "x1": 3, "y1": 4}


def test_query_defaults_page_and_table_fields_when_absent():
    match = {
        "id": "corpus::a.md::intro::0",
        "score": 0.5,
        "metadata": {
            "chunk_id": "corpus::a.md::intro::0",
            "source_file": "a.md",
            "section": "intro",
            "text": "hello",
        },
    }

    class _FakeIndexQuery:
        def query(self, vector, top_k, namespace, include_metadata):
            return {"matches": [match]}

    settings = _settings()
    results = query(_FakeIndexQuery(), settings, [0.1], top_k=5)

    assert results[0]["page_start"] is None
    assert results[0]["table_rows"] is None
    assert results[0]["block_type"] == "text"
    assert results[0]["source"] == "native"
