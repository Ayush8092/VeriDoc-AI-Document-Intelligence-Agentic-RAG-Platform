"""Tests for `app.vectorstore_local` — the free/$0 vector backend
(`VECTORSTORE_BACKEND=local`, the default; see `app/clients.py::get_pinecone`).

Phase 5 completion pass (round 5), item 13: this module had ZERO test
coverage before this file — `tests/test_vectorstore.py` only exercises
`app/vectorstore.py`'s backend-agnostic helpers (`_match_to_result`,
`upsert_chunks`, etc.) against a hand-rolled fake, never the real
`LocalVectorIndex`/`LocalPineconeClient` duck-types this project actually
ships and defaults to.

Every assertion below was manually verified against the real
implementation (not just written speculatively) before being committed
to this file — see the Phase 5 completion report for the "written AND
manually smoke-tested outside pytest, since pytest itself isn't
installed in the authoring sandbox" distinction from "run under pytest
in this environment", which is honestly NOT the same claim.
"""

from __future__ import annotations

import json
import threading

import pytest

from app.vectorstore_local import LocalPineconeClient, LocalVectorIndex, _cosine_similarity, _matches_filter


# ---------------------------------------------------------------------
# LocalVectorIndex — upsert / idempotency
# ---------------------------------------------------------------------


def test_upsert_creates_new_records(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=3)
    idx.upsert([{"id": "a", "values": [1, 0, 0], "metadata": {"text": "hello"}}], namespace="ns")
    stats = idx.describe_index_stats()
    assert stats["namespaces"]["ns"]["vector_count"] == 1


def test_upsert_same_id_twice_is_idempotent_not_duplicated(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=3)
    idx.upsert([{"id": "a", "values": [1, 0, 0], "metadata": {"text": "v1"}}], namespace="ns")
    idx.upsert([{"id": "a", "values": [1, 0, 0], "metadata": {"text": "v2"}}], namespace="ns")
    stats = idx.describe_index_stats()
    assert stats["namespaces"]["ns"]["vector_count"] == 1, "re-upsert of the same id must overwrite, not duplicate"
    fetched = idx.fetch(["a"], namespace="ns")
    assert fetched["vectors"]["a"]["metadata"]["text"] == "v2", "overwrite must actually replace the metadata"


def test_upsert_returns_pinecone_shaped_response(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=3)
    result = idx.upsert([{"id": "a", "values": [1, 0, 0], "metadata": {}}], namespace="ns")
    assert result == {"upserted_count": 1}


# ---------------------------------------------------------------------
# LocalVectorIndex — query (ranking, top_k, metadata, filtering)
# ---------------------------------------------------------------------


def test_query_ranks_by_cosine_similarity_best_first(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=3)
    idx.upsert(
        [
            {"id": "exact", "values": [1, 0, 0], "metadata": {}},
            {"id": "orthogonal", "values": [0, 1, 0], "metadata": {}},
            {"id": "opposite", "values": [-1, 0, 0], "metadata": {}},
        ],
        namespace="ns",
    )
    result = idx.query(vector=[1, 0, 0], top_k=3, namespace="ns")
    ids_in_order = [m["id"] for m in result["matches"]]
    assert ids_in_order == ["exact", "orthogonal", "opposite"]
    assert result["matches"][0]["score"] == pytest.approx(1.0)
    assert result["matches"][2]["score"] == pytest.approx(-1.0)


def test_query_cosine_is_scale_invariant_not_a_bare_dot_product(tmp_path):
    """A bare dot product would rank a longer (non-unit) vector higher
    just for having larger magnitude — cosine similarity must not."""
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert(
        [
            {"id": "unit", "values": [1, 0], "metadata": {}},
            {"id": "scaled_same_direction", "values": [100, 0], "metadata": {}},
        ],
        namespace="ns",
    )
    result = idx.query(vector=[1, 0], top_k=2, namespace="ns")
    assert result["matches"][0]["score"] == pytest.approx(result["matches"][1]["score"])


def test_query_respects_top_k(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert([{"id": str(i), "values": [1, i], "metadata": {}} for i in range(10)], namespace="ns")
    result = idx.query(vector=[1, 0], top_k=3, namespace="ns")
    assert len(result["matches"]) == 3


def test_query_include_metadata_false_omits_metadata(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert([{"id": "a", "values": [1, 0], "metadata": {"text": "secret"}}], namespace="ns")
    result = idx.query(vector=[1, 0], top_k=1, namespace="ns", include_metadata=False)
    assert result["matches"][0]["metadata"] == {}


def test_query_empty_namespace_returns_no_matches(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    result = idx.query(vector=[1, 0], top_k=5, namespace="never-upserted")
    assert result["matches"] == []


def test_query_with_metadata_filter_narrows_results(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert(
        [
            {"id": "a", "values": [1, 0], "metadata": {"block_type": "table"}},
            {"id": "b", "values": [1, 0], "metadata": {"block_type": "text"}},
        ],
        namespace="ns",
    )
    result = idx.query(vector=[1, 0], top_k=5, namespace="ns", filter={"block_type": {"$eq": "table"}})
    assert [m["id"] for m in result["matches"]] == ["a"]


# ---------------------------------------------------------------------
# Tenant isolation (Phase 5's core safety requirement, applied to the
# free/local backend specifically — see app/security/auth.py's
# `allowed_owner_ids` for the convention these metadata values follow).
# ---------------------------------------------------------------------


def test_tenant_filter_by_owner_id_via_in_operator(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert(
        [
            {"id": "public", "values": [1, 0], "metadata": {"owner_id": ""}},
            {"id": "mine", "values": [1, 0], "metadata": {"owner_id": "42"}},
            {"id": "someone_elses", "values": [1, 0], "metadata": {"owner_id": "99"}},
        ],
        namespace="ns",
    )
    # Mirrors exactly how app/vectorstore.py / app/retrieval.py build a
    # tenant-scoped filter: allowed = {"", str(current_user.id)}.
    result = idx.query(vector=[1, 0], top_k=10, namespace="ns", filter={"owner_id": {"$in": ["", "42"]}})
    ids = {m["id"] for m in result["matches"]}
    assert ids == {"public", "mine"}
    assert "someone_elses" not in ids


def test_tenant_filter_never_leaks_across_owners_at_full_scan(tmp_path):
    """A slightly larger version of the isolation test above — enough
    records that a filtering bug returning even one wrong owner's
    record among many is still caught, not hidden by a too-small sample.
    """
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    records = [{"id": f"user-a-{i}", "values": [1, 0], "metadata": {"owner_id": "a"}} for i in range(20)]
    records += [{"id": f"user-b-{i}", "values": [1, 0], "metadata": {"owner_id": "b"}} for i in range(20)]
    idx.upsert(records, namespace="ns")

    result = idx.query(vector=[1, 0], top_k=100, namespace="ns", filter={"owner_id": {"$eq": "a"}})
    assert len(result["matches"]) == 20
    assert all(m["id"].startswith("user-a-") for m in result["matches"])


# ---------------------------------------------------------------------
# fetch / delete
# ---------------------------------------------------------------------


def test_fetch_returns_only_requested_existing_ids(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert([{"id": "a", "values": [1, 0], "metadata": {"x": 1}}], namespace="ns")
    result = idx.fetch(["a", "does-not-exist"], namespace="ns")
    assert set(result["vectors"].keys()) == {"a"}
    assert result["vectors"]["a"]["metadata"] == {"x": 1}


def test_delete_by_ids_removes_only_those_ids(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert(
        [{"id": "a", "values": [1, 0], "metadata": {}}, {"id": "b", "values": [1, 0], "metadata": {}}],
        namespace="ns",
    )
    idx.delete(ids=["a"], namespace="ns")
    stats = idx.describe_index_stats()
    assert stats["namespaces"]["ns"]["vector_count"] == 1
    assert idx.fetch(["b"], namespace="ns")["vectors"]


def test_delete_all_clears_only_the_given_namespace(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert([{"id": "a", "values": [1, 0], "metadata": {}}], namespace="ns1")
    idx.upsert([{"id": "b", "values": [1, 0], "metadata": {}}], namespace="ns2")
    idx.delete(namespace="ns1", delete_all=True)
    assert idx.describe_index_stats()["namespaces"]["ns1"]["vector_count"] == 0
    assert idx.describe_index_stats()["namespaces"]["ns2"]["vector_count"] == 1


def test_delete_is_idempotent(tmp_path):
    idx = LocalVectorIndex(tmp_path / "idx.json", dimension=2)
    idx.upsert([{"id": "a", "values": [1, 0], "metadata": {}}], namespace="ns")
    idx.delete(ids=["a"], namespace="ns")
    idx.delete(ids=["a"], namespace="ns")  # must not raise the second time
    assert idx.describe_index_stats()["namespaces"]["ns"]["vector_count"] == 0


# ---------------------------------------------------------------------
# Persistence / reload — the whole reason this is a *file-backed* index
# ---------------------------------------------------------------------


def test_index_persists_across_process_restart_simulation(tmp_path):
    """A fresh `LocalVectorIndex` instance pointed at the same path (as
    happens on every real process restart — nothing is held in memory
    between requests) must see exactly what a prior instance wrote."""
    path = tmp_path / "idx.json"
    first = LocalVectorIndex(path, dimension=3)
    first.upsert([{"id": "a", "values": [1, 2, 3], "metadata": {"text": "persisted"}}], namespace="ns")

    second = LocalVectorIndex(path, dimension=3)  # simulates a new process
    result = second.fetch(["a"], namespace="ns")
    assert result["vectors"]["a"]["values"] == [1, 2, 3]
    assert result["vectors"]["a"]["metadata"]["text"] == "persisted"


def test_missing_file_is_treated_as_empty_index_not_an_error(tmp_path):
    idx = LocalVectorIndex(tmp_path / "does-not-exist-yet.json", dimension=3)
    assert idx.describe_index_stats() == {"namespaces": {}}
    assert idx.query(vector=[1, 0, 0], top_k=5, namespace="ns")["matches"] == []


def test_corrupt_file_degrades_to_empty_index_rather_than_crashing(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json at all")
    idx = LocalVectorIndex(path, dimension=3)
    assert idx.describe_index_stats() == {"namespaces": {}}


def test_write_is_atomic_no_temp_file_left_behind_on_success(tmp_path):
    path = tmp_path / "idx.json"
    idx = LocalVectorIndex(path, dimension=2)
    idx.upsert([{"id": "a", "values": [1, 0], "metadata": {}}], namespace="ns")
    leftover_temp_files = list(tmp_path.glob(".idx.json.tmp-*"))
    assert leftover_temp_files == [], "a successful write must not leave its temp file behind"
    # And the real file must be valid, complete JSON — not truncated.
    json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# Concurrent reads — many simultaneous queries against the same file
# must never raise or see a torn/partial read.
# ---------------------------------------------------------------------


def test_concurrent_reads_do_not_raise_or_see_partial_data(tmp_path):
    path = tmp_path / "idx.json"
    seed = LocalVectorIndex(path, dimension=2)
    seed.upsert([{"id": f"r{i}", "values": [1, 0], "metadata": {}} for i in range(50)], namespace="ns")

    errors: list[Exception] = []
    results: list[int] = []

    def read_once():
        try:
            reader = LocalVectorIndex(path, dimension=2)
            matches = reader.query(vector=[1, 0], top_k=100, namespace="ns")["matches"]
            results.append(len(matches))
        except Exception as exc:  # noqa: BLE001 - captured for the assertion below, not swallowed silently
            errors.append(exc)

    threads = [threading.Thread(target=read_once) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent reads raised: {errors}"
    assert all(count == 50 for count in results), "every concurrent reader must see the complete, consistent data"


# ---------------------------------------------------------------------
# LocalPineconeClient — the top-level duck-typed `pinecone.Pinecone` client
# ---------------------------------------------------------------------


def test_client_index_reads_dimension_from_existing_file(tmp_path, monkeypatch):
    settings = _FakeSettings(str(tmp_path))
    client = LocalPineconeClient(settings)
    client.create_index("my-index", dimension=5)
    idx = client.Index("my-index")
    assert idx.dimension == 5


def test_client_create_index_is_idempotent(tmp_path):
    settings = _FakeSettings(str(tmp_path))
    client = LocalPineconeClient(settings)
    client.create_index("my-index", dimension=5)
    client.Index("my-index").upsert([{"id": "a", "values": [1] * 5, "metadata": {}}], namespace="ns")
    client.create_index("my-index", dimension=5)  # must not wipe existing data
    assert client.Index("my-index").describe_index_stats()["namespaces"]["ns"]["vector_count"] == 1


def test_client_list_indexes_reflects_created_indexes(tmp_path):
    settings = _FakeSettings(str(tmp_path))
    client = LocalPineconeClient(settings)
    client.create_index("idx-a", dimension=3)
    client.create_index("idx-b", dimension=3)
    names = {i["name"] for i in client.list_indexes()}
    assert names == {"idx-a", "idx-b"}


def test_client_describe_index_reports_ready_true(tmp_path):
    settings = _FakeSettings(str(tmp_path))
    client = LocalPineconeClient(settings)
    client.create_index("idx-a", dimension=3)
    desc = client.describe_index("idx-a")
    assert desc.dimension == 3
    assert desc.status["ready"] is True


def test_client_persists_across_new_client_instances(tmp_path):
    """Two `LocalPineconeClient` instances (as `get_pinecone()` would
    construct on two different requests) sharing the same
    `local_vector_index_path` must see the same on-disk data."""
    settings = _FakeSettings(str(tmp_path))
    LocalPineconeClient(settings).create_index("idx-a", dimension=2)
    LocalPineconeClient(settings).Index("idx-a").upsert([{"id": "a", "values": [1, 0], "metadata": {}}], namespace="ns")

    fresh_client = LocalPineconeClient(settings)
    assert fresh_client.Index("idx-a").describe_index_stats()["namespaces"]["ns"]["vector_count"] == 1


class _FakeSettings:
    def __init__(self, local_vector_index_path: str):
        self.local_vector_index_path = local_vector_index_path


# ---------------------------------------------------------------------
# Filter operator semantics (unit-level, isolated from LocalVectorIndex)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metadata", "filt", "expected"),
    [
        ({"page": 3}, {"page": {"$eq": 3}}, True),
        ({"page": 3}, {"page": {"$eq": 4}}, False),
        ({"page": 3}, {"page": {"$lte": 3}}, True),
        ({"page": 3}, {"page": {"$lte": 2}}, False),
        ({"page": 3}, {"page": {"$gte": 3}}, True),
        ({"page": 3}, {"page": {"$gte": 4}}, False),
        ({"page": 3}, {"page": {"$lt": 4}}, True),
        ({"page": 3}, {"page": {"$lt": 3}}, False),
        ({"page": 3}, {"page": {"$gt": 2}}, True),
        ({"page": 3}, {"page": {"$gt": 3}}, False),
        ({"tag": "a"}, {"tag": {"$in": ["a", "b"]}}, True),
        ({"tag": "c"}, {"tag": {"$in": ["a", "b"]}}, False),
        ({"tag": "a"}, {"tag": "a"}, True),  # bare-value shorthand == $eq
        ({}, {"page": {"$lte": 3}}, False),  # missing field never matches a comparison
    ],
)
def test_matches_filter_operator_semantics(metadata, filt, expected):
    assert _matches_filter(metadata, filt) is expected


def test_matches_filter_multiple_keys_are_and_ed():
    metadata = {"owner_id": "42", "block_type": "table"}
    assert _matches_filter(metadata, {"owner_id": {"$eq": "42"}, "block_type": {"$eq": "table"}}) is True
    assert _matches_filter(metadata, {"owner_id": {"$eq": "42"}, "block_type": {"$eq": "figure"}}) is False


def test_matches_filter_unknown_operator_raises_rather_than_silently_matching():
    """Tenant-isolation filters flow through this exact function — a
    typo'd/unsupported operator must be a loud bug, never a silent
    'matches everything' that could leak data across tenants."""
    with pytest.raises(ValueError):
        _matches_filter({"owner_id": "42"}, {"owner_id": {"$unsupported_op": "42"}})


def test_cosine_similarity_handles_zero_vector_without_dividing_by_zero():
    assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
