"""Local, file-persisted vector backend — the free/$0 alternative to
Pinecone (VECTORSTORE_BACKEND=local, the default — see
app.core.config.Settings.vectorstore_backend).

Design: duck-type the exact surface of the real Pinecone SDK that this
codebase actually calls, so NOT ONE LINE of app/vectorstore.py,
app/clients.py's `ensure_index`/`get_index`, app/rag/graph.py,
app/services/ingestion_service.py, or app/api/source.py needs to know
which backend is active. `app.clients.get_pinecone()` is the ONLY
dispatch point (see its docstring) — everything downstream just calls
`.Index(name)`, `.query(...)`, `.upsert(...)`, `.fetch(...)`,
`.delete(...)`, `.describe_index_stats()`, `.list(...)`,
`.list_indexes()`, `.describe_index(...)`, `.create_index(...)` exactly
as it always has, and gets back objects `app.vectorstore._match_to_result`
already knows how to read (it already handles both dict-style and
attribute-style access -- see that function).

Storage format: one JSON file per index, at
`{settings.local_vector_index_path}/{settings.pinecone_index_name}.json`:

    {
      "dimension": 3072,
      "namespaces": {
        "veridoc-corpus": [
          {"id": "...", "values": [...], "metadata": {...}},
          ...
        ]
      }
    }

Every mutating call (upsert/delete/create) does a full read-modify-
atomic-write of this one file; every read call (query/fetch/stats) does
a full read. This is the SAME "cheap enough to just reload" philosophy
`app/lexical_index.py` already uses for BM25 -- correct and simple is
worth more than clever at this project's corpus size, and it means
there is no separate cache-invalidation story to get wrong. Cosine
similarity is computed as a genuine `dot(a,b) / (|a| * |b|)`, not a bare
dot product, so it behaves like Pinecone's `metric="cosine"` regardless
of whether the caller's vectors happen to already be unit-normalized
(`app/clients.py`'s `Embedder` does normalize, but this backend doesn't
rely on that).

Atomic replacement (write to a same-directory temp file, fsync, then
`os.replace`) is the same pattern -- and the same reasoning -- as
`app/lexical_index.py::save_snapshot`'s fix: a concurrent reader (every
`/ask` request) must never be able to observe a half-written file.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filter(metadata: dict, filter: dict) -> bool:
    """Evaluate one Pinecone-style metadata filter against one record's
    metadata. Supports exactly the operators this codebase's own filters
    actually use (`app/vectorstore.py`, `app/api/source.py`):
    `$eq`, `$in`, `$lte`, `$gte`, `$gt`, `$lt`. Every filter key is an
    implicit AND (matches Pinecone's own semantics for a flat filter
    dict). Anything else raises `ValueError` rather than silently
    matching everything or nothing -- a filter operator this backend
    doesn't understand is a bug to surface loudly, not a query to
    silently return the wrong results for (tenant-isolation filters flow
    through this exact function; silently ignoring an operator here
    could silently leak data across tenants).
    """
    for field, condition in filter.items():
        value = metadata.get(field)
        if not isinstance(condition, dict):
            # Bare-value shorthand (`{"field": "x"}` meaning `$eq`) --
            # Pinecone supports this too; not currently used anywhere in
            # this codebase's own filters, but cheap and harmless to
            # support for a more faithful duck-type.
            if value != condition:
                return False
            continue
        for op, expected in condition.items():
            if op == "$eq":
                if value != expected:
                    return False
            elif op == "$in":
                if value not in expected:
                    return False
            elif op == "$lte":
                if value is None or value > expected:
                    return False
            elif op == "$gte":
                if value is None or value < expected:
                    return False
            elif op == "$lt":
                if value is None or value >= expected:
                    return False
            elif op == "$gt":
                if value is None or value <= expected:
                    return False
            else:
                raise ValueError(f"unsupported filter operator: {op!r}")
    return True


class LocalVectorIndex:
    """Duck-types a Pinecone `Index` handle, backed by one JSON file.

    Constructed directly (`LocalVectorIndex(path, dimension=...)`) by
    `LocalPineconeClient.Index`/`.create_index`, or indirectly via
    `app.clients.ensure_index`/`get_index` -- callers never need to know
    the difference between this and a real `pinecone.Index`.
    """

    def __init__(self, path: Path | str, dimension: int):
        self.path = Path(path)
        self.dimension = dimension

    # -- internal file I/O -------------------------------------------------

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"dimension": self.dimension, "namespaces": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt/partially-written file (e.g. the process was
            # killed mid-write before atomic replacement existed, or the
            # disk filled up) degrades to an empty index rather than
            # crashing every query/upsert -- the same "fail open to
            # empty, never fail closed with a crash" choice
            # `app/lexical_index.py`'s `LexicalIndex._load` already
            # makes for the exact same failure mode.
            return {"dimension": self.dimension, "namespaces": {}}
        if not isinstance(data, dict) or "namespaces" not in data:
            return {"dimension": self.dimension, "namespaces": {}}
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data)
        fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), prefix=f".{self.path.name}.tmp-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)  # atomic on POSIX and Windows
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    # -- Pinecone-Index-shaped API ------------------------------------------

    def upsert(self, vectors: list[dict], namespace: str) -> dict:
        data = self._load()
        ns = data["namespaces"].setdefault(namespace, [])
        by_id = {rec["id"]: i for i, rec in enumerate(ns)}
        for record in vectors:
            if record["id"] in by_id:
                ns[by_id[record["id"]]] = record  # overwrite -> idempotent re-ingestion
            else:
                by_id[record["id"]] = len(ns)
                ns.append(record)
        self._save(data)
        return {"upserted_count": len(vectors)}

    def query(
        self,
        vector: list[float],
        top_k: int,
        namespace: str,
        include_metadata: bool = True,
        filter: dict | None = None,
    ) -> dict:
        data = self._load()
        records = data["namespaces"].get(namespace, [])
        if filter:
            records = [r for r in records if _matches_filter(r.get("metadata", {}), filter)]
        scored = [(r, _cosine_similarity(vector, r["values"])) for r in records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:top_k]
        return {
            "matches": [
                {"id": r["id"], "score": score, "metadata": r.get("metadata", {}) if include_metadata else {}}
                for r, score in top
            ]
        }

    def fetch(self, ids: list[str], namespace: str) -> dict:
        data = self._load()
        records = {r["id"]: r for r in data["namespaces"].get(namespace, [])}
        return {
            "vectors": {
                chunk_id: {
                    "id": chunk_id,
                    "values": records[chunk_id]["values"],
                    "metadata": records[chunk_id].get("metadata", {}),
                }
                for chunk_id in ids
                if chunk_id in records
            }
        }

    def delete(self, ids: list[str] | None = None, namespace: str = "", delete_all: bool = False) -> dict:
        data = self._load()
        if delete_all:
            data["namespaces"][namespace] = []
        else:
            id_set = set(ids or [])
            ns = data["namespaces"].get(namespace, [])
            data["namespaces"][namespace] = [r for r in ns if r["id"] not in id_set]
        self._save(data)
        return {}

    def describe_index_stats(self) -> dict:
        data = self._load()
        return {"namespaces": {ns: {"vector_count": len(records)} for ns, records in data["namespaces"].items()}}

    def list(self, namespace: str):
        """Generator yielding batches of IDs, matching the real
        `pinecone.Index.list()` iterator shape (`app.vectorstore.list_all_ids`
        iterates `for batch in index.list(...): ids.update(batch)`).
        Yields the whole namespace as one batch -- this backend's corpus
        sizes never warrant real pagination.
        """
        data = self._load()
        ids = [r["id"] for r in data["namespaces"].get(namespace, [])]
        if ids:
            yield ids


class LocalPineconeClient:
    """Duck-types the top-level `pinecone.Pinecone` client object --
    `.Index(name)`, `.list_indexes()`, `.describe_index(name)`,
    `.create_index(...)`. Constructed fresh per `get_pinecone()` call
    (see that function's docstring); every instance reads/writes the
    SAME on-disk directory for a given `settings.local_vector_index_path`,
    so persistence survives across instances (process restarts) -- see
    `tests/test_vectorstore_local.py::test_index_persists_across_process_restart_simulation`.
    """

    def __init__(self, settings):
        self._dir = Path(settings.local_vector_index_path)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self, name: str) -> Path:
        return self._dir / f"{name}.json"

    def Index(self, name: str) -> LocalVectorIndex:
        path = self._index_path(name)
        dimension = self._read_dimension(path)
        return LocalVectorIndex(path, dimension=dimension)

    def _read_dimension(self, path: Path) -> int:
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("dimension", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0

    def list_indexes(self) -> list[dict]:
        if not self._dir.is_dir():
            return []
        return [{"name": p.stem} for p in sorted(self._dir.glob("*.json"))]

    def describe_index(self, name: str) -> SimpleNamespace:
        path = self._index_path(name)
        dimension = self._read_dimension(path)
        # `app.clients.ensure_index` reads `.status["ready"]` (attribute
        # THEN dict-subscript) and `_validate_index_dimension` reads
        # `.dimension` (plain attribute, no dict fallback) -- a local
        # index is always immediately "ready" (no async provisioning
        # like a real Pinecone serverless index), so this never needs a
        # poll loop in practice, but still matches the shape exactly.
        return SimpleNamespace(dimension=dimension, status={"ready": True})

    def create_index(self, name: str, dimension: int, metric: str = "cosine", spec=None) -> None:
        # `metric`/`spec` accepted-and-ignored: real Pinecone needs them
        # for cloud provisioning (region, serverless spec, similarity
        # metric); a local JSON file has no equivalent concept, and
        # duck-typing `app.clients.ensure_index`'s exact call signature
        # (`pc.create_index(name=..., dimension=..., metric=..., spec=ServerlessSpec(...))`)
        # is the entire point (module docstring).
        path = self._index_path(name)
        if path.is_file():
            return  # already exists -- ensure_index only creates when missing anyway
        LocalVectorIndex(path, dimension=dimension)._save({"dimension": dimension, "namespaces": {}})