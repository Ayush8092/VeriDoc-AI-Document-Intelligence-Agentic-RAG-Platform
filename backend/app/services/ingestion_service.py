"""Ingestion pipeline: corpus + upload files -> chunks -> embeddings -> Pinecone,
plus document/version metadata persisted to the database.

Usage:
    python -m app.services.ingestion_service            # idempotent
    python -m app.services.ingestion_service --reset     # wipe namespace, ingest fresh

Supported input formats: `.md`, `.txt`, `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`
(see app/documents/pipeline.py). `README.txt`/`README.md` are corpus
documentation, never embedded (see app/chunking.py).

Two knowledge bases, one logical corpus: `data/corpus/` is the immutable,
shipped corpus; `data/uploads/` is the mutable, runtime knowledge base
`POST /documents/upload` writes into. Every run chunks and reconciles BOTH
directories together (see app/chunking.py's `chunk_directories`) so a
file's vectors are never mistaken for stale just because it lives in the
other directory this run.

Idempotency: chunk IDs are deterministic, so re-running never creates
duplicate vectors; unchanged chunks are skipped before the embedding call
(`vectorstore.fetch_existing_hashes`); stale vectors (deleted files,
removed sections) are cleaned up every run via
`existing_ids - current_chunk_ids`.

What's new versus the baseline project this was ported from (see
MIGRATION_PLAN.md): document-level metadata (page count, table count,
whether OCR ran, per-run audit trail) is now persisted to the database via
`app/db/models.py`, and a file's own content hash — separate from each
chunk's `content_hash` — is used to detect a "new version" of a document
and bump `DocumentVersion.version_number` instead of only tracking chunk
-level change.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from app import lexical_index, vectorstore
from app.chunking import Chunk, chunk_document
from app.clients import ensure_index, get_embeddings, get_pinecone
from app.core.config import Settings, get_settings
from app.db.models import Document, DocumentVersion, IngestionRun
from app.db.session import get_session_factory, init_db, session_scope
from app.documents.pipeline import SUPPORTED_EXTENSIONS, extract_document
from app.storage.base import CORPUS_KEY_PREFIX, UPLOADS_KEY_PREFIX, StorageBackend, get_backend, local_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_dir(settings: Settings, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_document_version(
    session, source_root: str, filename: str, file_hash: str, doc_summary: dict, chunk_count: int, owner_id: int | None = None
) -> None:
    document = (
        session.query(Document)
        .filter(Document.source_root == source_root, Document.filename == filename, Document.owner_id == owner_id)
        .one_or_none()
    )
    if document is None:
        document = Document(
            source_root=source_root,
            filename=filename,
            file_type=Path(filename).suffix.lower().lstrip("."),
            title=doc_summary.get("title", Path(filename).stem),
            owner_id=owner_id,
        )
        session.add(document)
        session.flush()

    latest_version = (
        session.query(DocumentVersion)
        .filter(DocumentVersion.document_id == document.id)
        .order_by(DocumentVersion.version_number.desc())
        .first()
    )
    if latest_version is not None and latest_version.file_hash == file_hash:
        return  # unchanged file content, no new version needed

    version_number = (latest_version.version_number + 1) if latest_version else 1
    version = DocumentVersion(
        document_id=document.id,
        version_number=version_number,
        file_hash=file_hash,
        page_count=doc_summary.get("page_count", 0),
        has_scanned_pages=doc_summary.get("has_scanned_pages", False),
        table_count=doc_summary.get("table_count", 0),
        chunk_count=chunk_count,
        mean_ocr_confidence=doc_summary.get("mean_ocr_confidence"),
        warnings="\n".join(doc_summary.get("warnings", [])),
    )
    session.add(version)
    session.flush()
    document.current_version_id = version.id


def _list_source_files(backend: StorageBackend, source_root: str, prefix: str) -> list[tuple[str, str, int | None]]:
    """Enumerate every ingestable file under `prefix` (a storage-key
    prefix, e.g. `"corpus"` or `"uploads"` in production — see
    `ingest_corpus`, which passes the ACTUAL directory name being
    scanned rather than a hardcoded constant, specifically so a caller
    that overrides `corpus_dir`/`upload_dir` with a differently-named
    directory — e.g. a test's `tmp_path / "empty_corpus"` — gets that
    exact directory scanned, never silently mismatched against a
    hardcoded `"corpus"`/`"uploads"` that happens not to exist) via the
    storage backend's `list_keys()` — the Phase 5 storage-abstraction
    migration this function replaces a direct `Path.iterdir()` walk with
    (see `app/storage/base.py`'s module docstring, "Honest scope note",
    which this function resolves).

    Returns `(key, filename, owner_id)` tuples:
      - `key` — the storage key, to pass to `local_path()`.
      - `filename` — the bare original filename (NOT the possibly-
        randomized temp-file name a remote backend's `get_local_path`
        returns — see `_ingest_one`, which needs the real name for
        `chunk_document`/`Document.filename`).
      - `owner_id` — `None` for `source_root == "corpus"` (always public)
        and for a legacy flat file directly under the uploads prefix;
        the uploader's user id for a `<prefix>/<user_id>/<filename>` key.

    `list_keys` is RECURSIVE (unlike the `Path.iterdir()` this replaces),
    so the per-user grouping that used to come from "which subdirectory
    is this file in" is reconstructed here by parsing each key's path
    segments instead — see the corpus-vs-uploads branches below. A key
    whose uploads-relative path doesn't match either the flat
    `<filename>` or `<user_id>/<filename>` shape (e.g. a stray non-
    numeric folder) is skipped, exactly as the original directory-walk
    version skipped a non-numeric subdirectory name rather than guessing
    an owner.
    """
    from app.chunking import EXCLUDED_FILENAMES

    def _is_ingestable(filename: str) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix in SUPPORTED_EXTENSIONS and filename.lower() not in EXCLUDED_FILENAMES

    key_prefix = f"{prefix}/"
    out: list[tuple[str, str, int | None]] = []
    for key in backend.list_keys(key_prefix):
        relative = key[len(key_prefix):]
        parts = relative.split("/")
        if source_root == "corpus":
            if len(parts) != 1 or not _is_ingestable(parts[0]):
                continue
            out.append((key, parts[0], None))
        else:
            if len(parts) == 1:
                if _is_ingestable(parts[0]):
                    out.append((key, parts[0], None))
            elif len(parts) == 2 and parts[0].isdigit():
                if _is_ingestable(parts[1]):
                    out.append((key, parts[1], int(parts[0])))
            # else: not a recognized shape -- skip, matching the original
            # walk's "not subdir.name.isdigit(): skip" behavior.

    # Deterministic order: public/root files first, then per-user files
    # grouped by owner, filename alphabetical within each group -- close
    # to (and functionally equivalent to, for ingestion correctness,
    # which does not depend on order) the original iterdir()-based order.
    out.sort(key=lambda item: (item[2] is not None, item[2] or 0, item[1]))
    return out


def _chunk_source_root(
    backend: StorageBackend, source_root: str, prefix: str, settings: Settings, session
) -> list[Chunk]:
    def _ingest_one(key: str, filename: str, owner_id: int | None) -> list[Chunk]:
        # `local_path()` handles cleanup automatically on exit -- see
        # app/storage/base.py's docstring for why this (rather than
        # get_local_path/release_local_path directly) is correct here:
        # this path never needs to outlive this function call.
        with local_path(backend, key) as raw_path:
            path = Path(raw_path)
            file_hash = _file_hash(path)
            extracted = extract_document(path, settings)
            file_chunks = chunk_document(
                extracted, filename=filename, source_root=source_root, owner_id=str(owner_id) if owner_id else None
            )

            ocr_confs = [p.ocr_confidence for p in extracted.pages if p.ocr_confidence is not None]
            summary = {
                "title": file_chunks[0].document_title if file_chunks else Path(filename).stem,
                "page_count": extracted.page_count,
                "has_scanned_pages": extracted.has_scanned_pages,
                "table_count": len(extracted.tables),
                "mean_ocr_confidence": (sum(ocr_confs) / len(ocr_confs)) if ocr_confs else None,
                "warnings": extracted.warnings,
            }
            if session is not None:
                _record_document_version(
                    session, source_root, filename, file_hash, summary, len(file_chunks), owner_id=owner_id
                )
            return file_chunks

    chunks: list[Chunk] = []
    for key, filename, owner_id in _list_source_files(backend, source_root, prefix):
        chunks.extend(_ingest_one(key, filename, owner_id))
    return chunks


def _record_ingestion_run(settings: Settings, *, reset: bool, summary: dict, run_error: str | None) -> None:
    """Persist an `IngestionRun` audit row in its own transaction.

    Deliberately independent of the ingestion-work session (see
    `ingest_corpus`) so the audit trail is written whether that session
    committed or rolled back, and never carries partial ingestion state
    into the main database transaction.
    """
    import datetime as dt

    with session_scope(settings) as session:
        run = IngestionRun(
            reset=reset,
            chunks_total=summary.get("chunks", 0),
            chunks_changed=summary.get("changed", 0),
            vectors_upserted=summary.get("upserted", 0),
            stale_vectors_deleted=summary.get("stale_deleted", 0),
            status="failed" if run_error else "success",
            detail=run_error or "",
        )
        run.finished_at = dt.datetime.now(dt.timezone.utc)
        session.add(run)


def ingest_corpus(
    reset: bool = False,
    settings: Settings | None = None,
    corpus_dir: Path | None = None,
    upload_dir: Path | None = None,
    persist_metadata: bool = True,
) -> dict:
    """Run the full ingestion pipeline once, across BOTH `corpus_dir` and
    `upload_dir` together. Returns a summary dict and records an
    `IngestionRun` audit row (when `persist_metadata` is true).

    Storage backend selection: if `corpus_dir`/`upload_dir` are passed
    explicitly (as every test in `tests/test_ingestion_service.py` does,
    and as any other caller that wants to point ingestion at a specific
    local directory tree can), scanning goes through a
    `LocalStorageBackend` rooted at their shared parent — this override
    is honored exactly as it was before the storage-abstraction
    migration, not silently ignored in favor of
    `settings.storage_local_root`. Only when NEITHER is overridden
    (the normal CLI/production path) does scanning go through
    `app.storage.base.get_backend(settings)`, which is what actually
    makes `STORAGE_BACKEND=s3` usable for corpus/upload scanning — see
    `app/storage/base.py`'s module docstring, "Honest scope note",
    which this resolves.
    """
    settings = settings or get_settings()
    explicit_dirs = corpus_dir is not None or upload_dir is not None
    corpus_dir = corpus_dir or _resolve_dir(settings, settings.corpus_dir)
    upload_dir = upload_dir or _resolve_dir(settings, settings.upload_dir)

    if explicit_dirs:
        # Scan the EXACT directories passed in, under whatever names they
        # actually have (e.g. a test's `tmp_path / "empty_corpus"`) —
        # never silently assume they're named "corpus"/"uploads". Both
        # must share a common parent so a single LocalStorageBackend can
        # be rooted there; `corpus_prefix`/`upload_prefix` below are each
        # directory's own name, used as the storage-key prefix instead of
        # the fixed CORPUS_KEY_PREFIX/UPLOADS_KEY_PREFIX constants (those
        # constants are only correct for the settings-derived default
        # path — see `app/storage/base.py`).
        roots = {corpus_dir.parent, upload_dir.parent}
        if len(roots) != 1:
            raise ValueError(
                "corpus_dir and upload_dir must share a common parent directory when "
                f"either is overridden (got parents {sorted(str(r) for r in roots)})"
            )
        from app.storage.local import LocalStorageBackend

        backend: StorageBackend = LocalStorageBackend(root=roots.pop())
        corpus_prefix, upload_prefix = corpus_dir.name, upload_dir.name
    else:
        backend = get_backend(settings)
        corpus_prefix, upload_prefix = CORPUS_KEY_PREFIX, UPLOADS_KEY_PREFIX

    if persist_metadata:
        init_db(settings)

    started = time.time()
    run_error: str | None = None

    def _do_ingest(session) -> dict:
        chunks: list[Chunk] = []
        seen: dict[str, str] = {}
        for source_root, prefix in (("corpus", corpus_prefix), ("uploads", upload_prefix)):
            root_chunks = _chunk_source_root(backend, source_root, prefix, settings, session)
            for chunk in root_chunks:
                if chunk.chunk_id in seen:
                    raise ValueError(f"chunk_id collision: '{chunk.chunk_id}'")
                seen[chunk.chunk_id] = source_root
            chunks.extend(root_chunks)

        if not chunks:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise SystemExit(f"No ingestable files ({supported}) found in {corpus_dir} or {upload_dir}")

        current_chunk_ids = {c.chunk_id for c in chunks}
        lexical_index_path = _resolve_dir(settings, settings.lexical_index_path)

        pc = get_pinecone(settings)
        index = ensure_index(pc, settings)

        if reset:
            vectorstore.delete_namespace(pc, settings)
            existing_hashes: dict[str, str] = {}
        else:
            existing_hashes = vectorstore.fetch_existing_hashes(index, settings, [c.chunk_id for c in chunks])

        changed = [c for c in chunks if existing_hashes.get(c.chunk_id) != c.content_hash]
        skipped = len(chunks) - len(changed)

        upserted = 0
        if changed:
            embeddings = get_embeddings(settings)
            vectors = embeddings.embed_documents([c.text for c in changed])
            upserted = vectorstore.upsert_chunks(index, settings, changed, vectors)

        if reset:
            stale_deleted = 0
            stale_chunk_ids: list[str] = []
        else:
            existing_ids = vectorstore.list_all_ids(index, settings)
            stale_ids = sorted(existing_ids - current_chunk_ids)
            stale_deleted = vectorstore.delete_vectors(index, settings, stale_ids)
            stale_chunk_ids = stale_ids

        total = vectorstore.vector_count(pc, settings)

        # Write the BM25 lexical index snapshot (app/lexical_index.py) LAST,
        # only once the Pinecone upsert + stale-vector cleanup above have
        # both actually succeeded. Writing it earlier would let the lexical
        # snapshot describe chunks that a subsequent Pinecone failure never
        # actually got to embed — the same "new state recorded, indexing
        # never finished" inconsistency this function's transaction split
        # (see `_record_ingestion_run`) exists to prevent. Both the vector
        # index and the lexical index must reflect the same, fully-completed
        # ingestion run — never a partial one.
        lexical_index.save_snapshot(chunks, lexical_index_path)

        return {
            "corpus_dir": str(corpus_dir),
            "upload_dir": str(upload_dir),
            "chunks": len(chunks),
            "changed": len(changed),
            "skipped_unchanged": skipped,
            "upserted": upserted,
            "stale_deleted": stale_deleted,
            "stale_chunk_ids": stale_chunk_ids,
            "namespace_vector_count": total,
            "chunk_ids": [c.chunk_id for c in chunks],
            "changed_chunk_ids": [c.chunk_id for c in changed],
            "source_files": sorted({c.source_file for c in chunks}),
            "table_chunks": sum(1 for c in chunks if c.block_type == "table"),
            "ocr_chunks": sum(1 for c in chunks if c.source == "ocr"),
            "lexical_index_path": str(lexical_index_path),
            "lexical_index_chunks": len(chunks),
        }

    summary: dict = {}
    if persist_metadata:
        # The ingestion work (`_do_ingest`) and the `IngestionRun` audit row
        # deliberately run in TWO SEPARATE transactions/sessions.
        #
        # A single-session version of this used to add the audit row in a
        # `finally` block and unconditionally `session.commit()` it — but
        # that also committed whatever Document/DocumentVersion rows
        # `_do_ingest` had already `session.add()`ed (via
        # `_record_document_version`) *before* it failed, e.g. on a
        # mid-run Pinecone/embedding error. The result: the database would
        # show a document "successfully" at version N with page/table
        # stats, while the corresponding vectors were never actually
        # upserted — an ingestion that half-failed but looked fully
        # ingested to `GET /documents`. That's exactly the
        # new-file-vs-stale-index inconsistency the upload/ingestion
        # pipeline must never produce.
        #
        # Splitting the transactions fixes it: `work_session` is rolled
        # back on ANY failure (discarding partial document/version writes
        # for this run), while the audit row is written and committed in
        # its own short-lived session regardless of outcome, so there is
        # always a record of what happened without that record being
        # contaminated by — or contaminating — partially-applied state.
        factory = get_session_factory(settings)
        work_session = factory()
        try:
            summary = _do_ingest(work_session)
        except BaseException as exc:
            # BaseException (not just Exception) so a "no ingestable
            # files found" SystemExit is also rolled back and recorded as
            # a failed run rather than silently reporting `status="success"`.
            work_session.rollback()
            run_error = str(exc)
            raise
        else:
            work_session.commit()
        finally:
            work_session.close()
            _record_ingestion_run(settings, reset=reset, summary=summary, run_error=run_error)
    else:
        summary = _do_ingest(None)

    summary["duration_seconds"] = round(time.time() - started, 3)
    return summary


def run(reset: bool = False, settings: Settings | None = None) -> dict:
    return ingest_corpus(reset=reset, settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus + uploads into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="Delete the namespace before ingesting.")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Index: {settings.pinecone_index_name}  Namespace: {settings.pinecone_namespace}")
    summary = run(reset=args.reset, settings=settings)

    print(f"Chunked corpus + uploads -> {summary['chunks']} chunks, {len(summary['source_files'])} files")
    print(f"  ({summary['table_chunks']} table chunks, {summary['ocr_chunks']} OCR-sourced chunks)")
    if args.reset:
        print("Namespace cleared (--reset)")
    if summary["skipped_unchanged"]:
        print(f"Skipped {summary['skipped_unchanged']} unchanged chunks")
    print(f"Lexical (BM25) index snapshot: {summary['lexical_index_chunks']} chunks -> {summary['lexical_index_path']}")
    print(f"Embedded + upserted {summary['upserted']} vectors")
    if summary["stale_deleted"]:
        print(f"Deleted {summary['stale_deleted']} stale vectors")
    print(f"Namespace now holds {summary['namespace_vector_count']} vectors")
    print(f"Done in {summary['duration_seconds']}s")


if __name__ == "__main__":
    main()