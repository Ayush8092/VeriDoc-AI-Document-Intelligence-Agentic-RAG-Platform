from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.db.models import Document, DocumentVersion, IngestionRun
from app.db.session import init_db, reset_engine_for_tests, session_scope
from app.services import ingestion_service as isvc


@pytest.fixture()
def settings(tmp_path):
    reset_engine_for_tests()
    s = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    init_db(s)
    yield s
    reset_engine_for_tests()


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.md").write_text("# A\n\n## Intro\n\nHello world.\n")
    return d


@pytest.fixture()
def upload_dir(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    return d


def _run_ingest(settings, corpus_dir, upload_dir, reset=False, existing_hashes=None):
    fake_index = MagicMock()
    fake_pc = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

    with patch.object(isvc, "get_pinecone", return_value=fake_pc), \
         patch.object(isvc, "ensure_index", return_value=fake_index), \
         patch.object(isvc, "get_embeddings", return_value=fake_embedder), \
         patch("app.vectorstore.fetch_existing_hashes", return_value=existing_hashes or {}), \
         patch("app.vectorstore.upsert_chunks", side_effect=lambda idx, s, chunks, vectors: len(chunks)), \
         patch("app.vectorstore.list_all_ids", return_value=set()), \
         patch("app.vectorstore.delete_vectors", return_value=0), \
         patch("app.vectorstore.vector_count", return_value=0):
        return isvc.ingest_corpus(settings=settings, corpus_dir=corpus_dir, upload_dir=upload_dir)


def test_ingest_creates_document_and_version(settings, corpus_dir, upload_dir):
    summary = _run_ingest(settings, corpus_dir, upload_dir)

    assert summary["chunks"] > 0
    assert summary["changed"] == summary["chunks"]  # first run, nothing exists yet

    with session_scope(settings) as session:
        docs = session.query(Document).all()
        assert len(docs) == 1
        assert docs[0].filename == "a.md"
        version = session.get(DocumentVersion, docs[0].current_version_id)
        assert version.version_number == 1

        runs = session.query(IngestionRun).all()
        assert len(runs) == 1
        assert runs[0].status == "success"


def test_ingest_skips_unchanged_chunks_on_second_run(settings, corpus_dir, upload_dir):
    first = _run_ingest(settings, corpus_dir, upload_dir)
    existing_hashes = {cid: "matching" for cid in first["chunk_ids"]}
    # Simulate Pinecone already having every chunk with the SAME content hash
    # as what we're about to (re-)compute, by patching content_hash lookups
    # to match — easiest is to fetch the real hashes from a second parse.
    from app.chunking import chunk_corpus

    real_chunks = chunk_corpus(corpus_dir, settings, source_root="corpus")
    existing_hashes = {c.chunk_id: c.content_hash for c in real_chunks}

    second = _run_ingest(settings, corpus_dir, upload_dir, existing_hashes=existing_hashes)

    assert second["changed"] == 0
    assert second["skipped_unchanged"] == second["chunks"]


def test_ingest_bumps_document_version_when_file_content_changes(settings, corpus_dir, upload_dir):
    _run_ingest(settings, corpus_dir, upload_dir)

    (corpus_dir / "a.md").write_text("# A\n\n## Intro\n\nHello world, updated!\n")
    _run_ingest(settings, corpus_dir, upload_dir)

    with session_scope(settings) as session:
        doc = session.query(Document).filter(Document.filename == "a.md").one()
        versions = session.query(DocumentVersion).filter(DocumentVersion.document_id == doc.id).all()
        assert len(versions) == 2
        assert {v.version_number for v in versions} == {1, 2}


def test_ingest_records_failed_run_on_error(settings, corpus_dir, upload_dir):
    fake_pc = MagicMock()
    with patch.object(isvc, "get_pinecone", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            isvc.ingest_corpus(settings=settings, corpus_dir=corpus_dir, upload_dir=upload_dir)

    with session_scope(settings) as session:
        runs = session.query(IngestionRun).all()
        assert len(runs) == 1
        assert runs[0].status == "failed"
        assert "boom" in runs[0].detail


def test_ingest_raises_systemexit_when_no_files_found(settings, tmp_path):
    empty_corpus = tmp_path / "empty_corpus"
    empty_corpus.mkdir()
    empty_uploads = tmp_path / "empty_uploads"
    empty_uploads.mkdir()

    with patch.object(isvc, "get_pinecone", return_value=MagicMock()), \
         patch.object(isvc, "ensure_index", return_value=MagicMock()):
        with pytest.raises(SystemExit):
            isvc.ingest_corpus(settings=settings, corpus_dir=empty_corpus, upload_dir=empty_uploads)


def test_ingest_failed_run_does_not_persist_partial_documents(settings, corpus_dir, upload_dir):
    """Regression test for the Phase 3A hardening fix: a mid-run failure
    must never leave Document/DocumentVersion rows committed for a file
    whose vectors were never actually upserted (see
    `ingestion_service.ingest_corpus` / `_record_ingestion_run`).

    `_chunk_directory_with_metadata` already `session.add()`ed a Document +
    DocumentVersion for `a.md` by the time `get_pinecone` (called right
    after) raises — the old implementation committed that partial write
    anyway because the `IngestionRun` audit row was added and committed in
    a shared `finally` block on the SAME session. It must now be rolled
    back instead.
    """
    with patch.object(isvc, "get_pinecone", side_effect=RuntimeError("pinecone unavailable")):
        with pytest.raises(RuntimeError):
            isvc.ingest_corpus(settings=settings, corpus_dir=corpus_dir, upload_dir=upload_dir)

    with session_scope(settings) as session:
        assert session.query(Document).all() == []
        assert session.query(DocumentVersion).all() == []

        runs = session.query(IngestionRun).all()
        assert len(runs) == 1
        assert runs[0].status == "failed"
        assert "pinecone unavailable" in runs[0].detail


def test_ingest_failed_run_does_not_write_lexical_snapshot(settings, corpus_dir, upload_dir, tmp_path):
    """The lexical index snapshot must reflect only a fully-completed
    ingestion run (see `ingest_corpus`'s `_do_ingest`: the snapshot is now
    written last, after the Pinecone upsert + stale-vector cleanup
    succeed) — never a run that failed partway through.
    """
    lexical_path = tmp_path / "lexical.json"
    settings.lexical_index_path = str(lexical_path)

    with patch.object(isvc, "get_pinecone", side_effect=RuntimeError("pinecone unavailable")):
        with pytest.raises(RuntimeError):
            isvc.ingest_corpus(settings=settings, corpus_dir=corpus_dir, upload_dir=upload_dir)

    assert not lexical_path.exists()


def test_lexical_index_snapshot_chunk_count_matches_corpus_after_ingest(settings, corpus_dir, upload_dir, tmp_path):
    """Section 13's explicit requirement: corpus chunks == lexical index
    chunks after a successful ingestion run.
    """
    from app import lexical_index

    lexical_path = tmp_path / "lexical.json"
    settings.lexical_index_path = str(lexical_path)

    summary = _run_ingest(settings, corpus_dir, upload_dir)

    index = lexical_index.LexicalIndex.from_path(lexical_path)
    assert len(index) == summary["chunks"]
    assert summary["lexical_index_chunks"] == summary["chunks"]
