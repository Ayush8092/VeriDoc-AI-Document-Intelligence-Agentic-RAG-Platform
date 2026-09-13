import pytest

from app.core.config import Settings
from app.db.models import Document, DocumentVersion, IngestionRun
from app.db.session import init_db, reset_engine_for_tests, session_scope


@pytest.fixture()
def settings(tmp_path):
    reset_engine_for_tests()
    s = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    init_db(s)
    yield s
    reset_engine_for_tests()


def test_create_document_and_version(settings):
    with session_scope(settings) as session:
        doc = Document(source_root="corpus", filename="a.md", file_type="md", title="A")
        session.add(doc)
        session.flush()
        version = DocumentVersion(
            document_id=doc.id,
            version_number=1,
            file_hash="abc123",
            page_count=1,
            has_scanned_pages=False,
            table_count=0,
            chunk_count=2,
        )
        session.add(version)
        session.flush()
        doc.current_version_id = version.id

    with session_scope(settings) as session:
        docs = session.query(Document).all()
        assert len(docs) == 1
        assert docs[0].filename == "a.md"
        assert docs[0].current_version_id is not None


def test_document_version_to_dict_parses_warnings(settings):
    with session_scope(settings) as session:
        doc = Document(source_root="corpus", filename="b.pdf", file_type="pdf", title="B")
        session.add(doc)
        session.flush()
        version = DocumentVersion(
            document_id=doc.id,
            version_number=1,
            file_hash="h",
            warnings="page 2: could not be rendered\npage 3: OCR found no text",
        )
        session.add(version)
        session.flush()
        d = version.to_dict()
        assert d["warnings"] == ["page 2: could not be rendered", "page 3: OCR found no text"]


def test_ingestion_run_roundtrip(settings):
    with session_scope(settings) as session:
        run = IngestionRun(reset=False, chunks_total=10, chunks_changed=3, vectors_upserted=3, status="success")
        session.add(run)

    with session_scope(settings) as session:
        runs = session.query(IngestionRun).all()
        assert len(runs) == 1
        assert runs[0].status == "success"
