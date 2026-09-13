from pathlib import Path

from app.core.config import Settings
from app.chunking import chunk_document, chunk_file, chunk_corpus
from app.documents.parsers.text_parser import parse_text


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_chunk_document_splits_by_heading_sections(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Doc\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n")
    extracted = parse_text(f)

    chunks = chunk_document(extracted, filename="doc.md", source_root="corpus")

    sections = {c.section for c in chunks}
    assert sections == {"Section A", "Section B"}
    assert all(c.document_title == "Doc" for c in chunks)


def test_chunk_document_table_is_always_its_own_chunk(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text(
        "# Doc\n\n## Pricing\n\nIntro text.\n\n"
        "| Plan | Price |\n| --- | --- |\n| Basic | $5 |\n\n"
        "Outro text.\n"
    )
    extracted = parse_text(f)
    chunks = chunk_document(extracted, filename="doc.md", source_root="corpus")

    table_chunks = [c for c in chunks if c.block_type == "table"]
    text_chunks = [c for c in chunks if c.block_type == "text"]

    assert len(table_chunks) == 1
    assert "Basic" in table_chunks[0].text
    # table content never leaks into a text chunk
    assert all("Basic" not in c.text for c in text_chunks)
    # surrounding prose still produced its own chunk(s)
    assert any("Intro text" in c.text for c in text_chunks)
    assert any("Outro text" in c.text for c in text_chunks)


def test_chunk_id_is_deterministic_across_runs(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Doc\n\n## Section A\n\nBody.\n")
    extracted = parse_text(f)

    chunks_1 = chunk_document(extracted, filename="doc.md", source_root="corpus")
    chunks_2 = chunk_document(extracted, filename="doc.md", source_root="corpus")

    assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]
    assert [c.content_hash for c in chunks_1] == [c.content_hash for c in chunks_2]


def test_chunk_no_headings_falls_back_to_one_section(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("Just some plain text with no structure at all.")
    extracted = parse_text(f)
    chunks = chunk_document(extracted, filename="notes.txt", source_root="uploads")

    assert len(chunks) == 1
    assert chunks[0].section == "content"


def test_oversized_section_is_split_into_overlapping_windows(tmp_path: Path):
    f = tmp_path / "doc.md"
    body = "word " * 500  # forces > MAX_CHUNK_CHARS
    f.write_text(f"# Doc\n\n## Long Section\n\n{body}\n")
    extracted = parse_text(f)
    chunks = chunk_document(extracted, filename="doc.md", source_root="corpus")

    assert len(chunks) > 1
    assert all(c.section == "Long Section" for c in chunks)


def test_chunk_corpus_skips_readme_and_missing_dir(tmp_path: Path):
    (tmp_path / "README.txt").write_text("not a knowledge document")
    (tmp_path / "a.md").write_text("# A\n\nBody\n")
    settings = _settings()

    chunks = chunk_corpus(tmp_path, settings, source_root="corpus")
    assert all(c.source_file != "README.txt" for c in chunks)
    assert any(c.source_file == "a.md" for c in chunks)

    assert chunk_corpus(tmp_path / "does-not-exist", settings) == []


def test_chunk_file_dispatches_by_extension(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("# A\n\nBody text.\n")
    chunks = chunk_file(f, _settings(), source_root="corpus")
    assert chunks[0].source_file == "a.md"
