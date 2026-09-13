from pathlib import Path

import pytest

from app.documents.models import BlockType
from app.documents.parsers.text_parser import parse_text


def test_parse_markdown_headings_and_paragraphs(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nIntro para.\n\n## Section One\n\nBody text here.\n")

    doc = parse_text(f)

    heading_texts = [b.text for b in doc.blocks if b.block_type == BlockType.HEADING]
    assert heading_texts == ["Title", "Section One"]

    paragraph_texts = [b.text for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
    assert "Intro para." in paragraph_texts
    assert "Body text here." in paragraph_texts


def test_parse_markdown_pipe_table_becomes_table_block(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text(
        "# Title\n\n"
        "## Pricing\n\n"
        "| Plan | Price |\n"
        "| --- | --- |\n"
        "| Basic | $5 |\n"
        "| Pro | $25 |\n\n"
        "Some text after the table.\n"
    )

    doc = parse_text(f)

    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].table.rows == (("Plan", "Price"), ("Basic", "$5"), ("Pro", "$25"))

    paragraphs = [b.text for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
    assert "Some text after the table." in paragraphs


def test_parse_txt_has_no_headings_and_no_tables(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("Just plain text.\n\n| not | a table |\nsecond line without separator\n")

    doc = parse_text(f)

    assert all(b.block_type != BlockType.HEADING for b in doc.blocks)
    assert all(b.block_type != BlockType.TABLE for b in doc.blocks)


def test_parse_text_empty_file_raises(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n\n  ")
    with pytest.raises(ValueError):
        parse_text(f)


def test_parse_text_invalid_utf8_raises(tmp_path: Path):
    f = tmp_path / "bad.txt"
    f.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValueError):
        parse_text(f)
