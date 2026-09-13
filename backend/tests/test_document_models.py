from app.documents.models import BlockType, Block, ExtractedDocument, TableData


def test_table_data_to_markdown_pads_ragged_rows():
    table = TableData(rows=(("A", "B", "C"), ("1", "2")))
    md = table.to_markdown()
    lines = md.splitlines()
    assert lines[0] == "| A | B | C |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 1 | 2 |  |"


def test_table_data_n_rows_n_cols():
    table = TableData(rows=(("A", "B"), ("1", "2"), ("3", "4")))
    assert table.n_rows == 3
    assert table.n_cols == 2


def test_extracted_document_plain_text_flattens_tables_to_markdown():
    doc = ExtractedDocument(source_path="x.md", file_type="md")
    doc.blocks.append(Block(block_type=BlockType.PARAGRAPH, text="intro", page_number=1, order=0))
    doc.blocks.append(
        Block(
            block_type=BlockType.TABLE,
            text="| a | b |\n| --- | --- |\n| 1 | 2 |",
            page_number=1,
            order=1,
            table=TableData(rows=(("a", "b"), ("1", "2"))),
        )
    )
    text = doc.plain_text()
    assert "intro" in text
    assert "| a | b |" in text


def test_has_scanned_pages_and_tables_properties():
    from app.documents.models import PageInfo

    doc = ExtractedDocument(source_path="x", file_type="png", pages=[PageInfo(page_number=1, is_scanned=True)])
    doc.blocks.append(Block(block_type=BlockType.TABLE, text="t", table=TableData(rows=(("a",),))))
    assert doc.has_scanned_pages is True
    assert len(doc.tables) == 1
