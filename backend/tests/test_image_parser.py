from pathlib import Path

import pytest

from app.documents.models import BlockType
from app.documents.parsers.image_parser import parse_image


def _has_tesseract() -> bool:
    import shutil

    return shutil.which("tesseract") is not None


def _build_table_image(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 300), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    d.text((20, 20), "Receipt", fill="black", font=font)
    rows = [["Item", "Qty", "Price"], ["Widget", "4", "$10"], ["Gadget", "2", "$25"]]
    y = 80
    for row in rows:
        for x, cell in zip([20, 300, 500], row):
            d.text((x, y), cell, fill="black", font=font)
        y += 50
    img.save(path)


@pytest.mark.skipif(not _has_tesseract(), reason="tesseract binary not available")
def test_parse_image_extracts_table_via_ocr(tmp_path: Path):
    f = tmp_path / "receipt.png"
    _build_table_image(f)

    doc = parse_image(f)

    assert doc.pages[0].is_scanned is True
    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    flat = " ".join(cell for row in tables[0].table.rows for cell in row)
    assert "Widget" in flat
    assert "Gadget" in flat
    assert tables[0].source.value == "ocr"
    assert 0.0 < tables[0].confidence <= 1.0


@pytest.mark.skipif(not _has_tesseract(), reason="tesseract binary not available")
def test_parse_image_no_text_raises(tmp_path: Path):
    from PIL import Image

    f = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), "white").save(f)

    with pytest.raises(ValueError):
        parse_image(f)
