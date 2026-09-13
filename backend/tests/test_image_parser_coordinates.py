"""Regression tests: image_parser.py must map OCR bboxes from the
PROCESSED image's pixel grid back to the ORIGINAL image's pixel grid, and
must report the ORIGINAL image's dimensions on `PageInfo` — not the
processed image's.

`preprocess_and_ocr` is monkeypatched to return deterministic, hand-
constructed `(words, ocr_image, preprocessing_metadata)` for one
transformation at a time (resize / crop / rotation / deskew / combined),
so this test suite is exact and independent of whatever
`select_strategy`'s real heuristics would decide to do with a given real
image — it verifies `parse_image`'s coordinate-mapping wiring itself,
which `tests/test_coordinate_transform.py` (unit tests on the transform
math in isolation) doesn't exercise end-to-end through the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.documents.models import BlockType, CoordinateSpace
from app.documents.ocr.engine import OcrWord
import app.documents.parsers.image_parser as image_parser_module


def _word(text, left, top, width, height, confidence=95.0, line=0, word=0):
    return OcrWord(
        text=text, left=left, top=top, width=width, height=height, confidence=confidence,
        block_num=1, par_num=1, line_num=line, word_num=word,
    )


def _make_original_image(tmp_path: Path, size=(400, 200)) -> Path:
    path = tmp_path / "original.png"
    Image.new("RGB", size, "white").save(path)
    return path


def _patch_preprocess_and_ocr(monkeypatch, words, ocr_image_size, metadata):
    ocr_image = Image.new("L", ocr_image_size, 255)

    def _fake(image, language="eng", confidence_floor=0.0, preprocess=True):
        return words, ocr_image, metadata

    # `parse_image` does `from app.documents.visual.ocr_preprocessing import
    # preprocess_and_ocr` INSIDE the function body, so that import is
    # resolved fresh from the source module at call time — patching the
    # source module's attribute is what actually takes effect.
    import app.documents.visual.ocr_preprocessing as ocr_preprocessing_module

    monkeypatch.setattr(ocr_preprocessing_module, "preprocess_and_ocr", _fake)


def test_page_info_reports_original_dimensions_not_processed(tmp_path, monkeypatch):
    """The core bug: PageInfo must never report the processed image's
    (possibly resized/reoriented) dimensions as if they were the
    original's."""
    original_path = _make_original_image(tmp_path, size=(400, 200))
    words = [_word("Hello", left=10, top=10, width=50, height=20)]
    # Processed image is a DIFFERENT size than the original (simulates
    # resolution normalization) — this is exactly the bug: pre-fix,
    # PageInfo.width/height came from this (wrong) processed size.
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(800, 400),
        metadata={"resolution_scale": 2.0, "fallback_used": False},
    )

    doc = image_parser_module.parse_image(original_path)

    assert doc.pages[0].width == 400
    assert doc.pages[0].height == 200
    assert doc.pages[0].coordinate_space == CoordinateSpace.IMAGE_PIXELS


def test_resize_transform_maps_bbox_back_to_original_scale(tmp_path, monkeypatch):
    """Processed image is 2x the original's resolution -> every OCR bbox
    coordinate must be divided by 2 to land back in original-image space.
    """
    original_path = _make_original_image(tmp_path, size=(400, 200))
    # A word at (20, 20, 120, 60) in the 2x-upscaled processed image
    # should map to (10, 10, 60, 30) in the original.
    words = [_word("Hello", left=20, top=20, width=100, height=40)]
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(800, 400),
        metadata={"resolution_scale": 2.0, "fallback_used": False},
    )

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    assert para.bbox.x0 == pytest.approx(10.0)
    assert para.bbox.y0 == pytest.approx(10.0)
    assert para.bbox.x1 == pytest.approx(60.0)
    assert para.bbox.y1 == pytest.approx(30.0)
    assert para.bbox.coordinate_space == CoordinateSpace.IMAGE_PIXELS


def test_crop_transform_adds_back_crop_offset(tmp_path, monkeypatch):
    """Border cleanup cropped 15px off the left and 25px off the top ->
    every OCR bbox must have that offset ADDED BACK to land in original
    (pre-crop) space."""
    original_path = _make_original_image(tmp_path, size=(400, 200))
    words = [_word("Hello", left=5, top=5, width=50, height=20)]
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(370, 170),
        metadata={"crop_offset": {"x": 15, "y": 25}, "fallback_used": False},
    )

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    assert para.bbox.x0 == pytest.approx(20.0)  # 5 + 15
    assert para.bbox.y0 == pytest.approx(30.0)  # 5 + 25


def test_rotation_transform_maps_bbox_through_inverse_rotation(tmp_path, monkeypatch):
    """A 90-degree orientation correction (expand=True, canvas
    dimensions swap) must be inverted, not left as-is."""
    original_path = _make_original_image(tmp_path, size=(400, 200))
    # Processed (post-orientation) canvas is 200x400 (swapped from the
    # 400x200 original) — consistent with a 90-degree rotation.
    words = [_word("Hello", left=90, top=190, width=20, height=20)]
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(200, 400),
        metadata={
            "orientation_corrected_degrees": 90.0,
            "post_orientation_size": [200, 400],
            "fallback_used": False,
        },
    )

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    # Must not equal the raw processed-space coordinates (90, 190) — the
    # whole point of the fix is that these get transformed.
    assert (para.bbox.x0, para.bbox.y0) != (90, 190)
    # And must land within the ORIGINAL canvas bounds (400x200) — a bug
    # that fails to account for the canvas swap would produce
    # out-of-bounds or nonsensical coordinates.
    assert 0 <= para.bbox.x0 <= 400
    assert 0 <= para.bbox.y0 <= 200


def test_deskew_transform_is_applied_within_same_canvas(tmp_path, monkeypatch):
    """A small deskew angle (expand=False) rotates within the SAME
    canvas size — must still be inverted, not passed through unchanged."""
    original_path = _make_original_image(tmp_path, size=(400, 200))
    words = [_word("Hello", left=200, top=100, width=40, height=20)]
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(400, 200),
        metadata={"skew_corrected_degrees": 3.0, "fallback_used": False},
    )

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    # A nonzero skew correction must move the coordinates measurably
    # (unless the point happens to sit exactly on the rotation center,
    # which it doesn't here).
    assert (para.bbox.x0, para.bbox.y0) != (200, 100)


def test_combined_transformations_compose_correctly(tmp_path, monkeypatch):
    """Resize + crop together: the two operations must be undone in the
    correct order (undo resize, THEN add back the crop offset which was
    measured in pre-resize pixels) — see CoordinateTransform.point_to_original.
    """
    original_path = _make_original_image(tmp_path, size=(400, 200))
    words = [_word("Hello", left=30, top=30, width=40, height=20)]
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(770, 370),
        metadata={
            "crop_offset": {"x": 15, "y": 25},
            "resolution_scale": 2.0,
            "fallback_used": False,
        },
    )

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    # undo resize: 30/2=15, 30/2=15; then add back crop: 15+15=30, 15+25=40
    assert para.bbox.x0 == pytest.approx(30.0)
    assert para.bbox.y0 == pytest.approx(40.0)


def test_fallback_used_skips_transformation_entirely(tmp_path, monkeypatch):
    """When preprocessing fell back to the original image
    (`fallback_used=True`), the OCR'd image IS the original — bboxes must
    pass through unchanged, not be mangled by a transform for
    operations that didn't actually happen."""
    original_path = _make_original_image(tmp_path, size=(400, 200))
    words = [_word("Hello", left=50, top=50, width=40, height=20)]
    _patch_preprocess_and_ocr(
        monkeypatch, words, ocr_image_size=(400, 200),
        metadata={"fallback_used": True, "fallback_reason": "preprocessing raised"},
    )

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    assert para.bbox.x0 == pytest.approx(50.0)
    assert para.bbox.y0 == pytest.approx(50.0)


def test_no_preprocessing_metadata_is_identity_transform(tmp_path, monkeypatch):
    """preprocess=False -> preprocessing_metadata is None -> bboxes pass
    through unchanged (identity transform)."""
    original_path = _make_original_image(tmp_path, size=(400, 200))
    words = [_word("Hello", left=50, top=50, width=40, height=20)]
    _patch_preprocess_and_ocr(monkeypatch, words, ocr_image_size=(400, 200), metadata=None)

    doc = image_parser_module.parse_image(original_path)

    para = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH][0]
    assert para.bbox.x0 == pytest.approx(50.0)
    assert para.bbox.y0 == pytest.approx(50.0)
