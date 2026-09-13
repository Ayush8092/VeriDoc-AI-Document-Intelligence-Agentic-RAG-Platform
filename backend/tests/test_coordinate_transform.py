"""Tests for app.documents.visual.coordinates.CoordinateTransform.

Phase 4 Issues 2/3: OCR preprocessing can rotate/crop/resize the image
OCR actually runs against, so a raw OCR bbox is in that processed
image's pixel grid, not the original page's. These tests validate the
inverse transform against PIL's OWN `rotate()` output (not just the
transform's internal consistency) using a marker-pixel technique: place
a single dark pixel at a known location, apply the exact rotation
`ocr_preprocessing.preprocess_image` would apply, locate where PIL
actually put it, and confirm `CoordinateTransform` maps that location
back to (approximately) the original — round-trip correctness measured
against ground truth, not just "the math is internally symmetric".
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.documents.models import BoundingBox, CoordinateSpace
from app.documents.visual.coordinates import CoordinateTransform, _invert_rotation_point


def _find_marker(image: Image.Image) -> tuple[float, float]:
    arr = np.asarray(image.convert("L"))
    ys, xs = np.where(arr < 128)
    assert len(xs) > 0, "marker pixel not found after rotation"
    return float(xs.mean()), float(ys.mean())


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_invert_rotation_point_matches_real_pil_orientation_rotation(angle):
    w, h = 200, 100
    img = Image.new("L", (w, h), 255)
    mx, my = 150, 20
    img.putpixel((mx, my), 0)

    rotated = img.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=255)
    found_x, found_y = _find_marker(rotated)

    recovered_x, recovered_y = _invert_rotation_point(
        found_x, found_y, angle, src_size=(w, h), dst_size=rotated.size
    )
    assert recovered_x == pytest.approx(mx, abs=1.5)
    assert recovered_y == pytest.approx(my, abs=1.5)


@pytest.mark.parametrize("angle", [15.0, -7.5, 2.3])
def test_invert_rotation_point_matches_real_pil_skew_rotation(angle):
    w, h = 200, 100
    img = Image.new("L", (w, h), 255)
    mx, my = 150, 20
    img.putpixel((mx, my), 0)

    rotated = img.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=255)
    found_x, found_y = _find_marker(rotated)

    recovered_x, recovered_y = _invert_rotation_point(
        found_x, found_y, angle, src_size=(w, h), dst_size=(w, h)
    )
    assert recovered_x == pytest.approx(mx, abs=1.5)
    assert recovered_y == pytest.approx(my, abs=1.5)


def test_identity_transform_is_a_noop():
    t = CoordinateTransform(rendered_width=1000, rendered_height=1400)
    bbox = BoundingBox(x0=10, y0=20, x1=110, y1=70)
    out = t.bbox_to_original(bbox)
    assert (out.x0, out.y0, out.x1, out.y1) == (10, 20, 110, 70)
    assert out.coordinate_space == CoordinateSpace.IMAGE_PIXELS


def test_resolution_scale_only():
    # Processed image is 2x the rendered image's size.
    t = CoordinateTransform(rendered_width=1000, rendered_height=1400, resolution_scale=2.0)
    bbox = BoundingBox(x0=200, y0=400, x1=400, y1=600)  # in processed-image pixels
    out = t.bbox_to_original(bbox)
    assert out.x0 == pytest.approx(100)
    assert out.y0 == pytest.approx(200)
    assert out.x1 == pytest.approx(200)
    assert out.y1 == pytest.approx(300)


def test_crop_offset_only():
    t = CoordinateTransform(rendered_width=1000, rendered_height=1400, crop_offset_x=30, crop_offset_y=50)
    bbox = BoundingBox(x0=10, y0=10, x1=60, y1=60)  # in cropped-image pixels
    out = t.bbox_to_original(bbox)
    assert (out.x0, out.y0, out.x1, out.y1) == (40, 60, 90, 110)


def test_dpi_conversion_to_pdf_points():
    # A PDF rendered at 200 DPI: 1 PDF point = 200/72 rendered pixels.
    t = CoordinateTransform(
        rendered_width=1700, rendered_height=2200, dpi=200, output_space=CoordinateSpace.PDF_POINTS
    )
    bbox = BoundingBox(x0=200, y0=200, x1=400, y1=400)  # rendered-image pixels (no preprocessing)
    out = t.bbox_to_original(bbox)
    assert out.x0 == pytest.approx(200 * 72 / 200)
    assert out.x1 == pytest.approx(400 * 72 / 200)
    assert out.coordinate_space == CoordinateSpace.PDF_POINTS


def test_full_pipeline_orientation_then_crop_then_resize_round_trips_a_real_marker():
    """End-to-end: build an image, run it through the EXACT sequence
    `preprocess_image` uses (orientation rotate -> crop -> resize),
    locate a known marker's real post-processing position, and confirm
    `CoordinateTransform.from_preprocessing`-style construction recovers
    the marker's original position."""
    w, h = 300, 200
    img = Image.new("L", (w, h), 255)
    mx, my = 220, 40
    img.putpixel((mx, my), 0)

    orientation_degrees = 90
    rotated = img.rotate(orientation_degrees, resample=Image.NEAREST, expand=True, fillcolor=255)
    post_orientation_size = rotated.size

    crop_left, crop_top = 15, 10
    cropped_arr = np.asarray(rotated)[crop_top:, crop_left:]
    cropped = Image.fromarray(cropped_arr)
    post_crop_size = cropped.size

    scale = 1.5
    resized = cropped.resize((round(cropped.width * scale), round(cropped.height * scale)), resample=Image.NEAREST)

    found_x, found_y = _find_marker(resized)

    metadata = {
        "orientation_corrected_degrees": orientation_degrees,
        "post_orientation_size": list(post_orientation_size),
        "skew_corrected_degrees": None,
        "crop_offset": {"x": crop_left, "y": crop_top},
        "post_crop_size": list(post_crop_size),
        "resolution_scale": resized.width / post_crop_size[0],
        "fallback_used": False,
    }
    transform = CoordinateTransform.from_preprocessing(
        rendered_size=(w, h), preprocessing_metadata=metadata, dpi=None, output_space=CoordinateSpace.IMAGE_PIXELS
    )
    recovered_x, recovered_y = transform.point_to_original(found_x, found_y)
    assert recovered_x == pytest.approx(mx, abs=2.0)
    assert recovered_y == pytest.approx(my, abs=2.0)


def test_from_preprocessing_with_none_metadata_is_identity_plus_dpi():
    t = CoordinateTransform.from_preprocessing(
        rendered_size=(1000, 1400), preprocessing_metadata=None, dpi=200, output_space=CoordinateSpace.PDF_POINTS
    )
    assert t.orientation_degrees == 0.0
    assert t.resolution_scale == 1.0
    assert t.dpi == 200


def test_from_preprocessing_fallback_used_ignores_metadata_but_keeps_dpi():
    """If preprocessing itself fell back to the original image
    (`OCRPreprocessingResult.fallback_used`), OCR ran against the
    RENDERED image directly -- any operations dict from a failed attempt
    must be ignored, not partially applied."""
    metadata = {
        "fallback_used": True,
        "orientation_corrected_degrees": 90,  # must be ignored
        "resolution_scale": 3.0,  # must be ignored
    }
    t = CoordinateTransform.from_preprocessing(
        rendered_size=(1000, 1400), preprocessing_metadata=metadata, dpi=200, output_space=CoordinateSpace.PDF_POINTS
    )
    assert t.orientation_degrees == 0.0
    assert t.resolution_scale == 1.0


def test_bbox_to_original_sets_image_ref_only_for_image_pixels_space():
    t = CoordinateTransform(
        rendered_width=100, rendered_height=100, output_space=CoordinateSpace.PDF_POINTS, image_ref="page-7.png"
    )
    out = t.bbox_to_original(BoundingBox(x0=0, y0=0, x1=10, y1=10))
    assert out.image_ref is None  # PDF_POINTS is page-level, not tied to one raster

    t2 = CoordinateTransform(
        rendered_width=100, rendered_height=100, output_space=CoordinateSpace.IMAGE_PIXELS, image_ref="upload.png"
    )
    out2 = t2.bbox_to_original(BoundingBox(x0=0, y0=0, x1=10, y1=10))
    assert out2.image_ref == "upload.png"
