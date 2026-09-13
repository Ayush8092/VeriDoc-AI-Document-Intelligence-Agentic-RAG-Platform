"""OCR coordinate transformation (Phase 4 Issues 2/3).

OCR preprocessing (`app.documents.visual.ocr_preprocessing`) can rotate
(orientation correction, deskew), crop (border cleanup), and resize
(resolution normalization) the image it hands to Tesseract. Tesseract's
own word/line bounding boxes are then in THAT processed image's pixel
grid — which can be a different size, a different orientation, and a
different origin than the image OCR preprocessing originally started
from. Using those coordinates directly as if they were coordinates on
the original page (for source-viewer highlighting, citation click-
through, table-cell provenance, a page overlay, ...) silently draws the
box in the wrong place.

This module is the ONE place that undoes exactly what preprocessing did,
in reverse order, using the same numeric operations
`app.documents.visual.ocr_preprocessing.preprocess_image` recorded (see
its `applied` dict / `OCRPreprocessingResult.to_metadata_dict()`):

    processed-image pixels
      -> [undo resolution normalize]  (divide by resolution_scale)
      -> [undo border crop]           (add back crop_offset_x/y)
      -> [undo deskew]                (inverse rotation about center, same canvas size)
      -> [undo orientation correction] (inverse rotation about center, canvas SHRINKS back)
      -> rendered-image pixels        (the image preprocessing was originally called with)
      -> [PDF-page path only: divide by dpi/72] -> PDF points
      -> canonical page/document coordinate space

Two canonical output spaces (see `app.documents.models.CoordinateSpace`):

- `CoordinateSpace.PDF_POINTS` — for a scanned PDF page. The rendered
  image (`render_pdf_pages`'s output, before preprocessing) came from
  the PDF page at a known DPI, so "rendered pixels -> PDF points" is a
  simple, exact `72 / dpi` scale (that's the standard PDF/points-per-inch
  relationship). This makes an OCR'd page's block bboxes land in the
  SAME coordinate system pdfplumber already uses for native pages'
  bboxes (`app/documents/parsers/pdf_parser.py::_parse_native_page`) — a
  document mixing native and scanned pages gets ONE consistent space,
  matching `PageInfo.width`/`height` (always `page.width`/`page.height`,
  i.e. PDF points, regardless of whether a given page ended up native or
  OCR'd — see `PageInfo`'s docstring).
- `CoordinateSpace.IMAGE_PIXELS` — for a standalone image upload. There's
  no PDF page to convert to; the canonical space is just the pixel grid
  of the ORIGINAL uploaded file (`dpi=None` — no render-DPI conversion
  step).

The rotation-inversion math (`_invert_rotation_point`) implements the
exact inverse of `PIL.Image.rotate(angle, expand=...)`, empirically
verified against PIL's own output (see
`tests/test_coordinate_transform.py`) rather than assumed from a
textbook formula's sign convention, since PIL's counter-clockwise-in-
display-terms rotation is easy to get a sign wrong on when translating
to (x-right, y-down) pixel-coordinate rotation matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.documents.models import BoundingBox, CoordinateSpace


def _invert_rotation_point(
    x: float,
    y: float,
    angle_degrees: float,
    src_size: tuple[float, float],
    dst_size: tuple[float, float],
) -> tuple[float, float]:
    """Given a point `(x, y)` in the OUTPUT canvas of a
    `PIL.Image.rotate(angle_degrees, expand=...)` call whose INPUT canvas
    was `src_size` and whose OUTPUT canvas was `dst_size`, return the
    corresponding point in the input canvas.

    Verified empirically (see module docstring) that PIL's forward
    mapping is:
        x' = (x-cx)*cosθ + (y-cy)*sinθ + cx'
        y' = -(x-cx)*sinθ + (y-cy)*cosθ + cy'
    where `(cx, cy)` is the input canvas center, `(cx', cy')` is the
    output canvas center, and θ = `angle_degrees` in radians — for
    `expand=False`, `dst_size == src_size` so `(cx',cy') == (cx,cy)`; for
    `expand=True`, `dst_size` is PIL's own enlarged bounding-box canvas.
    Since rotation matrices are orthogonal, the inverse is the same
    matrix transposed (equivalently, rotate by `-angle_degrees`):
        a = cosθ*(x'-cx') - sinθ*(y'-cy')
        b = sinθ*(x'-cx') + cosθ*(y'-cy')
        (x, y) = (a+cx, b+cy)
    """
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    cx_src, cy_src = src_w / 2.0, src_h / 2.0
    cx_dst, cy_dst = dst_w / 2.0, dst_h / 2.0

    a = cos_t * (x - cx_dst) - sin_t * (y - cy_dst)
    b = sin_t * (x - cx_dst) + cos_t * (y - cy_dst)
    return a + cx_src, b + cy_src


@dataclass(frozen=True)
class CoordinateTransform:
    """Maps a bbox in OCR-PROCESSED-image pixel coordinates back to a
    document's canonical coordinate space. Construct via
    `CoordinateTransform.from_preprocessing` — the fields below mirror
    exactly what `preprocess_image`'s `applied` dict records, in the
    order those operations actually ran (orientation -> skew -> crop ->
    resize), so a `CoordinateTransform` built from a real
    `OCRPreprocessingResult` is an exact inverse, not an approximation
    (module docstring, "undoes exactly what preprocessing did").

    An all-default instance (no orientation/skew/crop, scale 1.0, no
    dpi) is the identity transform — safe to construct even when
    preprocessing didn't run at all (`preprocess=False`) or fell back to
    the original image (`OCRPreprocessingResult.fallback_used`).
    """

    rendered_width: float
    rendered_height: float
    orientation_degrees: float = 0.0
    orientation_applied_size: tuple[float, float] | None = None  # canvas size right after orientation rotation
    skew_degrees: float = 0.0
    crop_offset_x: float = 0.0
    crop_offset_y: float = 0.0
    resolution_scale: float = 1.0
    dpi: float | None = None  # None => output stays in rendered-image pixel space (standalone image upload)
    output_space: CoordinateSpace = CoordinateSpace.IMAGE_PIXELS
    image_ref: str | None = None  # only meaningful when output_space == IMAGE_PIXELS

    def point_to_original(self, x: float, y: float) -> tuple[float, float]:
        """Map one point from processed-image pixels to this transform's
        canonical output space. `bbox_to_original` (below) is almost
        always what a caller actually wants — a rotation can tilt a
        box's edges, so only mapping opposite corners under-covers the
        true extent; this is exposed separately mainly for testing.
        """
        # 1. undo resolution normalization
        if self.resolution_scale and self.resolution_scale != 1.0:
            x = x / self.resolution_scale
            y = y / self.resolution_scale

        # 2. undo border crop (crop happened in the post-orientation/skew
        # canvas; the offset was measured there, so add it straight back)
        x = x + self.crop_offset_x
        y = y + self.crop_offset_y

        # 3. undo skew (expand=False -> same canvas before and after)
        base_size = self.orientation_applied_size or (self.rendered_width, self.rendered_height)
        if self.skew_degrees:
            x, y = _invert_rotation_point(x, y, self.skew_degrees, base_size, base_size)

        # 4. undo orientation correction (expand=True -> canvas shrinks
        # back down to the original rendered image's size)
        if self.orientation_degrees:
            x, y = _invert_rotation_point(
                x, y, self.orientation_degrees, (self.rendered_width, self.rendered_height), base_size
            )

        # 5. rendered-image pixels -> PDF points, only for the PDF path
        if self.dpi:
            x = x * 72.0 / self.dpi
            y = y * 72.0 / self.dpi

        return x, y

    def bbox_to_original(self, bbox: BoundingBox) -> BoundingBox:
        """Map a processed-image-pixel bbox to this transform's
        canonical output space. Transforms all 4 corners (not just the
        two opposite ones) and takes their enclosing axis-aligned box —
        required for correctness under rotation, where a box's edges are
        no longer axis-aligned in the source space.
        """
        corners = [
            (bbox.x0, bbox.y0),
            (bbox.x1, bbox.y0),
            (bbox.x1, bbox.y1),
            (bbox.x0, bbox.y1),
        ]
        mapped = [self.point_to_original(x, y) for x, y in corners]
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        return BoundingBox(
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
            coordinate_space=self.output_space,
            image_ref=self.image_ref if self.output_space == CoordinateSpace.IMAGE_PIXELS else None,
        )

    @classmethod
    def from_preprocessing(
        cls,
        rendered_size: tuple[float, float],
        preprocessing_metadata: dict | None,
        *,
        dpi: float | None,
        output_space: CoordinateSpace,
        image_ref: str | None = None,
    ) -> "CoordinateTransform":
        """Build from the metadata dict `preprocess_and_ocr` returns
        (`OCRPreprocessingResult.to_metadata_dict()`), or from `None`
        (`preprocess=False`, or nothing to undo) — in which case this is
        the identity transform except for the DPI conversion (PDF path).

        `rendered_size` is the size of the image `preprocess_and_ocr` was
        CALLED with (`render_pdf_pages`'s output for a PDF page, or the
        original upload's own size for a standalone image) — i.e. the
        space this transform maps INTO before the final DPI step.
        """
        if not preprocessing_metadata or preprocessing_metadata.get("fallback_used"):
            # No preprocessing ran (disabled, or it fell back to the
            # original image on error — see `run_preprocessing`) -> the
            # OCR'd image IS the rendered image; only the DPI step (if
            # any) still applies.
            return cls(
                rendered_width=rendered_size[0],
                rendered_height=rendered_size[1],
                dpi=dpi,
                output_space=output_space,
                image_ref=image_ref,
            )

        m = preprocessing_metadata
        crop = m.get("crop_offset") or {"x": 0, "y": 0}
        post_orientation_size = m.get("post_orientation_size")
        return cls(
            rendered_width=rendered_size[0],
            rendered_height=rendered_size[1],
            orientation_degrees=float(m.get("orientation_corrected_degrees") or 0),
            orientation_applied_size=tuple(post_orientation_size) if post_orientation_size else None,
            skew_degrees=float(m.get("skew_corrected_degrees") or 0.0),
            crop_offset_x=float(crop.get("x", 0)),
            crop_offset_y=float(crop.get("y", 0)),
            resolution_scale=float(m.get("resolution_scale") or 1.0),
            dpi=dpi,
            output_space=output_space,
            image_ref=image_ref,
        )
