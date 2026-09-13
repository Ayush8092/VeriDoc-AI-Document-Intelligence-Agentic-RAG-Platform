// Phase 4.1(5): the ONE place in the frontend allowed to know how to map
// a backend BoundingBox onto a rendered page image's pixels. Everywhere
// else, a bbox/coordinate_space pair is passed around opaquely — see
// lib/types.ts's CoordinateSpace docstring: "NEVER interpret a bbox
// without checking this."
//
// Two coordinate spaces exist (backend app/documents/models.py::
// CoordinateSpace), and they are NOT interchangeable:
//
//   PDF_POINTS   — origin top-left, units are PDF points (1/72 inch),
//                  scale is the PDF PAGE's own point dimensions
//                  (typically ~612x792 for US Letter), independent of
//                  whatever DPI a page was later rasterized at.
//   IMAGE_PIXELS — origin top-left, units are pixels in the SOURCE
//                  image's own native resolution (a standalone
//                  image upload, or an already-rasterized scanned
//                  page) — independent of PDF points entirely.
//
// A rendered page image (backend GET /documents/{id}/pages/{n}/image)
// is always some PIXEL size — `renderedWidth`/`renderedHeight`, read
// from that response's X-Rendered-Width/-Height headers (see
// lib/api.ts::getPageImage). Projecting a bbox onto that image means:
//
//   PDF_POINTS   -> multiply by (renderedWidth / pdfPageWidthPoints)
//                   Requires knowing the PDF page's own point
//                   dimensions — NOT assumed here (see
//                   `projectBBox`'s `pdfPageSize` parameter); guessing a
//                   fixed page size (e.g. always-Letter) would silently
//                   misplace every highlight on an A4 or custom-size
//                   PDF, which is worse than not drawing one at all.
//   IMAGE_PIXELS -> multiply by (renderedWidth / originalImageWidthPx)
//                   — a scanned page rendered at a different DPI than
//                   OCR ran at is a real, expected case (OCR runs at
//                   `Settings.ocr_dpi`; the viewer may request a
//                   different `dpi` query param), so this scale factor
//                   is virtually never exactly 1.

import type { BoundingBox, CoordinateSpace } from "./types";

export interface PixelRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ProjectionContext {
  renderedWidth: number;
  renderedHeight: number;
  // The bbox's own coordinate system's native page/image size —
  // REQUIRED to scale a PDF_POINTS or IMAGE_PIXELS bbox onto the
  // rendered pixel size above. `null` when unknown (caller couldn't
  // determine it) — `projectBBox` returns `null` rather than guess.
  nativeWidth: number | null;
  nativeHeight: number | null;
}

/**
 * Project a backend `BoundingBox` onto a rendered page image's pixel
 * rectangle, respecting its `coordinate_space`. Returns `null` (never a
 * best-guess rectangle) when the space is `"unspecified"` or the native
 * page/image size needed to scale it isn't available — an absent
 * highlight is honest; a wrongly-placed one is actively misleading for
 * a "source grounding" product whose entire premise is that citations
 * are checkable.
 */
export function projectBBox(bbox: BoundingBox, ctx: ProjectionContext): PixelRect | null {
  if (bbox.coordinate_space === "unspecified") return null;
  if (!ctx.nativeWidth || !ctx.nativeHeight || !ctx.renderedWidth || !ctx.renderedHeight) return null;

  const scaleX = ctx.renderedWidth / ctx.nativeWidth;
  const scaleY = ctx.renderedHeight / ctx.nativeHeight;

  return {
    left: bbox.x0 * scaleX,
    top: bbox.y0 * scaleY,
    width: (bbox.x1 - bbox.x0) * scaleX,
    height: (bbox.y1 - bbox.y0) * scaleY,
  };
}

/** Human-readable label for a coordinate space, for debug/inspector UI —
 * never used to DECIDE how to project (that's `projectBBox` only). */
export function coordinateSpaceLabel(space: CoordinateSpace | null | undefined): string {
  switch (space) {
    case "pdf_points":
      return "PDF points";
    case "image_pixels":
      return "Image pixels";
    case "unspecified":
      return "Unspecified";
    default:
      return "—";
  }
}