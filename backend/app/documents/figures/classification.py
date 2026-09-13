"""Heuristic figure/chart type classification.

**This is a heuristic pixel-statistics classifier, not a trained model.**
No image-classification model is used (see docs — Veridoc has no ML
dependency for this beyond Pillow/numpy, matching the project's existing
"heuristic, honestly-scored" pattern for OCR table reconstruction). Every
result carries a `confidence` in [0, 1] that reflects how cleanly the
image matched ONE category's signature versus looking ambiguous between
several — never a fabricated high score for a genuinely uncertain image.
`VisualObjectType.UNKNOWN_VISUAL` with low confidence is the correct, honest
output for an image these heuristics can't confidently place, and is
expected to be common — this is explicitly not claimed as production
-grade classification accuracy (see the Phase 4 evaluation module,
`app/evaluation/phase4_metrics.py`, for how classification accuracy is
actually measured against a small hand-labeled set rather than assumed).

Signals used (all cheap, all computed from a single decoded image, no
network/model call):

- **Color palette size** — a photo has thousands of distinct colors
  (continuous tone); a diagram/chart/screenshot has a small, discrete
  palette (flat-filled shapes, UI chrome).
- **Edge density** — a simple horizontal+vertical gradient-magnitude
  proxy (no scipy/cv2 dependency). Charts and diagrams have many sharp,
  mostly axis-aligned edges (bars, gridlines, node borders); photos have
  comparatively few strong edges relative to their smooth gradients.
- **Aspect ratio + straight-line fraction** — a screenshot is very often
  a "typical UI" aspect ratio (close to 4:3, 16:9, or a tall mobile
  ratio) with a LOT of perfectly horizontal/vertical edges (window
  chrome, text lines, form fields) even more than a chart does.
- **Grayscale-ness** — many scanned diagrams/illustrations are near
  -grayscale even when saved as RGB; a genuine photo almost never is.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.documents.models import VisualObjectType


@dataclass(frozen=True)
class ClassificationResult:
    object_type: VisualObjectType
    confidence: float
    signals: dict  # raw feature values, for debugging/evaluation - not a promise of stability


def _load_rgb_array(image_bytes: bytes, max_side: int = 512):
    import numpy as np
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return np.asarray(img, dtype=np.int16)


def _distinct_color_fraction(arr) -> float:
    import numpy as np

    flat = arr.reshape(-1, 3).astype(np.int64)
    # Quantize to reduce JPEG-noise-induced near-duplicate colors from
    # inflating the count for what's visually a flat-color image.
    quantized = (flat // 8) * 8
    packed = quantized[:, 0] * 65536 + quantized[:, 1] * 256 + quantized[:, 2]
    n_distinct = len(np.unique(packed))
    return min(1.0, n_distinct / max(1, flat.shape[0]))


def _edge_density(arr) -> float:
    import numpy as np

    gray = arr.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_pixels = (dx > 20).sum() + (dy > 20).sum()
    total = dx.size + dy.size
    return edge_pixels / max(1, total)


def _straight_line_fraction(arr) -> float:
    """Fraction of rows/columns that are near-uniform in gradient (a
    strong horizontal or vertical rule/edge runs the full width/height) —
    high for UI chrome, bar charts, tables rendered as images; low for
    organic photo content.
    """
    import numpy as np

    gray = arr.mean(axis=2)
    row_var = gray.std(axis=1)
    col_var = gray.std(axis=0)
    flat_rows = (row_var < 5).sum() / max(1, len(row_var))
    flat_cols = (col_var < 5).sum() / max(1, len(col_var))
    return max(flat_rows, flat_cols)


def _grayscale_fraction(arr) -> float:
    import numpy as np

    channel_spread = arr.max(axis=2) - arr.min(axis=2)
    return (channel_spread < 12).mean()


def classify_image(image_bytes: bytes) -> ClassificationResult:
    try:
        arr = _load_rgb_array(image_bytes)
    except Exception:
        return ClassificationResult(VisualObjectType.UNKNOWN_VISUAL, 0.0, {})

    if arr.size == 0:
        return ClassificationResult(VisualObjectType.UNKNOWN_VISUAL, 0.0, {})

    distinct = _distinct_color_fraction(arr)
    edges = _edge_density(arr)
    lines = _straight_line_fraction(arr)
    grayscale = _grayscale_fraction(arr)
    signals = {
        "distinct_color_fraction": round(distinct, 4),
        "edge_density": round(edges, 4),
        "straight_line_fraction": round(lines, 4),
        "grayscale_fraction": round(grayscale, 4),
    }

    scores: dict[VisualObjectType, float] = {
        # Photo: high color diversity, few hard edges relative to that
        # diversity, low straight-line fraction.
        VisualObjectType.PHOTO: max(0.0, distinct * 2.5 - edges * 1.5 - lines * 1.5),
        # Screenshot: very high straight-line fraction (UI chrome/text
        # rows) AND a fairly small color palette (flat UI fills).
        VisualObjectType.SCREENSHOT: max(0.0, lines * 2.0 - distinct * 1.0),
        # Chart: high edge density AND moderate-to-high straight-line
        # fraction (axes, gridlines, bars) but not AS line-dominated as a
        # screenshot, plus a small palette (flat data-series colors).
        VisualObjectType.CHART: max(0.0, edges * 1.8 + lines * 0.6 - distinct * 1.2),
        # Diagram: high edges, low color diversity, often grayscale
        # (line-art / flowcharts), but lower straight-line-fraction than
        # a chart/screenshot (curved connectors, varied shapes).
        VisualObjectType.DIAGRAM: max(0.0, edges * 1.2 + grayscale * 0.8 - distinct * 1.0 - lines * 0.4),
        # Illustration: low edges, low-to-moderate color diversity,
        # everything else roughly in between -- the "none of the above
        # signatures fired strongly" bucket, scored low on purpose so it
        # only wins when nothing else clears a real signal.
        VisualObjectType.ILLUSTRATION: max(0.0, 0.15 - edges * 0.3),
    }

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    total = sum(scores.values()) or 1.0
    # Confidence = how much this type "won" relative to the field, not
    # the raw score itself (which has no natural upper bound) -- an image
    # where every signature scores similarly low/high is genuinely
    # ambiguous and should report low confidence.
    confidence = round(min(1.0, best_score / total), 3) if best_score > 0 else 0.0

    if confidence < 0.35:
        return ClassificationResult(VisualObjectType.UNKNOWN_VISUAL, confidence, signals)
    return ClassificationResult(best_type, confidence, signals)