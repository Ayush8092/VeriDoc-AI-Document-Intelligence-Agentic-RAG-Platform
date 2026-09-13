"""OCR preprocessing (Phase 4.1/4.2, "better OCR" / "preprocessing").

Tesseract's accuracy is sensitive to exactly the kind of degradation real
scanned/photographed documents have: a few degrees of skew from an
imperfect scan or phone photo, low contrast, uneven illumination (a
shadow or glare on one side of a phone photo), a 90/180/270-degree
rotation, and sensor/compression noise. This module measures those
specific problems in a source image (`analyze_image`), decides what's
actually worth correcting (`select_strategy` — never do more than the
image needs, since every operation risks losing information a later step
could have used), and applies the correction (`preprocess_image`).
`run_preprocessing` is the single entry point that ties all three
together with timing and guaranteed graceful fallback — see its
docstring.

Deliberately classical image processing (PIL + numpy, plus Tesseract's
own orientation/script detection since `pytesseract` is already a hard
dependency), not a learned preprocessing model: this sandbox's network
allowlist doesn't include a model hub (no huggingface.co, no model-weight
CDN — see the note in app/rag/cross_encoder.py about the same
constraint), so a "preprocessing model" isn't something that could
actually be downloaded and verified working here. Deskew via
projection-profile variance, orientation via Tesseract OSD, and
binarization via Otsu's method / local adaptive mean are well-established,
deterministic, zero-download techniques that solve the specific problems
that actually occur in this project's corpus (see
data/corpus/06_warehouse_receiving_log_scanned.png) — see each function's
docstring for the algorithm and its real limits.

Every step is individually optional and independently gated by
`select_strategy`'s thresholds, and `preprocess_image` always returns
BOTH the processed image and a metadata dict recording exactly what was
applied and why — nothing here is a silent transformation.

**Shared by both ingestion paths (Phase 4 Issue 1 fix).** Before this
module's `preprocess_and_ocr` existed, `app/documents/parsers/image_parser.py`
called `analyze_image`/`select_strategy`/`preprocess_image` before OCR,
but `app/documents/parsers/pdf_parser.py`'s scanned-page path called
`run_ocr` directly, bypassing preprocessing entirely — a scanned PDF page
got worse OCR than the exact same page saved as a standalone PNG would
have. `preprocess_and_ocr` is now the ONE place that sequences
"preprocess -> OCR" for an already-rendered page image, and both parsers
call it — see their source for the (now-identical) call site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# --- thresholds (tuned for typical 150-300 DPI scans/photos of documents) --

_SKEW_CORRECTION_THRESHOLD_DEGREES = 0.4
_SKEW_SEARCH_RANGE_DEGREES = 5.0
_SKEW_SEARCH_STEP_DEGREES = 0.2
_LOW_CONTRAST_STD_THRESHOLD = 45.0
_NOISE_VARIANCE_THRESHOLD = 900.0
_BINARIZE_MIN_DYNAMIC_RANGE = 60  # skip binarization on already near-uniform images
_ILLUMINATION_VARIANCE_THRESHOLD = 120.0  # cell-mean variance above this => uneven lighting
_ADAPTIVE_GRID = 4  # illumination measured on a 4x4 grid of cells
_ADAPTIVE_WINDOW_FRACTION = 0.06  # local-mean window as a fraction of image width
_LOW_RES_WIDTH_PX = 900
_HIGH_RES_WIDTH_PX = 4000
_TARGET_NORMALIZED_WIDTH_PX = 1800  # ~ a 200-250 DPI US-letter page width
_BORDER_SCAN_MARGIN_FRACTION = 0.12  # only look for a border within the outer 12% of the image
_BORDER_UNIFORMITY_STD_THRESHOLD = 8.0  # a border strip this uniform is a scan artifact, not content


@dataclass(frozen=True)
class ImageAnalysis:
    """What's actually wrong with the image, measured — not guessed."""

    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    dynamic_range: int  # max - min pixel value in the grayscale histogram
    estimated_skew_degrees: float
    noise_variance: float
    orientation_degrees: int = 0  # 0/90/180/270, from Tesseract OSD; 0 if undetermined
    orientation_confidence: float = 0.0  # Tesseract's own OSD confidence, 0 if undetermined
    illumination_variance: float = 0.0  # variance of per-cell mean intensity; high = uneven lighting


@dataclass(frozen=True)
class PreprocessConfig:
    """What `preprocess_image` should actually do, decided from `ImageAnalysis`.

    This is `OCRPreprocessingConfig` in the Phase 4 spec's terminology —
    kept as `PreprocessConfig` (its pre-Phase-4 name) rather than renamed,
    since renaming a class already used across `image_parser.py`,
    `pdf_parser.py`, and their tests is a pure churn risk with no
    behavioral benefit (see this project's "smallest safe extension"
    convention).
    """

    correct_skew: bool = False
    skew_degrees: float = 0.0
    correct_orientation: bool = False
    orientation_degrees: int = 0  # rotation to APPLY (i.e. -1 * detected orientation), 90/180/270
    denoise: bool = False
    enhance_contrast: bool = False
    binarize: bool = False
    use_adaptive_threshold: bool = False  # when True, overrides `binarize`'s global Otsu with local adaptive
    normalize_resolution: bool = False
    target_width: int = _TARGET_NORMALIZED_WIDTH_PX
    sharpen: bool = False
    clean_borders: bool = False


def _grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _estimate_skew_degrees(gray: np.ndarray) -> float:
    """Projection-profile skew estimate.

    For each candidate rotation angle, rotate a (small, for speed)
    grayscale copy and sum pixel intensities row-by-row, producing a
    1-D "projection profile". When text lines are exactly horizontal,
    each line of text produces a sharp peak (dark ink concentrated in a
    narrow band of rows) — so the profile's variance is maximized at the
    TRUE deskew angle and drops off as rotation moves lines out of
    alignment. This is a standard, well-established technique (does not
    require a trained model) and is reliable for the common case this
    project targets: a scan/photo of a mostly-horizontal text document,
    off by a few degrees. It is NOT reliable for documents with little
    text, dense non-text imagery, or skew beyond
    `_SKEW_SEARCH_RANGE_DEGREES` — `select_strategy` only trusts it
    within that range and only applies a correction when the measured
    skew clears `_SKEW_CORRECTION_THRESHOLD_DEGREES`, so a noisy estimate
    on a bad candidate for this technique just doesn't trigger a
    correction rather than actively making things worse.
    """
    # A near-blank/solid-color image has no text structure for this
    # technique to measure at all — worse, rotating it with a fillcolor
    # (see the search loop below) introduces corner artifacts that would
    # themselves look like row-to-row "structure" and produce a spurious
    # non-zero result. Checked on the ORIGINAL, unrotated image, before
    # any rotation is attempted, so this can't be fooled by fill-color
    # artifacts the search itself would otherwise introduce.
    if float(gray.std()) < 1.0:
        return 0.0

    # Downsample for speed — skew estimation doesn't need full resolution,
    # and this runs one rotation per candidate angle.
    small = Image.fromarray(gray.astype(np.uint8)).copy()
    small.thumbnail((400, 400))
    small_arr = np.asarray(small, dtype=np.float64)

    best_angle = 0.0
    best_variance = -1.0
    angle = -_SKEW_SEARCH_RANGE_DEGREES
    small_img = Image.fromarray(small_arr.astype(np.uint8))
    while angle <= _SKEW_SEARCH_RANGE_DEGREES:
        rotated = small_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
        arr = np.asarray(rotated, dtype=np.float64)
        row_sums = arr.sum(axis=1)
        variance = float(np.var(row_sums))
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
        angle += _SKEW_SEARCH_STEP_DEGREES

    return round(best_angle, 2)


def _estimate_noise_variance(gray: np.ndarray) -> float:
    """High-frequency energy via a discrete Laplacian, as a noise proxy.

    A clean scan of text is mostly smooth background with sharp, sparse
    edges at glyph boundaries; sensor/compression noise adds high-
    frequency variation everywhere. Convolving with a Laplacian kernel and
    taking the variance of the result is a standard, cheap noise estimator
    (no ML model): a genuinely clean image has most of its Laplacian
    response near zero (low variance); pervasive per-pixel noise inflates
    it. Computed via `numpy` 2-D correlation — no OpenCV dependency.
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # Manual 2D convolution via striding is overkill for a 3x3 kernel;
    # a straightforward padded-sum is fast enough for typical page sizes
    # and keeps this module dependency-free (no scipy).
    padded = np.pad(gray, 1, mode="edge")
    response = (
        kernel[0, 0] * padded[:-2, :-2] + kernel[0, 1] * padded[:-2, 1:-1] + kernel[0, 2] * padded[:-2, 2:]
        + kernel[1, 0] * padded[1:-1, :-2] + kernel[1, 1] * padded[1:-1, 1:-1] + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 0] * padded[2:, :-2] + kernel[2, 1] * padded[2:, 1:-1] + kernel[2, 2] * padded[2:, 2:]
    )
    return float(np.var(response))


def _estimate_orientation(image: Image.Image) -> tuple[int, float]:
    """Detect a 0/90/180/270-degree page rotation via Tesseract's own
    orientation-and-script-detection (OSD) mode, NOT the projection-
    profile skew estimator above — skew (a couple of degrees from an
    imperfect scan) and orientation (a whole page rotated 90/180/270,
    e.g. a phone photo taken sideways) are different problems needing
    different signals. OSD is Tesseract's own text-orientation classifier
    (`tesseract --psm 0`), which is a real, well-established technique
    distinct from — and more reliable for this specific problem than —
    the projection-profile heuristic used for sub-5-degree skew.

    Returns `(degrees, confidence)` where `degrees` is the rotation
    ALREADY PRESENT in the image (Tesseract's own convention) — the
    correction to apply is `-degrees`. Returns `(0, 0.0)` if OSD can't
    determine orientation (common on sparse-text or table-only images;
    Tesseract itself raises in that case), which is treated as "no
    rotation detected", not "definitely upright" — `select_strategy` only
    acts on this when Tesseract itself reports it confidently.
    """
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        degrees = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0.0))
        return degrees, confidence
    except Exception:
        return 0, 0.0


def _illumination_variance(gray: np.ndarray) -> float:
    """Variance of per-cell mean intensity across an `_ADAPTIVE_GRID` x
    `_ADAPTIVE_GRID` grid of regions. A uniformly-lit scan has roughly the
    same average brightness in every region; a phone photo with a shadow
    on one side or a glare spot has one or two regions noticeably
    brighter/darker than the rest, which is exactly what this variance
    picks up. This is what `select_strategy` uses to prefer LOCAL adaptive
    thresholding over Otsu's single GLOBAL threshold — Otsu assumes the
    whole image shares one ink/background split point, which is a poor
    assumption under uneven lighting.
    """
    h, w = gray.shape
    grid = _ADAPTIVE_GRID
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    means = []
    for r in range(grid):
        for c in range(grid):
            cell = gray[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            if cell.size:
                means.append(float(cell.mean()))
    return round(float(np.var(means)), 2) if means else 0.0


def analyze_image(image: Image.Image) -> ImageAnalysis:
    """Measure what's actually wrong with `image` before deciding what to fix."""
    gray = _grayscale_array(image)
    hist_min, hist_max = int(gray.min()), int(gray.max())
    orientation_degrees, orientation_confidence = _estimate_orientation(image)
    return ImageAnalysis(
        width=image.width,
        height=image.height,
        mean_intensity=round(float(gray.mean()), 2),
        std_intensity=round(float(gray.std()), 2),
        dynamic_range=hist_max - hist_min,
        estimated_skew_degrees=_estimate_skew_degrees(gray),
        noise_variance=round(_estimate_noise_variance(gray), 2),
        orientation_degrees=orientation_degrees,
        orientation_confidence=orientation_confidence,
        illumination_variance=_illumination_variance(gray),
    )


# Tesseract's own OSD confidence is unbounded in principle but in practice
# a confident, correct detection reliably scores well above this on real
# scans/photos; below it, a wrong OSD guess is common enough that acting on
# it would do more harm than the rotation itself.
_ORIENTATION_MIN_CONFIDENCE = 1.0


def select_strategy(analysis: ImageAnalysis) -> PreprocessConfig:
    """Decide which corrections `analysis` actually calls for.

    Every threshold check is a real measurement comparison, not a fixed
    "always do X" — a clean, already-high-contrast, unskewed image gets an
    all-`False` config and `preprocess_image` becomes a no-op (returns the
    original image), which matters because every one of these operations
    is lossy and should only run when it's actually solving a real,
    measured problem.
    """
    prefer_adaptive = analysis.illumination_variance >= _ILLUMINATION_VARIANCE_THRESHOLD
    needs_binarize = analysis.dynamic_range >= _BINARIZE_MIN_DYNAMIC_RANGE
    is_noisy = analysis.noise_variance >= _NOISE_VARIANCE_THRESHOLD
    is_low_contrast = analysis.std_intensity <= _LOW_CONTRAST_STD_THRESHOLD

    orientation_degrees = analysis.orientation_degrees
    correct_orientation = (
        orientation_degrees in (90, 180, 270) and analysis.orientation_confidence >= _ORIENTATION_MIN_CONFIDENCE
    )

    return PreprocessConfig(
        correct_skew=abs(analysis.estimated_skew_degrees) >= _SKEW_CORRECTION_THRESHOLD_DEGREES,
        skew_degrees=analysis.estimated_skew_degrees,
        correct_orientation=correct_orientation,
        orientation_degrees=orientation_degrees if correct_orientation else 0,
        denoise=is_noisy,
        enhance_contrast=is_low_contrast,
        binarize=needs_binarize and not prefer_adaptive,
        use_adaptive_threshold=needs_binarize and prefer_adaptive,
        normalize_resolution=analysis.width < _LOW_RES_WIDTH_PX or analysis.width > _HIGH_RES_WIDTH_PX,
        target_width=_TARGET_NORMALIZED_WIDTH_PX,
        # Sharpening amplifies noise, so it's only offered as a complement
        # to contrast enhancement on an image that ISN'T already flagged
        # noisy — never combined with denoise (a real, well-known
        # interaction, not an independent third signal).
        sharpen=is_low_contrast and not is_noisy,
        clean_borders=True,  # cheap to check; `preprocess_image` is a no-op if no real border is found
    )


def _otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's method: the grayscale threshold that best separates the
    image into two classes (ink / background) by minimizing within-class
    variance — the standard, parameter-free binarization threshold, not a
    fixed guess like "128".
    """
    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg, weight_bg = 0.0, 0
    best_threshold = 0
    best_variance = -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = t
    return best_threshold


def _adaptive_threshold(gray: np.ndarray, window_fraction: float = _ADAPTIVE_WINDOW_FRACTION) -> np.ndarray:
    """Local (adaptive) mean thresholding: each pixel is binarized against
    the MEAN of its own neighborhood, not one global threshold — the
    correct choice under uneven illumination (see `_illumination_variance`),
    where a single global cut (Otsu) would binarize a shadowed region
    entirely to black or a glare region entirely to white.

    Implemented via a box filter (uniform local mean) computed through a
    2-D integral image (summed-area table), giving an O(H*W) local mean at
    any window size without a per-pixel Python loop or a scipy/OpenCV
    dependency — the same complexity class as a real adaptive-threshold
    implementation, just hand-rolled with numpy's cumulative sum.
    """
    h, w = gray.shape
    window = max(3, int(round(min(h, w) * window_fraction)))
    if window % 2 == 0:
        window += 1
    half = window // 2

    padded = np.pad(gray, half, mode="edge")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")  # so integral[0,:] and [:,0] are 0

    ph, pw = padded.shape
    # For each original pixel (r, c), sum over padded[r:r+window, c:c+window].
    r0, c0 = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r1, c1 = r0 + window, c0 + window
    box_sum = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    local_mean = box_sum / (window * window)

    # A small constant bias keeps light-gray background from tipping over
    # into "foreground" purely from local-mean noise near a flat region.
    bias = 7.0
    return np.where(gray >= (local_mean - bias), 255, 0).astype(np.uint8)


def _clean_borders(gray: np.ndarray) -> tuple[np.ndarray, bool, int, int]:
    """Crop away a uniform (near-white or near-black) margin from each
    edge, within the outer `_BORDER_SCAN_MARGIN_FRACTION` of the image —
    the classic "black frame" or "scanner bed edge" artifact from a flatbed
    scan or a photo that includes a bit of the surface around the page.
    Only crops a strip whose row/column standard deviation is below
    `_BORDER_UNIFORMITY_STD_THRESHOLD` (i.e. actually featureless — real
    page content, even a mostly-blank margin with a page-number or ruled
    line, has more variation than a scanner-bed artifact does), so normal
    page whitespace is never mistaken for a border to crop.

    Returns `(array, changed, crop_left, crop_top)` — `changed=False` means
    no real border was found and `array` is the input unchanged (safe
    no-op); `crop_left`/`crop_top` are the pixel offsets that were
    trimmed from each edge (0 when `changed=False`), needed by
    `app.documents.visual.coordinates.CoordinateTransform` (Phase 4 Issue
    2) to map a bbox found in the cropped image back to a bbox in the
    pre-crop image.
    """
    h, w = gray.shape
    max_top = int(h * _BORDER_SCAN_MARGIN_FRACTION)
    max_bottom = int(h * _BORDER_SCAN_MARGIN_FRACTION)
    max_left = int(w * _BORDER_SCAN_MARGIN_FRACTION)
    max_right = int(w * _BORDER_SCAN_MARGIN_FRACTION)

    top = 0
    while top < max_top and float(gray[top, :].std()) < _BORDER_UNIFORMITY_STD_THRESHOLD:
        top += 1
    bottom = h
    while (h - bottom) < max_bottom and float(gray[bottom - 1, :].std()) < _BORDER_UNIFORMITY_STD_THRESHOLD:
        bottom -= 1
    left = 0
    while left < max_left and float(gray[:, left].std()) < _BORDER_UNIFORMITY_STD_THRESHOLD:
        left += 1
    right = w
    while (w - right) < max_right and float(gray[:, right - 1].std()) < _BORDER_UNIFORMITY_STD_THRESHOLD:
        right -= 1

    if top == 0 and bottom == h and left == 0 and right == w:
        return gray, False, 0, 0
    if bottom - top < h * 0.5 or right - left < w * 0.5:
        # Refuse to crop away more than half the page in either dimension —
        # a genuine border artifact is a thin strip, not most of the image;
        # this guards against a pathological all-uniform (e.g. blank) page.
        return gray, False, 0, 0
    return gray[top:bottom, left:right], True, left, top


@dataclass(frozen=True)
class OCRPreprocessingResult:
    """Everything Phase 4.1 asks a preprocessing result to carry: the
    processed image, exactly which operations ran (`operations`), quality
    signals from `ImageAnalysis` (`quality_signals`), and whether/why this
    fell back to the original, unprocessed image (`fallback_used`,
    `fallback_reason`).
    """

    image: Image.Image
    analysis: ImageAnalysis
    config: PreprocessConfig
    operations: dict
    quality_signals: dict
    fallback_used: bool
    fallback_reason: str | None
    processing_time_ms: float

    def to_metadata_dict(self) -> dict:
        """The flat dict shape `PageInfo.ocr_preprocessing` / `Block`
        metadata expects — same shape `preprocess_image` alone used to
        return, plus the new Phase 4.1 fields, so existing consumers
        (`app/schemas/documents.py`, the frontend) keep working unchanged.
        """
        return {
            **self.operations,
            "quality_signals": self.quality_signals,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "processing_time_ms": self.processing_time_ms,
        }


def preprocess_image(image: Image.Image, config: PreprocessConfig) -> tuple[Image.Image, dict]:
    """Apply exactly the corrections `config` calls for, in a fixed,
    sensible order, and return both the result and a metadata dict
    recording what actually ran.

    Order: grayscale -> orientation (whole 90/180/270 turns, before
    anything angle-sensitive) -> deskew (sub-5-degree fine rotation) ->
    border cleanup (after both rotations, so the border is axis-aligned)
    -> resolution normalize -> denoise -> contrast -> sharpen -> binarize
    (adaptive or global Otsu, never both). Any step whose config flag is
    False is skipped entirely (the image passes through unchanged for
    that step) — this function never applies a correction
    `select_strategy` didn't ask for. Kept as the lower-level "apply a
    given config" function (unchanged signature) — see `run_preprocessing`
    for the higher-level, fallback-guaranteed entry point Phase 4 adds.
    """
    result = image.convert("L")
    applied: dict = {
        "original_size": [image.width, image.height],
        "grayscale": True,
        "orientation_corrected_degrees": None,
        "skew_corrected_degrees": None,
        "border_cleaned": False,
        "resolution_normalized": False,
        "denoised": False,
        "contrast_enhanced": False,
        "sharpened": False,
        "binarized": False,
        "adaptive_threshold": False,
        # Phase 4 Issue 2 fix: exact numeric transform data, not just
        # boolean "did this run" flags — this is what
        # `app.documents.visual.coordinates.CoordinateTransform` needs to
        # map an OCR bbox found in THIS (fully processed) image back to
        # the image `preprocess_image` was originally called with. Every
        # field defaults to the identity value for a step that didn't
        # run, so a transform built from an all-default `applied` dict is
        # itself the identity transform.
        "post_orientation_size": [image.width, image.height],  # canvas size right after orientation (before skew/crop/resize)
        "crop_offset": {"x": 0, "y": 0},   # pixels trimmed from (left, top) by border cleanup, in post-orientation/skew space
        "post_crop_size": [image.width, image.height],          # canvas size right after border crop (before resize)
        "resolution_scale": 1.0,            # processed_width / post_crop_width
    }

    if config.correct_orientation and config.orientation_degrees in (90, 180, 270):
        # PIL rotates counter-clockwise for a positive angle; Tesseract's
        # OSD `rotate` value is how far clockwise the text is rotated FROM
        # upright, so the correction is a counter-clockwise rotation by
        # that same amount — i.e. `rotate(degrees)` directly undoes it.
        result = result.rotate(config.orientation_degrees, resample=Image.BICUBIC, expand=True, fillcolor=255)
        applied["orientation_corrected_degrees"] = config.orientation_degrees
        applied["post_orientation_size"] = [result.width, result.height]

    if config.correct_skew and abs(config.skew_degrees) >= _SKEW_CORRECTION_THRESHOLD_DEGREES:
        result = result.rotate(config.skew_degrees, resample=Image.BICUBIC, expand=False, fillcolor=255)
        applied["skew_corrected_degrees"] = config.skew_degrees

    if config.clean_borders:
        gray_arr = np.asarray(result, dtype=np.uint8)
        cropped, changed, crop_left, crop_top = _clean_borders(gray_arr)
        if changed:
            result = Image.fromarray(cropped)
            applied["border_cleaned"] = True
            applied["crop_offset"] = {"x": crop_left, "y": crop_top}
    applied["post_crop_size"] = [result.width, result.height]

    if config.normalize_resolution and result.width > 0:
        scale = config.target_width / result.width
        if abs(scale - 1.0) >= 0.05:  # skip a normalization that would barely change anything
            new_size = (max(1, round(result.width * scale)), max(1, round(result.height * scale)))
            resample = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
            result = result.resize(new_size, resample=resample)
            applied["resolution_normalized"] = True
            applied["resolution_scale"] = result.width / applied["post_crop_size"][0]

    if config.denoise:
        # A 3x3 median filter removes salt-and-pepper/sensor noise while
        # preserving glyph edges far better than a blur would — the
        # standard first-line denoise for OCR preprocessing pipelines.
        result = result.filter(ImageFilter.MedianFilter(size=3))
        applied["denoised"] = True

    if config.enhance_contrast:
        # Autocontrast stretches the existing intensity range to fill
        # [0, 255] using the image's own histogram (a cutoff clips the
        # extreme 1% of pixels so a few outlier bright/dark pixels don't
        # dominate the stretch) — no arbitrary brightness/contrast
        # constants, it adapts to what's actually in the image.
        result = ImageOps.autocontrast(result, cutoff=1)
        applied["contrast_enhanced"] = True

    if config.sharpen:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        applied["sharpened"] = True

    if config.use_adaptive_threshold:
        gray_arr = np.asarray(result, dtype=np.float64)
        binary_arr = _adaptive_threshold(gray_arr)
        result = Image.fromarray(binary_arr)
        applied["adaptive_threshold"] = True
        applied["binarized"] = True
    elif config.binarize:
        gray_arr = np.asarray(result, dtype=np.float64)
        threshold = _otsu_threshold(gray_arr)
        binary_arr = np.where(gray_arr >= threshold, 255, 0).astype(np.uint8)
        result = Image.fromarray(binary_arr)
        applied["binarized"] = True
        applied["otsu_threshold"] = threshold

    applied["processed_size"] = [result.width, result.height]
    return result, applied


def run_preprocessing(image: Image.Image) -> OCRPreprocessingResult:
    """The Phase 4.1 high-level entry point: analyze -> select strategy ->
    preprocess, timed, with a GUARANTEED graceful fallback to the
    original, unmodified image if any step raises for any reason (a
    malformed image, an unexpected numpy edge case, a missing `tesseract`
    binary breaking orientation detection, ...).

    This is what `preprocess_and_ocr` (below) calls, and is also suitable
    to call directly wherever code wants the full `OCRPreprocessingResult`
    (e.g. `evaluation/document_intelligence/ocr_benchmark.py`) rather than
    just the `(image, metadata)` tuple `preprocess_image` returns.
    """
    started = time.perf_counter()
    try:
        analysis = analyze_image(image)
        config = select_strategy(analysis)
        processed, operations = preprocess_image(image, config)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return OCRPreprocessingResult(
            image=processed,
            analysis=analysis,
            config=config,
            operations=operations,
            quality_signals={
                "std_intensity": analysis.std_intensity,
                "dynamic_range": analysis.dynamic_range,
                "noise_variance": analysis.noise_variance,
                "estimated_skew_degrees": analysis.estimated_skew_degrees,
                "illumination_variance": analysis.illumination_variance,
                "orientation_degrees": analysis.orientation_degrees,
                "orientation_confidence": analysis.orientation_confidence,
            },
            fallback_used=False,
            fallback_reason=None,
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:  # noqa: BLE001 - preprocessing must never break ingestion
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return OCRPreprocessingResult(
            image=image,
            analysis=ImageAnalysis(
                width=image.width,
                height=image.height,
                mean_intensity=0.0,
                std_intensity=0.0,
                dynamic_range=0,
                estimated_skew_degrees=0.0,
                noise_variance=0.0,
            ),
            config=PreprocessConfig(),
            operations={"original_size": [image.width, image.height], "processed_size": [image.width, image.height]},
            quality_signals={},
            fallback_used=True,
            fallback_reason=f"{type(exc).__name__}: {exc}",
            processing_time_ms=elapsed_ms,
        )


def preprocess_and_ocr(
    image: Image.Image,
    language: str = "eng",
    confidence_floor: float = 0.0,
    preprocess: bool = True,
):
    """Preprocess (if `preprocess`) then OCR one already-rendered page
    image — the ONE shared implementation `image_parser.py` (standalone
    PNG/JPG) and `pdf_parser.py` (scanned PDF pages, after
    `render_pdf_pages`) both call, closing the gap where scanned-PDF OCR
    used to bypass preprocessing entirely (Phase 4 Issue 1).

    Returns `(words, ocr_image, preprocessing_metadata)`:
      - `words`: `list[OcrWord]` from `app.documents.ocr.engine.run_ocr`,
        run against the (possibly preprocessed) image.
      - `ocr_image`: the image OCR actually ran against — needed by
        callers that report `width`/`height` in `PageInfo` from the image
        actually processed, not the original.
      - `preprocessing_metadata`: `OCRPreprocessingResult.to_metadata_dict()`,
        or `None` if `preprocess=False`.
    """
    from app.documents.ocr.engine import run_ocr

    ocr_image = image
    preprocessing_metadata: dict | None = None
    if preprocess:
        result = run_preprocessing(image)
        ocr_image = result.image
        preprocessing_metadata = result.to_metadata_dict()

    words = run_ocr(ocr_image, language=language, confidence_floor=confidence_floor)
    return words, ocr_image, preprocessing_metadata