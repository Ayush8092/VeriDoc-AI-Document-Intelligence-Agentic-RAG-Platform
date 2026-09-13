"""Caption detection and figure/caption association.

Two-step process:

1. `find_caption_blocks` — scan a document's text `Block`s for ones that
   LOOK like a caption: start with "Figure", "Fig.", "Chart", "Table",
   or "Image" followed by a number and a separator (spec section 9,
   "caption detection"). This is a text-pattern match only — it doesn't
   know yet which figure (if any) a matched block belongs to.
2. `associate_captions` — for each extracted `VisualObject`, find the
   BEST matching caption block: same page, closest vertically (PDF —
   real bbox proximity), or nearest by document order (DOCX — no bbox).
   A caption block can only be consumed once (the closest unclaimed
   match wins) so two figures on the same page don't both claim the same
   caption text.

Never invents a caption: a figure with no plausible nearby caption block
keeps `caption=None` / `caption_confidence=0.0` (see `VisualObject`'s
docstring, "never a fabricated placeholder").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.documents.models import Block, BoundingBox, VisualObjectType

_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|chart|image|photo|illustration|diagram)\s*\d+\s*[:.\-–—]?\s*",
    re.IGNORECASE,
)

# How far below (or, more rarely, above) a figure a caption block may sit,
# in PDF points, to be considered a plausible match at all — beyond this,
# it's almost certainly captioning something else on a dense page.
_MAX_CAPTION_DISTANCE_PT = 60.0


@dataclass(frozen=True)
class CaptionMatch:
    block_index: int
    text: str
    confidence: float


def looks_like_caption(text: str) -> bool:
    return bool(_CAPTION_RE.match(text.strip()))


def find_caption_blocks(blocks: list[Block]) -> list[int]:
    """Indices into `blocks` of every block whose text matches the
    caption pattern. Order-preserving; a block is a candidate regardless
    of its `block_type` (a caption is very often mis-parsed as a plain
    PARAGRAPH, which is fine — this function only looks at the text).
    """
    return [i for i, b in enumerate(blocks) if looks_like_caption(b.text)]


def _vertical_distance(fig_bbox: BoundingBox, cap_bbox: BoundingBox) -> float | None:
    """Signed-aware distance: caption below the figure (common case)
    measured top-of-caption minus bottom-of-figure; caption above,
    bottom-of-figure... i.e. just the gap between the two boxes
    vertically, whichever side. Returns None if the boxes overlap
    horizontally so little they're unlikely to relate at all.
    """
    h_overlap = min(fig_bbox.x1, cap_bbox.x1) - max(fig_bbox.x0, cap_bbox.x0)
    min_width = min(fig_bbox.x1 - fig_bbox.x0, cap_bbox.x1 - cap_bbox.x0)
    if min_width > 0 and h_overlap / min_width < 0.15:
        return None
    if cap_bbox.y0 >= fig_bbox.y1:
        return cap_bbox.y0 - fig_bbox.y1  # caption below
    if cap_bbox.y1 <= fig_bbox.y0:
        return fig_bbox.y0 - cap_bbox.y1  # caption above
    return 0.0  # overlapping vertically too — treat as touching


def associate_captions(visual_blocks: list[Block], all_blocks: list[Block]) -> dict[str, CaptionMatch]:
    """Returns `{visual_object_id: CaptionMatch}` for every visual that
    found a plausible caption. `visual_blocks` must be the subset of
    `all_blocks` whose `.visual` is set (figures/charts already
    extracted); `all_blocks` is the FULL document block list captions are
    searched in.
    """
    caption_indices = find_caption_blocks(all_blocks)
    claimed: set[int] = set()
    result: dict[str, CaptionMatch] = {}

    # Sort visuals by page then position so an earlier figure on a page
    # gets first claim on the nearest caption when two figures are close
    # together (deterministic, order-stable outcome).
    ordered = sorted(visual_blocks, key=lambda b: (b.page_number or 0, b.order))

    for vblock in ordered:
        visual = vblock.visual
        if visual is None:
            continue

        best_idx: int | None = None
        best_score = float("inf")

        for cidx in caption_indices:
            if cidx in claimed:
                continue
            cblock = all_blocks[cidx]
            if cblock.page_number != vblock.page_number:
                continue

            if vblock.bbox is not None and cblock.bbox is not None:
                dist = _vertical_distance(vblock.bbox, cblock.bbox)
                if dist is None or dist > _MAX_CAPTION_DISTANCE_PT:
                    continue
                score = dist
            else:
                # No bbox on one or both sides (DOCX, or an OCR'd figure
                # without geometry) -- fall back to document-order
                # adjacency: only the IMMEDIATELY next/previous block is
                # eligible, scored by that order distance.
                order_dist = abs(cblock.order - vblock.order)
                if order_dist > 1:
                    continue
                score = 1000 + order_dist  # always worse than any real geometric match

            if score < best_score:
                best_score = score
                best_idx = cidx

        if best_idx is not None:
            claimed.add(best_idx)
            cap_text = all_blocks[best_idx].text.strip()
            # Confidence reflects match quality: a close geometric match
            # (small gap) scores high; an order-only fallback match scores
            # lower, since it has no geometric confirmation at all.
            confidence = 0.95 if best_score < 1000 else 0.55
            confidence = round(max(0.3, confidence - (min(best_score, 60) / 60) * 0.25), 3) if best_score < 1000 else confidence
            result[visual.object_id] = CaptionMatch(block_index=best_idx, text=cap_text, confidence=confidence)

    return result