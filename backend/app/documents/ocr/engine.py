"""Tesseract OCR wrapper: image -> words with position + confidence.

Deliberately does NOT do `image -> OCR text -> plain string` (see
docs/ocr.md, "OCR must preserve structure"). `pytesseract.image_to_data`
returns per-word bounding boxes, confidence, and Tesseract's own line/
paragraph/block grouping, which is what everything downstream (line
reconstruction, table detection in app/documents/tables/reconstruct.py)
is built on.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class OcrWord:
    text: str
    left: float
    top: float
    width: float
    height: float
    confidence: float          # 0-100, -1 for non-text regions (dropped by caller)
    block_num: int
    par_num: int
    line_num: int
    word_num: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


@dataclass(frozen=True)
class OcrLine:
    words: tuple[OcrWord, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0 = min(w.left for w in self.words)
        y0 = min(w.top for w in self.words)
        x1 = max(w.left + w.width for w in self.words)
        y1 = max(w.top + w.height for w in self.words)
        return (x0, y0, x1, y1)

    @property
    def mean_confidence(self) -> float:
        confs = [w.confidence for w in self.words if w.confidence >= 0]
        return sum(confs) / len(confs) if confs else 0.0


class OcrError(RuntimeError):
    """Raised when the OCR engine itself fails (missing binary, bad image)."""


def run_ocr(image: Image.Image, language: str = "eng", confidence_floor: float = 0.0) -> list[OcrWord]:
    """Run Tesseract over a PIL image and return word-level results.

    Words below `confidence_floor` (Tesseract confidence, 0-100) are
    dropped — Tesseract emits confidence -1 for structural, non-text
    regions, which are always dropped regardless of the floor.
    """
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover
        raise OcrError("OCR requires the 'pytesseract' package and the tesseract binary") from exc

    try:
        data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        raise OcrError(f"tesseract OCR failed: {exc}") from exc

    words: list[OcrWord] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        if conf < confidence_floor:
            continue
        words.append(
            OcrWord(
                text=text,
                left=float(data["left"][i]),
                top=float(data["top"][i]),
                width=float(data["width"][i]),
                height=float(data["height"][i]),
                confidence=conf,
                block_num=int(data["block_num"][i]),
                par_num=int(data["par_num"][i]),
                line_num=int(data["line_num"][i]),
                word_num=int(data["word_num"][i]),
            )
        )
    return words


def group_lines(words: list[OcrWord]) -> list[OcrLine]:
    """Group words into lines using Tesseract's own block/par/line grouping."""
    buckets: dict[tuple[int, int, int], list[OcrWord]] = {}
    for w in words:
        key = (w.block_num, w.par_num, w.line_num)
        buckets.setdefault(key, []).append(w)

    lines: list[OcrLine] = []
    for key in sorted(buckets.keys()):
        ws = sorted(buckets[key], key=lambda w: w.word_num)
        lines.append(OcrLine(words=tuple(ws)))
    return lines
