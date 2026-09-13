# OCR

## When OCR runs

- **PNG/JPG**: always — there's no native text layer to try first.
- **PDF**: per page. Each page is parsed natively first
  (`pdfplumber`); if the resulting text is shorter than
  `OCR_MIN_NATIVE_CHARS_PER_PAGE` (default 20 characters) *and* no native
  table was found on that page either, the page is treated as scanned:
  rendered to an image (`pdf2image`/poppler, at `OCR_DPI`, default 200)
  and OCR'd. A PDF can mix native and scanned pages — e.g. a signature
  page scanned into an otherwise-native contract — and each page is
  judged independently; `ExtractedDocument.has_scanned_pages` reflects
  this at the document level, and each individual page's `PageInfo`
  carries its own `is_scanned` + `ocr_confidence`.
- **DOCX/TXT/MD**: never — these formats are always native text.

## What OCR preserves

`app/documents/ocr/engine.py` calls `pytesseract.image_to_data`, not
`image_to_string` — the difference matters. `image_to_data` returns, per
word: text, bounding box (left/top/width/height), a 0–100 confidence
score, and Tesseract's own block/paragraph/line grouping. This is what
lets:

- lines be reconstructed in the right reading order (`group_lines`),
- table regions be detected from horizontal whitespace gaps between word
  groups (`app/documents/tables/reconstruct.py`), and
- every resulting `Block` carry a real confidence score, not a guess.

Plain `image_to_string` would collapse all of this into one string with no
position or confidence information, making structure preservation and
"never claim perfect preservation" (below) both impossible.

## Confidence, not certainty

Every OCR-sourced `Block` carries `source=ExtractionSource.OCR` and a
`confidence` in `[0, 1]` (Tesseract's mean word confidence for that
block/table region, normalized). This confidence is:

- stored on the chunk (`Chunk.confidence`, `Chunk.source`),
- stored on the Pinecone vector's metadata (`confidence`, `source`
  fields — see `app/vectorstore.py`),
- returned on every citation (`Citation.confidence`, `Citation.source` in
  `app/schemas/documents.py`),
- aggregated per document version (`DocumentVersion.mean_ocr_confidence`
  in `app/db/models.py`), visible via `GET /documents`.

Nothing in this pipeline claims OCR output is ground truth. A citation
sourced from a low-confidence OCR region should be presented to an end
user as "extracted via OCR, confidence 62%" — not silently treated the
same as a citation from a native PDF text layer (confidence 100% by
construction, since there's no OCR uncertainty in that path).

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `OCR_LANGUAGE` | `eng` | Tesseract language code |
| `OCR_DPI` | `200` | Render DPI for scanned PDF pages |
| `OCR_MIN_NATIVE_CHARS_PER_PAGE` | `20` | Below this, a PDF page is treated as scanned |
| `OCR_CONFIDENCE_FLOOR` | `0.0` | Words below this Tesseract confidence are dropped entirely |

## System dependencies

OCR requires two system packages beyond the Python requirements —
`pip install` alone is not enough:

```
apt-get install -y tesseract-ocr poppler-utils
```

(`tesseract-ocr` for OCR itself; `poppler-utils` for `pdf2image`'s PDF→
image rendering.) If these are missing, `app/documents/ocr/engine.py` and
`app/documents/ocr/scanned_pdf.py` raise a clear `OcrError`/`RuntimeError`
identifying exactly which system dependency is missing, rather than
failing with an opaque import error deep in the call stack.

## Known limitations

See MIGRATION_PLAN.md — heuristic (not model-based) table reconstruction
on scanned content, and no rotation/deskew correction before OCR (a
significantly rotated scan will produce low-confidence or garbled output;
this is flagged via the confidence score, not silently hidden).
