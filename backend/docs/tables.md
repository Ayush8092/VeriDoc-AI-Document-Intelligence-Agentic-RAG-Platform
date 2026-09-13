# Table extraction

## Three extraction paths, one output shape

Regardless of source format, a table always becomes one `Block` with
`block_type=TABLE` and a `TableData` (a grid of cell strings) — see
`app/documents/models.py`. Three different extraction paths feed this:

1. **Native PDF** (`app/documents/parsers/pdf_parser.py`): `pdfplumber`'s
   `page.find_tables()`, which uses the PDF's actual vector line/rect
   objects (and text alignment as a fallback) to find real table
   boundaries. Confidence: 1.0 (exact — no ambiguity, it's not OCR).
2. **Native DOCX** (`app/documents/parsers/docx_parser.py`):
   `python-docx` table objects are real structural elements in the
   `.docx` XML — no detection needed, just extraction. Confidence: 1.0.
3. **Markdown pipe tables** (`app/documents/parsers/text_parser.py`):
   standard `| a | b |` / `|---|---|` GFM syntax, parsed directly.
   Confidence: 1.0.
4. **OCR (scanned PDF pages, PNG/JPG)**
   (`app/documents/tables/reconstruct.py`): no structural table markup
   exists in a rasterized image, so table *shape* itself has to be
   inferred from where the words landed. Confidence: computed, < 1.0 in
   the general case (see below).

## The OCR reconstruction heuristic

1. Group OCR words into lines (`app/documents/ocr/engine.py:group_lines`,
   using Tesseract's own line grouping).
2. Split each line into "cell groups" using a whitespace-gap heuristic: a
   horizontal gap between two words wider than ~1.8× the taller word's
   height is treated as a column boundary rather than a normal space.
3. A run of ≥2 consecutive lines that each split into ≥2 cell groups is a
   candidate table region.
4. Cell-group x-positions across the whole region are clustered into
   column bins (within a 40px tolerance), giving the column count and
   boundaries.
5. Each line's cell groups are assigned to their nearest column bin,
   producing the final grid.
6. Confidence = `0.7 × mean_OCR_word_confidence + 0.3 × row_length_consistency`
   — a table where every row split into exactly the same number of cell
   groups, extracted from high-confidence OCR text, scores close to 1.0;
   a table where row cell-counts varied (a likely sign of misdetected
   columns) or OCR confidence was low scores lower.

This is a real, tested algorithm (`tests/test_table_reconstruction.py`),
not a stub — but it is explicitly a *heuristic*, not a trained
table-structure-recognition model. It works well on clean, left-aligned,
consistently-spaced tabular text (the common case for invoices, logs,
simple forms) and degrades — via a lower confidence score, or by not
firing at all and falling back to plain paragraph blocks — on rotated,
hand-written, merged-cell-heavy, or inconsistently-spaced tables.

## Irregular and merged cells

Native extraction (PDF/DOCX) can represent some merged-cell layouts
correctly, depending on what `pdfplumber`/`python-docx` themselves expose
for the source file. OCR-reconstructed tables do not attempt merge
detection at all — a merged cell in a scanned table will typically appear
as one cell group assigned to a single column, with adjacent columns left
empty on that row, rather than being detected as a intentional span. This
is a known simplification: `docs/tables.md`'s `TableData.n_rows`/`n_cols`
describe the reconstructed grid shape, not a guaranteed-correct
interpretation of the original layout's merges.

## Where table data ends up

- **Chunking** (`app/chunking.py`): a TABLE block is *always* its own
  chunk — never merged into surrounding paragraph text — so a table's
  content is never diluted into (or diluting) unrelated prose in the
  embedding.
- **Vector metadata** (`app/vectorstore.py`): `block_type`, `table_json`
  (the raw grid), `bbox_json`, `page_start`/`page_end`, `source`,
  `confidence` are all stored alongside the vector.
- **Citations** (`app/schemas/documents.py::Citation`): carry
  `table_rows`, `block_type`, `page_start`/`page_end`, `source`,
  `confidence` — enough for a client to render the actual table the
  answer came from, not just a text snippet.
- **Answer generation** (`app/rag/llm.py`): the answer prompt explicitly
  instructs the model to read a table EVIDENCE block by row/column and
  state which row/column a figure came from when it disambiguates the
  answer.
