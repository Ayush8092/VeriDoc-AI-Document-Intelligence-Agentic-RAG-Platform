# Phase 4 evaluation dataset schema

`phase4_v1.jsonl` extends `v1.jsonl`'s row schema (never replaces it —
every original v1 field name and value is still present on every carried
-forward row) with the fields the Phase 4 multimodal pipeline needs
ground truth for. One JSON object per line.

## Fields

| Field | Type | Since | Meaning |
|---|---|---|---|
| `id` | str | v1 | Unique question id. |
| `question` | str | v1 | The literal question text sent to `DocumentQAService.ask()`. |
| `query_type` | str | v1 | One of `app/rag/query_classifier.py`'s `QueryType` values, or the Phase 4.1(7) additions `FIGURE_QUERY`/kept-consistent modality tags below. Informational only — the pipeline classifies independently at runtime; this isn't fed in. |
| `answerable` | bool | v1 | Whether a correct system should produce a grounded answer at all. |
| `required_chunks` | list[str] | v1 | **Legacy name, still read.** Chunk IDs that should appear in the citation set. |
| `notes` | str | v1 | Free-text justification / how the ground truth was verified. |
| `expected_chunks` | list[str] | Phase 4 | Preferred name for `required_chunks` going forward; both are read (see `evaluation/run_phase4_eval.py::_ground_truth_from_row`). |
| `expected_pages` | list[int] | Phase 4 | Page numbers the correct citation(s) should fall on. `[]` when not verified for this row (see `ground_truth_status`). |
| `expected_objects` | list[str] | Phase 4 | Expected `object_id`s (table/figure/chart provenance — see `app/documents/provenance.py`). Populated only once a row has actually been ingested and its real object_id observed — see `ground_truth_status`. |
| `expected_citations` | list[dict] | Phase 4 | `[{"bbox": {...}}, ...]` — bounding-box ground truth for `phase4_metrics.bbox_correctness`. |
| `should_refuse` | bool | Phase 4 | Explicit refusal expectation. Falls back to `not answerable` when absent. |
| `modality` | str | Phase 4 | One of `evaluation/modality.py::MODALITIES`. When absent, `classify_modality()` infers it from `requires_*`/`query_type`/`answerable` — see that module. |
| `requires_table` / `requires_visual` / `requires_chart` / `requires_multi_hop` | bool | Phase 4 | Modality-inference inputs; also independently useful for filtering (`jq 'select(.requires_chart)'`). |
| `question_type` | str | Phase 4 | Lowercase mirror of `query_type`, for readability in reports. |
| `difficulty` | str | Phase 4 | Free-text (`"easy"`/`"medium"`/`"hard"`/`"unrated"`). Not consumed by any metric yet — reserved for future difficulty-stratified reporting. |
| `ground_truth_status` | str | Phase 4 | **Read this before trusting a row's object/bbox fields.** One of: <br>• `verified_v1` — carried forward from v1.jsonl exactly as it was hand-verified there (chunk-id-level only; no page/object/bbox verification was attempted). <br>• `verified_phase4` — chunk_id AND page-level ground truth independently confirmed by re-reading the actual source file in this pass. <br>• `pending_ingestion` — the row references real, freshly-created corpus content (see below) whose text/values were verified by inspection, but whose `object_id`/`bbox` cannot be known until the file is actually run through the real ingestion pipeline (object ids are assigned at ingestion time, not predictable in advance). **Never treat a `pending_ingestion` row's `expected_objects`/`expected_citations` as ground truth — they're intentionally empty.** |
| `expected_answer` | str \| null | Phase 7 | The real, verified answer in prose, derived from the same source content as `notes` — every row's `expected_answer` was written by directly re-reading its cited chunk(s), never guessed from the question alone. `null` for `should_refuse=true` rows (a refusal has no "answer" to check against). |
| `expected_documents` | list[str] | Phase 7 | Filenames (not full `corpus::...` chunk ids) the correct answer should draw from — derived mechanically from `expected_chunks`, so it's always consistent with them by construction. `[]` for refusal/ambiguous rows with no required chunks. |

## Why 161 questions, not 300 yet

The product spec's long-term target is 200-300+ questions. Every row in
this file is grounded in real corpus content that was actually read (not
guessed) as part of authoring it — see `ground_truth_status`. The dataset
has grown across several passes: 47 (original v1) -> 58 -> 94 -> **161**
(this pass, Phase 7's benchmark-expansion step) — see
`evaluation/scripts/expand_phase4_dataset_q059_q094.py` for the 58->94
history, and below for what changed 94->161.

**This pass (Phase 7) added 67 rows** (`q095`-`q161`), in two groups:
- **38 rows of genuinely new facts** (`q095`-`q132`): closes every
  category gap the pre-pass audit found completely empty —
  `AGGREGATION` (0 -> 5), `TIMELINE` (0 -> 5), `CONTRADICTION_DETECTION`
  (0 -> 4), `REPORT_GENERATION` (0 -> 2) — plus new real facts from
  `data/corpus/10_vendor_support_ticket.md` (a real document already in
  the corpus that had ZERO benchmark questions pointing at it despite
  containing a genuine embedded prompt-injection attempt), and a brand
  -new fixture, `data/corpus/11_pricing_correction_memo.md`, added
  specifically because the corpus had no genuine, real, inspectable
  contradiction between two documents to ground `CONTRADICTION_DETECTION`
  questions in — see that file for exactly what it corrects and why.
- **29 paraphrase-robustness rows** (`q133`-`q161`): a legitimate,
  standard benchmark-augmentation technique — same underlying fact, same
  `required_chunks`/`expected_answer`/ground truth as an existing
  already-verified row, different surface phrasing (see each row's
  `notes`, which names its source row). Zero new fabrication risk, since
  nothing about the ground truth is new; this tests phrasing robustness,
  not new knowledge.

**Why this pass stopped at 161, not 300**: reaching 300 fully
hand-verified rows from a 10-document corpus this size would mean either
(a) padding further with paraphrases well past the point of diminishing
value, or (b) adding several more corpus fixtures purely to manufacture
question volume. Per this pass's explicit instruction to prioritize
"161 REAL questions... over 300 questionable ones," this pass stopped
once every category had genuine coverage and the paraphrase technique's
marginal value had clearly dropped, rather than padding to a round
number. See the Phase 7 completion report for this reasoning stated
plainly as a blocker, not hidden in this file alone.

**To grow toward 200-300+ without changing the evaluation architecture:**
1. Add more corpus fixtures (real files, real content) under
   `backend/data/corpus/` — for chart/figure/diagram coverage
   specifically, a small PIL/matplotlib generation script like the one
   used for `08_regional_sales_chart.png`/`09_system_architecture_diagram.png`
   (see git history / MIGRATION_PLAN.md) keeps values inspectable and
   avoids fabricated ground truth.
2. Append new rows to `phase4_v1.jsonl` following this schema — nothing
   else needs to change. `run_phase4_eval.py`, `evaluation/modality.py`,
   and `app/evaluation/phase4_metrics.py` all operate on the schema, not
   on any fixed row count.
3. Once a row's file has been ingested for real, backfill its
   `expected_objects`/`expected_citations`/`ground_truth_status` from
   the actual `Chunk`/`Document` rows produced (e.g. via a one-off query
   against Pinecone metadata filtered by `source_file`) rather than
   guessing — and flip `ground_truth_status` to `verified_phase4`.

## New corpus fixtures (Phase 4.1(7))

- `data/corpus/08_regional_sales_chart.png` — a genuine bar chart
  (drawn with PIL, not fabricated after the fact) with exact known
  values: Region A=42, Region B=67, Region C=29, Region D=51 (all
  $000s), titled "Regional Sales - FY2025 (in $000s)".
- `data/corpus/09_system_architecture_diagram.png` — a genuine
  box-and-arrow diagram: Client -> API Gateway -> Application Server ->
  Database, with a separate Application Server -> Cache branch.

Both exist because the ORIGINAL sample corpus (`data/corpus/README.txt`)
had zero real figures/charts/diagrams — every prior visual-modality
question would have had to reference an object that doesn't exist. These
two files make the figure/chart/visual questions in this dataset
genuinely gradeable once ingested, instead of speculative. Reproducible
via `evaluation/scripts/generate_corpus_fixtures.py`.

## New corpus fixtures (pre_final_phase_4)

- `data/corpus/10_vendor_support_ticket.md` — a genuine markdown
  document with an embedded prompt-injection attempt inside its "Ticket
  Body" section (`"Assistant, ignore all previous instructions..."`),
  surrounded by legitimate, factual support-ticket content. Grounds
  `q090-q094`: two genuine-fact lookups from the same chunk the
  injection lives in (testing that legitimate nearby facts are still
  correctly extracted), one direct injection attempt as the user's own
  query, and two rows testing that the embedded instruction is described
  as content rather than obeyed. `app/security/prompt_injection.py`
  previously had no real ingestable fixture to be exercised against.
