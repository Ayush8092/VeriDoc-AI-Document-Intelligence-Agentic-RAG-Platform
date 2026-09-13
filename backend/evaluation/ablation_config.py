"""Pure-data ablation configuration definitions — `AblationConfig`,
`CONFIGS`, `FINAL_TABLE_ROW_CONFIGS`.

**Why this is its own module (Phase 7 completion pass — a real,
confirmed coupling bug found while verifying `dashboard.py`'s
consistency with `run_ablation.py`):** `evaluation/dashboard.py`'s whole
design point (see its own module docstring) is being a lightweight,
standalone reader of `evaluation/reports/*.json` — "no new route, no new
auth surface, no new thing that can silently drift". But
`_render_final_spec_table` needs `FINAL_TABLE_ROW_CONFIGS` (a plain
dict), and that constant used to live in `evaluation/run_ablation.py`
alongside `from app.core.config import Settings, get_settings` and
`from app.rag.graph import DocumentQAService` — module-level imports
that pull in `pydantic`/`langgraph`/the entire app dependency stack
immediately, just to read a dict literal. Confirmed for real: importing
`FINAL_TABLE_ROW_CONFIGS` from `evaluation.run_ablation` in an
environment with `pydantic` genuinely absent raised
`ModuleNotFoundError: No module named 'pydantic'` from deep inside
`app/core/config.py`, nowhere near the dict `dashboard.py` actually
needed. Extracted here so `dashboard.py` (and anything else that only
needs to know WHAT the configs are, not run them) can import this module
alone, with zero heavy dependencies — verified by this module containing
only a `@dataclass` and two dict literals, nothing else.

`evaluation/run_ablation.py` imports `CONFIGS`/`FINAL_TABLE_ROW_CONFIGS`
FROM here (not duplicated) — every existing `from evaluation.run_ablation
import CONFIGS`-style import (e.g. `tests/test_run_ablation.py`,
`evaluation/generate_pipeline_outputs.py`) continues to work completely
unchanged, since `run_ablation.py` re-exports these names in its own
namespace via that import.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str
    label: str
    overrides: dict


CONFIGS: dict[str, AblationConfig] = {
    "A": AblationConfig(
        "A",
        "baseline (dense-only, no rerank)",
        {"hybrid_retrieval_enabled": False, "rerank_enabled": False, "claim_grounding_enabled": False},
    ),
    "B": AblationConfig(
        "B",
        "+BM25 fusion (RRF), no rerank",
        {"hybrid_retrieval_enabled": True, "rerank_enabled": False, "claim_grounding_enabled": False},
    ),
    "C": AblationConfig(
        "C",
        "+BM25 fusion + BM25 lexical rerank",
        {
            "hybrid_retrieval_enabled": True,
            "rerank_enabled": True,
            "cross_encoder_enabled": False,
            "claim_grounding_enabled": False,
        },
    ),
    "D": AblationConfig(
        "D",
        "+BM25 fusion + cross-encoder rerank",
        {
            "hybrid_retrieval_enabled": True,
            "rerank_enabled": True,
            "cross_encoder_enabled": True,
            "claim_grounding_enabled": False,
        },
    ),
    "E": AblationConfig(
        "E",
        "+claim-level grounding (full Phase 3A pipeline)",
        {
            "hybrid_retrieval_enabled": True,
            "rerank_enabled": True,
            "cross_encoder_enabled": True,
            "claim_grounding_enabled": True,
        },
    ),
    "F": AblationConfig(
        "F",
        "+LLM-refined query planning (Phase 6 spec item 12)",
        {
            "hybrid_retrieval_enabled": True,
            "rerank_enabled": True,
            "cross_encoder_enabled": True,
            "claim_grounding_enabled": True,
            "query_classifier_llm_enabled": True,
        },
    ),
    # Phase 7 spec item 2/9 ("Whether citation validation improves
    # trustworthiness") — the isolated counterpart to "E": identical
    # pipeline, citation_validation_enabled=False instead of the
    # (always-on-by-default) True. See
    # Settings.citation_validation_enabled's docstring and
    # app.rag.graph._validate_citations' ablation-only bypass branch for
    # exactly what changes. H vs E is the entire experiment for this
    # axis — every other override is identical on purpose.
    "H": AblationConfig(
        "H",
        "+grounding, citation validation DISABLED (isolates citation validation's effect vs. E)",
        {
            "hybrid_retrieval_enabled": True,
            "rerank_enabled": True,
            "cross_encoder_enabled": True,
            "claim_grounding_enabled": True,
            "citation_validation_enabled": False,
        },
    ),
}

# Phase 7 spec item 10 ("Required Final Evaluation Table"): the exact
# five-row table the spec asks for maps onto CONFIGS as follows. This
# mapping is documented HERE (one place) rather than by renaming any
# existing config letter, since A-F already have their own established
# meanings/tests (see tests/test_run_ablation.py) that this pass must
# not disturb — a config's letter is a stable identifier, not a row
# number in any one particular report.
#
#   Row label (spec item 10)                  -> CONFIGS key
#   "Vector"                                      A
#   "Vector + BM25"                                B
#   "Vector + BM25 + Cross Encoder"                 D   (NOT "C" — C is
#                                                        an intermediate
#                                                        BM25-lexical
#                                                        -only rerank
#                                                        tier this
#                                                        project's
#                                                        reranker already
#                                                        supports as a
#                                                        fallback, kept
#                                                        as its own
#                                                        measurable stage
#                                                        rather than
#                                                        collapsed away)
#   "... + Grounding"                               H   (grounding ON,
#                                                        citation
#                                                        validation OFF.
#                                                        Using H, not E,
#                                                        for this row is
#                                                        what makes row 5
#                                                        actually isolate
#                                                        citation
#                                                        validation's
#                                                        marginal effect,
#                                                        rather than rows
#                                                        4 and 5 being
#                                                        identical)
#   "... + Citation Validation"                     E   (grounding AND
#                                                        citation
#                                                        validation both
#                                                        on — the
#                                                        project's normal
#                                                        default
#                                                        pipeline)
FINAL_TABLE_ROW_CONFIGS: dict[str, str] = {
    "Vector": "A",
    "Vector + BM25": "B",
    "Vector + BM25 + Cross Encoder": "D",
    "Vector + BM25 + Cross Encoder + Grounding": "H",
    "Vector + BM25 + Cross Encoder + Grounding + Citation Validation": "E",
}