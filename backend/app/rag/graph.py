"""LangGraph workflow for grounded Q&A over the corpus.

Graph shape (Phase 3A — bounded agentic RAG):

    START -> classify_query --(UNANSWERABLE)--> refuse -> END
                  |
                  v (otherwise)
              retrieve -> rerank -> grade_chunks --(sufficient)--> generate_answer -> validate_citations -> claim_grounding -> END
                                        |
                                 (insufficient, loops < MAX_RETRIEVAL_LOOPS)
                                        v
                                   rewrite_query -> retrieve  (loops back into rerank -> grade_chunks)
                                        |
                                 (loops exhausted)
                                        v
                                      refuse -> END

New in Phase 3A versus the Phase 1/2 graph:

- `classify_query` (query understanding, `app/rag/query_classifier.py`) is
  a new entry node. It is a deterministic, rule-based classification by
  default (no added latency) and is the ONE place allowed to short-circuit
  straight to `refuse` before spending a retrieval round trip — but only
  for the narrow `UNANSWERABLE` case (empty input, greeting, small talk),
  never for a genuine question the rules are merely unsure about
  (`AMBIGUOUS` still goes through retrieval; the grader/generator decide
  from there, same as before).
- `retrieve` now calls `retrieval.hybrid_retrieve_single`/`hybrid_retrieve_pooled`
  instead of the dense-only functions — these fuse dense (Pinecone) and
  BM25 (`app/lexical_index.py`) results with Reciprocal Rank Fusion
  (`app/rag/fusion.py`) when `HYBRID_RETRIEVAL_ENABLED=true` (default),
  and degrade to dense-only when it's false. The dense-only functions are
  unchanged and still used internally by the hybrid path.
- `rerank` now prefers a transformer cross-encoder
  (`app/rag/cross_encoder.py`) over the BM25 lexical reranker
  (`app/reranker.py`) when `CROSS_ENCODER_ENABLED=true` (default) and the
  model is actually loadable; otherwise it falls back to BM25
  automatically. Either way this stays a single node with one edge in,
  one edge out — the choice of reranker is an internal implementation
  detail, not a graph-shape change.
- `claim_grounding` is a new terminal node (claim-level grounding,
  `app/rag/grounding.py`) that runs AFTER citation validation, on
  answered (found=True) responses only. It never changes `found`,
  `answer`, or `citations` — it only adds `claims` and `citation_precision`
  metadata to the response for observability/evaluation. A refusal skips
  it entirely (nothing to ground).

Guardrails encoded here (unchanged from Phase 1/2, still hold):
- `grade_chunks` is still the one required branch point that can retry;
  the LLM only supplies a judgment, `_route_after_grade` is a plain Python
  function that makes the actual routing decision. `classify_query`'s
  UNANSWERABLE short-circuit is the only OTHER branch, and it is also
  decided by plain Python from the classifier's output, never by asking
  an LLM "should I refuse?" directly.
- The `loops` counter plus `recursion_limit` passed to `graph.invoke` both
  bound the retry loop, so the graph can never run forever even if the
  counter logic had a bug. Reranking and claim grounding add nodes to the
  overall flow but not new edges back to an earlier node, so neither can
  introduce a new cycle.
- `validate_citations` re-checks every cited chunk_id against the chunks
  actually retrieved for this request, so a fabricated citation can never
  reach the API response. `claim_grounding` runs strictly after this and
  only grades claims against citations that already passed validation —
  it cannot reintroduce an invalid citation.
- The first retrieval attempt is always a single query (cheap); the
  bounded retry is the only place multi-query fan-out (QUERY_FANOUT) is used.
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from app import reranker, retrieval
from app.clients import get_chat, get_embeddings, get_index, get_pinecone
from app.core.config import Settings, get_settings
from app.rag import cross_encoder, llm
from app.rag import figure_reasoning, table_reasoning
from app.rag.grounding import citation_recall, ground_answer
from app.rag.llm import REFUSAL_TEXT
from app.rag.query_classifier import QueryType, classify_query


class GraphState(TypedDict):
    question: str
    query_type: str
    queries: list[str]
    chunks: list[dict]
    grade: dict
    loops: int
    answer: str
    found: bool
    citations: list[str] | list[dict]
    claims: list[dict]
    grounded_claim_rate: float | None
    citation_precision: float | None
    trace: Annotated[list[str], operator.add]
    stage_latencies: Annotated[list[dict], operator.add]
    # Retrieval/reranking observability (Phase 3A hardening — problems #2/#3):
    # the raw per-stage candidate lists and the actual (not just configured)
    # reranker that ran, kept separately from `chunks` (which is overwritten
    # in place as the pipeline narrows it down) so evaluation code can
    # compute Recall@K/MRR/nDCG against every stage, not just the final one,
    # and so an experiment can never mislabel a BM25 fallback as a
    # cross-encoder result.
    dense_candidates: list[dict]
    bm25_candidates: list[dict]
    fused_candidates: list[dict]
    reranked_candidates: list[dict]
    configured_reranker: str
    actual_reranker: str
    fallback_used: bool
    # Phase 5 tenant isolation: the set of owner_id strings ("" = shared
    # public corpus, plus the requesting user's own id when
    # authenticated — see app.security.auth.allowed_owner_ids) this
    # request is permitted to retrieve from. `None` = no restriction
    # (every pre-Phase-5 caller of `ask()`, and every direct test that
    # builds a GraphState by hand) — see `_retrieve` below.
    allowed_owner_ids: frozenset | None
    # Phase 6 (spec item 7, "document-aware" conversation memory):
    # restrict retrieval to this specific set of filenames — the same
    # `source_files` parameter `app.retrieval.hybrid_retrieve_single`/
    # `_pooled` and `app.rag.multi_doc`'s comparison/extraction/
    # summarization/report callers already use. `None` (every
    # pre-Phase-6 caller, including the plain `/ask` endpoint) means
    # "the whole corpus" — completely unchanged default behavior; this
    # is purely additive.
    source_files: frozenset[str] | None


def route_after_grade(grade: dict, loops: int, max_retrieval_loops: int) -> str:
    """Pure branch decision — factored out of the class so it is unit-testable
    without building real Gemini/Groq/Pinecone clients.

    This is the primary retry/refuse branch point: "sufficient" takes the
    good path, otherwise we retry up to `max_retrieval_loops` times before
    refusing.
    """
    if grade.get("sufficient"):
        return "generate_answer"
    if loops < max_retrieval_loops:
        return "rewrite_query"
    return "refuse"


def route_after_classify(query_type: str) -> str:
    """The only other branch point in the graph: an UNANSWERABLE question
    (empty input, greeting, small talk — see app/rag/query_classifier.py)
    skips retrieval entirely and refuses immediately. Every other query
    type, including AMBIGUOUS, still goes through retrieval/grading —
    classification is a routing hint, not a verdict on answerability.
    """
    if query_type == QueryType.UNANSWERABLE.value:
        return "refuse"
    return "retrieve"


class DocumentQAService:
    """Builds the graph once (per process) and answers questions through it."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chat = get_chat(self.settings)
        self.embeddings = get_embeddings(self.settings)
        self.pc = get_pinecone(self.settings)
        self.index = get_index(self.pc, self.settings)
        self.graph = self._build_graph()

    # ---- timing helper --------------------------------------------------

    def _timed_trace(self, stage: str, started: float, message: str) -> dict:
        """Build the (trace line, stage-latency record) pair every node
        returns, so per-stage latency (Phase 3A requirement #40 — measure
        retrieval/reranking/generation/citation-validation separately, not
        just total request latency) is available without instrumenting
        every node by hand at the API layer.
        """
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        return {"stage": stage, "elapsed_ms": elapsed_ms, "message": message}

    # ---- nodes --------------------------------------------------------

    def _classify_query(self, state: GraphState) -> dict:
        started = time.monotonic()
        query_type = classify_query(
            state["question"],
            chat=self.chat if self.settings.query_classifier_llm_enabled else None,
            model=self.settings.answer_model if self.settings.query_classifier_llm_enabled else None,
            use_llm=self.settings.query_classifier_llm_enabled,
        )
        record = self._timed_trace("classify_query", started, f"query_type={query_type.value}")
        return {
            "query_type": query_type.value,
            "trace": [f"classify_query: {query_type.value}"],
            "stage_latencies": [record],
        }

    def _retrieve(self, state: GraphState) -> dict:
        started = time.monotonic()
        allowed_owner_ids = state.get("allowed_owner_ids")
        source_files = state.get("source_files")
        if state.get("loops", 0) == 0:
            breakdown = retrieval.hybrid_retrieve_single(
                self.index,
                self.embeddings,
                self.settings,
                state["question"],
                allowed_owner_ids=allowed_owner_ids,
                source_files=source_files,
            )
            queries = [state["question"]]
        else:
            queries = state["queries"]
            breakdown = retrieval.hybrid_retrieve_pooled(
                self.index,
                self.embeddings,
                self.settings,
                queries,
                allowed_owner_ids=allowed_owner_ids,
                source_files=source_files,
            )
        chunks = breakdown["fused"]
        record = self._timed_trace("retrieve", started, f"{len(chunks)} chunks")
        return {
            "chunks": chunks,
            "dense_candidates": breakdown["dense"],
            "bm25_candidates": breakdown["bm25"],
            "fused_candidates": breakdown["fused"],
            "trace": [
                f"retrieve: queries={queries} -> {len(chunks)} chunks "
                f"(hybrid={self.settings.hybrid_retrieval_enabled}) "
                f"{[c['chunk_id'] for c in chunks]}"
            ],
            "stage_latencies": [record],
        }

    def _rerank(self, state: GraphState) -> dict:
        """Reorder/trim `state['chunks']` before grading.

        Runs on every pass through the retrieve loop. Prefers the
        cross-encoder (`app/rag/cross_encoder.py`) when
        `CROSS_ENCODER_ENABLED=true` and the model actually loads;
        otherwise falls back to the always-available BM25 lexical
        reranker (`app/reranker.py`). Disableable entirely via
        `RERANK_ENABLED=false` — when disabled, or when there's nothing
        to rerank, this is a passthrough.

        Records `configured_reranker` (what the settings asked for),
        `actual_reranker` (what actually ran), and `fallback_used` (Phase
        3A hardening — problem #3: "never label a result Cross Encoder if
        the system actually used the fallback"). Evaluation code
        (`evaluation/run_ablation.py`) reads these instead of trusting
        `CROSS_ENCODER_ENABLED` alone, so a cross-encoder load failure
        during a real experiment run is visible in the report rather than
        silently misattributed.
        """
        started = time.monotonic()
        chunks = state["chunks"]
        configured = "cross_encoder" if self.settings.cross_encoder_enabled else "bm25"
        if not self.settings.rerank_enabled or not chunks:
            record = self._timed_trace("rerank", started, "skipped")
            return {
                "reranked_candidates": chunks,
                "configured_reranker": configured,
                "actual_reranker": "none",
                "fallback_used": False,
                "trace": [
                    f"rerank: skipped (enabled={self.settings.rerank_enabled}, "
                    f"{len(chunks)} candidates)"
                ],
                "stage_latencies": [record],
            }

        reranked = None
        method = "bm25"
        fallback_used = False
        if self.settings.cross_encoder_enabled:
            reranked = cross_encoder.rerank(
                state["question"], chunks, self.settings.rerank_top_k, self.settings.cross_encoder_model
            )
            if reranked is not None:
                method = "cross_encoder"
            else:
                # Cross-encoder was configured but unavailable (model not
                # loadable, predict() failed) — the BM25 path below is a
                # genuine fallback, not the requested reranker.
                fallback_used = True
        if reranked is None:
            reranked = reranker.rerank(state["question"], chunks, self.settings.rerank_top_k)

        record = self._timed_trace("rerank", started, f"method={method}")
        return {
            "chunks": reranked,
            "reranked_candidates": reranked,
            "configured_reranker": configured,
            "actual_reranker": method,
            "fallback_used": fallback_used,
            "trace": [
                f"rerank ({method}{', FALLBACK from cross_encoder' if fallback_used else ''}): "
                f"{len(chunks)} candidates -> top {len(reranked)} "
                f"{[c['chunk_id'] for c in reranked]}"
            ],
            "stage_latencies": [record],
        }

    def _grade_chunks(self, state: GraphState) -> dict:
        started = time.monotonic()
        grade = llm.grade_chunks(self.chat, self.settings.answer_model, state["question"], state["chunks"])
        record = self._timed_trace("grade_chunks", started, f"sufficient={grade['sufficient']}")
        return {
            "grade": grade,
            "trace": [
                f"grade_chunks: sufficient={grade['sufficient']} "
                f"relevant={grade['relevant_chunk_ids']} reason={grade['reason']!r}"
            ],
            "stage_latencies": [record],
        }

    def _rewrite_query(self, state: GraphState) -> dict:
        loops = state.get("loops", 0) + 1
        previous = state["queries"][0] if state.get("queries") else state["question"]
        fanout = self.settings.query_fanout
        query_type = state.get("query_type")
        if query_type in {t.value for t in (
            QueryType.MULTI_HOP, QueryType.COMPARISON, QueryType.CROSS_DOCUMENT, QueryType.SUMMARIZATION,
        )}:
            # Query types that inherently span more than one fact/document
            # benefit from a wider retry fan-out (see query_classifier.py's
            # WIDE_FANOUT_TYPES) — bounded by the settings' own max (5).
            fanout = min(fanout + 1, 5)
        queries = llm.rewrite_query(self.chat, self.settings.answer_model, state["question"], previous, fanout)
        return {
            "queries": queries,
            "loops": loops,
            "trace": [f"rewrite_query (loop {loops}/{self.settings.max_retrieval_loops}, fanout={fanout}): {queries}"],
        }

    def _generate_answer(self, state: GraphState) -> dict:
        started = time.monotonic()
        relevant_ids = set(state["grade"]["relevant_chunk_ids"])
        relevant_chunks = [c for c in state["chunks"] if c["chunk_id"] in relevant_ids]

        # Phase 6, spec item 9/10 ("prefer deterministic Python
        # calculations over LLM arithmetic" for tables; chart value/trend
        # extraction for figures). Tried FIRST, unconditionally — both
        # `try_compute_from_chunks` calls are pure Python (regex + dict
        # lookups), not LLM calls, so trying them costs nothing on the
        # (vast majority of) questions where they correctly find nothing
        # to compute and return `None`. Deliberately NOT gated on
        # `query_type`: no dedicated FIGURE_QUERY/CHART_QUERY classifier
        # type exists (see app/rag/query_classifier.py's comment on
        # AGGREGATION), and gating on evidence content + the question's
        # own wording (each module's `detect_*_operation`) is more robust
        # than gating on upstream classification anyway — this fires
        # exactly when there's both a computable question AND matching
        # table/chart evidence among what grading already deemed
        # relevant, regardless of how `classify_query` labeled the
        # question. `try_compute_from_chunks` never raises — its own
        # module docstring guarantees `None` (not an exception) for
        # "can't compute this deterministically".
        deterministic = table_reasoning.try_compute_from_chunks(relevant_chunks, state["question"])
        if deterministic is None:
            deterministic = figure_reasoning.try_compute_from_chunks(relevant_chunks, state["question"])
        if deterministic is not None:
            record = self._timed_trace("generate_answer", started, "found=True (deterministic)")
            return {
                "found": True,
                "answer": deterministic["explanation"] or deterministic["label"],
                # Already-validated citation dicts (validate_citations ran
                # INSIDE try_compute_from_chunks against the exact chunk(s)
                # the number came from) — see _validate_citations' pass
                # -through check for the "already dicts, not raw IDs" case
                # this relies on (the same convention _claim_grounding
                # already uses one node later).
                "citations": deterministic["citations"],
                "trace": [
                    f"generate_answer: deterministic {deterministic.get('operation')} -> "
                    f"{deterministic['label']!r}"
                ],
                "stage_latencies": [record],
            }

        result = llm.generate_answer(
            self.chat,
            self.settings.answer_model,
            state["question"],
            relevant_chunks,
            temperature=self.settings.answer_temperature,
        )
        record = self._timed_trace("generate_answer", started, f"found={result['found']}")
        return {
            "found": result["found"],
            "answer": result["answer"],
            "citations": result["cited_chunk_ids"],  # raw IDs; validated next
            "trace": [
                f"generate_answer: found={result['found']} cited={result['cited_chunk_ids']}"
            ],
            "stage_latencies": [record],
        }

    def _validate_citations(self, state: GraphState) -> dict:
        started = time.monotonic()
        cited_ids = state["citations"]
        if cited_ids and isinstance(cited_ids[0], dict):
            # Phase 6: _generate_answer's deterministic table/figure path
            # already produced fully-validated citation dicts (via
            # app.rag.llm.validate_citations, called inside
            # try_compute_from_chunks against the exact source chunk) —
            # re-running validate_citations here would break (it expects
            # `list[str]` chunk IDs, not dicts) and would be redundant
            # regardless. Pass through unchanged, same "citations may
            # already be dicts" convention `_claim_grounding` (the next
            # node) already relies on.
            record = self._timed_trace("validate_citations", started, f"{len(cited_ids)} pre-validated (deterministic)")
            return {
                "citations": cited_ids,
                "trace": [f"validate_citations: {len(cited_ids)} pre-validated (deterministic path)"],
                "stage_latencies": [record],
            }

        if not self.settings.citation_validation_enabled:
            # Phase 7 ablation-only bypass (see Settings.citation_validation_enabled's
            # docstring) — NEVER the production default. Still looks each
            # cited_id up against state["chunks"] (so a genuinely correct
            # citation still gets its real source_file/page/score/etc.,
            # keeping downstream claim grounding meaningful), but — unlike
            # the normal path below — never DROPS an id that doesn't
            # resolve to a real retrieved chunk, and never forces a
            # refusal because of the citation check. The point of this
            # branch existing at all is to let evaluation/run_ablation.py's
            # config H measure exactly what the normal path's drop/refuse
            # behavior (below) is protecting against.
            verified = llm.validate_citations(cited_ids, state["chunks"])
            verified_ids = {c["chunk_id"] for c in verified}
            unverified = [
                {
                    "chunk_id": cid,
                    "source_file": "",
                    "section": "",
                    "snippet": "",
                    "score": 0.0,
                    "block_type": "text",
                    "page_start": None,
                    "page_end": None,
                    "source": "native",
                    "confidence": 0.0,
                    "table_rows": None,
                    "object_id": None,
                    "bbox": None,
                    "coordinate_space": None,
                    "visual_type": None,
                    "chart_type": None,
                    "caption": None,
                    "related_text": None,
                    "verified": False,
                }
                for cid in dict.fromkeys(cited_ids)  # de-dupe, preserve order
                if cid not in verified_ids
            ]
            passthrough = verified + unverified
            record = self._timed_trace(
                "validate_citations", started, f"{len(verified)} verified, {len(unverified)} unverified (validation DISABLED)"
            )
            return {
                "citations": passthrough,
                "trace": [
                    f"validate_citations: DISABLED (ablation) — {len(verified)} would-be-valid, "
                    f"{len(unverified)} would-have-been-dropped, none forced to refusal"
                ],
                "stage_latencies": [record],
            }

        valid = llm.validate_citations(cited_ids, state["chunks"])
        dropped = len(set(cited_ids)) - len(valid)
        if not state.get("found") or not valid:
            record = self._timed_trace("validate_citations", started, "refused")
            return {
                "found": False,
                "answer": REFUSAL_TEXT,
                "citations": [],
                "trace": [
                    f"validate_citations: {len(valid)} valid, {dropped} dropped -> refusal "
                    "(no grounded citation survived validation)"
                ],
                "stage_latencies": [record],
            }
        record = self._timed_trace("validate_citations", started, f"{len(valid)} valid")
        return {
            "citations": valid,
            "trace": [f"validate_citations: {len(valid)} valid, {dropped} dropped"],
            "stage_latencies": [record],
        }

    def _claim_grounding(self, state: GraphState) -> dict:
        """Claim-level grounding (Phase 3A) — runs after citation validation,
        only on answered (found=True) responses. Never changes `found`,
        `answer`, or `citations`; only adds `claims`/`grounded_claim_rate`/
        `citation_precision` metadata. See app/rag/grounding.py.
        """
        started = time.monotonic()
        if not self.settings.claim_grounding_enabled or not state.get("found") or not state.get("citations"):
            record = self._timed_trace("claim_grounding", started, "skipped")
            return {"trace": [f"claim_grounding: skipped (found={state.get('found')})"], "stage_latencies": [record]}

        cited_chunks = state["citations"] if isinstance(state["citations"][0], dict) else []
        result = ground_answer(
            self.chat,
            self.settings.answer_model,
            state["answer"],
            cited_chunks,
            max_claims=self.settings.claim_grounding_max_claims,
        )
        record = self._timed_trace(
            "claim_grounding", started, f"grounded_claim_rate={result['grounded_claim_rate']}"
        )
        return {
            "claims": result["claims"],
            "grounded_claim_rate": result["grounded_claim_rate"],
            "citation_precision": result["citation_precision"],
            "trace": [
                f"claim_grounding: {len(result['claims'])} claims, "
                f"grounded_claim_rate={result['grounded_claim_rate']}, "
                f"citation_precision={result['citation_precision']}"
            ],
            "stage_latencies": [record],
        }

    def _refuse(self, state: GraphState) -> dict:
        query_type = state.get("query_type")
        if query_type == QueryType.UNANSWERABLE.value:
            reason = "question classified as not answerable from the corpus (empty/greeting/small talk)"
        else:
            reason = f"retrieval loops exhausted ({state.get('loops', 0)}/{self.settings.max_retrieval_loops})"
        return {
            "found": False,
            "answer": REFUSAL_TEXT,
            "citations": [],
            "trace": [f"refuse: {reason}"],
        }

    # ---- routing --------------------------------------------------------

    def _route_after_grade(self, state: GraphState) -> str:
        return route_after_grade(
            state["grade"], state.get("loops", 0), self.settings.max_retrieval_loops
        )

    def _route_after_classify(self, state: GraphState) -> str:
        return route_after_classify(state["query_type"])

    # ---- graph ------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("classify_query", self._classify_query)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("grade_chunks", self._grade_chunks)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_node("claim_grounding", self._claim_grounding)
        graph.add_node("refuse", self._refuse)

        graph.set_entry_point("classify_query")
        graph.add_conditional_edges(
            "classify_query",
            self._route_after_classify,
            {"retrieve": "retrieve", "refuse": "refuse"},
        )
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "grade_chunks")
        graph.add_conditional_edges(
            "grade_chunks",
            self._route_after_grade,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "refuse": "refuse",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate_answer", "validate_citations")
        graph.add_edge("validate_citations", "claim_grounding")
        graph.add_edge("claim_grounding", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    # ---- public API ---------------------------------------------------

    def ask(
        self,
        question: str,
        allowed_owner_ids: frozenset | None = None,
        source_files: frozenset[str] | None = None,
    ) -> dict:
        initial: GraphState = {
            "question": question,
            "query_type": "",
            "queries": [],
            "chunks": [],
            "grade": {},
            "loops": 0,
            "answer": "",
            "found": False,
            "citations": [],
            "claims": [],
            "grounded_claim_rate": None,
            "citation_precision": None,
            "trace": [],
            "stage_latencies": [],
            "dense_candidates": [],
            "bm25_candidates": [],
            "fused_candidates": [],
            "reranked_candidates": [],
            "configured_reranker": "",
            "actual_reranker": "",
            "fallback_used": False,
            "allowed_owner_ids": allowed_owner_ids,
            "source_files": source_files,
        }
        final = self.graph.invoke(
            initial, config={"recursion_limit": self.settings.recursion_limit}
        )
        required_ids = []  # populated by the evaluation runner, not live traffic
        final_citations = final["citations"] or []
        return {
            "answer": final["answer"],
            "found": final["found"],
            "citations": final_citations,
            "trace": final["trace"],
            "query_type": final.get("query_type"),
            "claims": final.get("claims") or [],
            "grounded_claim_rate": final.get("grounded_claim_rate"),
            "citation_precision": final.get("citation_precision"),
            "citation_recall": citation_recall(
                [c["chunk_id"] for c in final_citations] if final_citations else [], required_ids
            ),
            "stage_latencies": final.get("stage_latencies") or [],
            # Retrieval/reranking observability (Phase 3A hardening —
            # problems #2/#3). Not part of the public HTTP response
            # (app/api/ask.py only reads the fields above); consumed
            # directly by evaluation/run_ablation.py so it can compute
            # real Recall@K/MRR/nDCG per stage and report the reranker
            # that actually ran instead of trusting configuration alone.
            "dense_candidates": final.get("dense_candidates") or [],
            "bm25_candidates": final.get("bm25_candidates") or [],
            "fused_candidates": final.get("fused_candidates") or [],
            "reranked_candidates": final.get("reranked_candidates") or [],
            "final_evidence": final_citations,
            "configured_reranker": final.get("configured_reranker") or "",
            "actual_reranker": final.get("actual_reranker") or "",
            "fallback_used": bool(final.get("fallback_used")),
        }