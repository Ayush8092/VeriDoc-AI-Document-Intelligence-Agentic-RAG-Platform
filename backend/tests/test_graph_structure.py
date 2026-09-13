"""Offline unit tests for the graph's nodes/routing — no network, no LLM.

Constructs `DocumentQAService` via `object.__new__` to skip `__init__`
(which would otherwise build real Gemini/Groq/Pinecone clients) — only
`self.settings` is set, which is all the node methods under test need
(any node that would need `self.chat`/`self.embeddings`/`self.index` is
replaced with a fake in these tests).
"""

from app.core.config import Settings
from app.rag.graph import DocumentQAService, route_after_classify, route_after_grade
from app.rag.query_classifier import QueryType
from app.rag import cross_encoder

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def create(self, **kwargs):
        # {"claims": []} makes app.rag.grounding.ground_answer take its
        # documented vacuous-grounding short-circuit (an answer with no
        # extracted claims is "vacuously fully grounded" — see that
        # function's docstring) in exactly ONE call, rather than this
        # fixture needing to also fake the per-claim
        # check_claim_support() follow-up call shape.
        import json

        return _FakeCompletion(json.dumps({"claims": []}))


class _FakeChatClient:
    """Matches the Groq client shape `self.chat` is (`client.chat.completions
    .create(...)`, per `app.rag.llm`) — same fake-client pattern
    `tests/test_grounding.py::_FakeChat` already uses, duplicated here
    (rather than imported across test files) to keep this file's own
    stated convention ("no network, no LLM" + "any node that would need
    self.chat ... is replaced with a fake") self-contained. Used ONLY by
    `test_deterministic_table_path_reachable_through_full_graph_invocation`,
    where `_claim_grounding` runs as a REAL node (that test's whole
    point) and therefore genuinely calls `self.chat` even though the
    ANSWER itself came from the deterministic table path, not an LLM.
    """

    def __init__(self):
        self.chat = type("_C", (), {"completions": _FakeCompletions()})()


def _make_service(**settings_overrides) -> DocumentQAService:
    service = object.__new__(DocumentQAService)
    service.settings = Settings(_env_file=None, **settings_overrides)
    return service


def _chunk(chunk_id, text, score=0.5):
    return {
        "chunk_id": chunk_id,
        "source_file": chunk_id.split("::")[0] + ".md",
        "section": "x",
        "text": text,
        "score": score,
    }


def test_route_after_grade_sufficient_goes_to_generate_answer():
    assert route_after_grade({"sufficient": True}, loops=0, max_retrieval_loops=2) == "generate_answer"


def test_route_after_grade_insufficient_retries_while_loops_remain():
    assert route_after_grade({"sufficient": False}, loops=0, max_retrieval_loops=2) == "rewrite_query"
    assert route_after_grade({"sufficient": False}, loops=1, max_retrieval_loops=2) == "rewrite_query"


def test_route_after_grade_refuses_once_loops_exhausted():
    assert route_after_grade({"sufficient": False}, loops=2, max_retrieval_loops=2) == "refuse"


def test_route_after_classify_unanswerable_goes_straight_to_refuse():
    assert route_after_classify(QueryType.UNANSWERABLE.value) == "refuse"


def test_route_after_classify_every_other_type_goes_to_retrieve():
    for query_type in QueryType:
        if query_type is QueryType.UNANSWERABLE:
            continue
        assert route_after_classify(query_type.value) == "retrieve"


def test_classify_query_node_sets_query_type_and_trace():
    service = _make_service(QUERY_CLASSIFIER_LLM_ENABLED=False)
    result = service._classify_query({"question": "Compare the two policies."})
    assert result["query_type"] == QueryType.COMPARISON.value
    assert any("classify_query" in t for t in result["trace"])
    assert result["stage_latencies"][0]["stage"] == "classify_query"


def test_classify_query_node_flags_empty_question_unanswerable():
    service = _make_service(QUERY_CLASSIFIER_LLM_ENABLED=False)
    result = service._classify_query({"question": "   "})
    assert result["query_type"] == QueryType.UNANSWERABLE.value


def test_rerank_node_reorders_chunks_by_lexical_relevance():
    # Cross-encoder disabled explicitly so this exercises the deterministic
    # BM25 fallback path without needing sentence-transformers installed.
    service = _make_service(RERANK_ENABLED=True, RERANK_TOP_K=2, CROSS_ENCODER_ENABLED=False)
    state = {
        "question": "quarterly revenue growth Asia Pacific",
        "chunks": [
            _chunk("b::x::0", "Unrelated content about something else entirely.", score=0.6),
            _chunk("a::x::0", "Asia Pacific posted quarterly revenue growth of 22%.", score=0.5),
        ],
    }

    result = service._rerank(state)

    assert result["chunks"][0]["chunk_id"] == "a::x::0"
    assert len(result["chunks"]) == 2
    assert any("rerank" in t for t in result["trace"])
    # BM25 was explicitly configured (not a fallback) here.
    assert result["configured_reranker"] == "bm25"
    assert result["actual_reranker"] == "bm25"
    assert result["fallback_used"] is False


def test_rerank_node_falls_back_to_bm25_when_cross_encoder_unavailable(monkeypatch):
    """Cross-encoder is configured but unavailable -> BM25 fallback."""

    monkeypatch.setattr(
        cross_encoder,
        "rerank",
        lambda question, chunks, top_k, model: None,
    )

    service = _make_service(
        RERANK_ENABLED=True,
        RERANK_TOP_K=2,
        CROSS_ENCODER_ENABLED=True,
    )

    state = {
        "question": "notice period",
        "chunks": [
            _chunk("a::x::0", "notice period is 60 days"),
            _chunk("b::x::0", "completely unrelated rent and deposit terms"),
        ],
    }

    result = service._rerank(state)

    assert result["chunks"][0]["chunk_id"] == "a::x::0"
    assert any("rerank (bm25" in t for t in result["trace"])
    assert result["configured_reranker"] == "cross_encoder"
    assert result["actual_reranker"] == "bm25"
    assert result["fallback_used"] is True

def test_rerank_node_respects_configured_top_k():
    service = _make_service(RERANK_ENABLED=True, RERANK_TOP_K=1, CROSS_ENCODER_ENABLED=False)
    state = {
        "question": "notice period",
        "chunks": [
            _chunk("a::x::0", "notice period is 60 days"),
            _chunk("b::x::0", "notice period clause details here too"),
            _chunk("c::x::0", "completely unrelated rent and deposit terms"),
        ],
    }
    result = service._rerank(state)
    assert len(result["chunks"]) == 1


def test_rerank_node_disabled_is_a_passthrough():
    service = _make_service(RERANK_ENABLED=False)
    original_chunks = [_chunk("a::x::0", "content")]
    state = {"question": "q", "chunks": original_chunks}

    result = service._rerank(state)

    assert "chunks" not in result
    assert any("skipped" in t.lower() for t in result["trace"])


def test_claim_grounding_node_skipped_when_disabled():
    service = _make_service(CLAIM_GROUNDING_ENABLED=False)
    state = {"found": True, "citations": [{"chunk_id": "a::x::0", "text": "content"}], "answer": "the answer"}
    result = service._claim_grounding(state)
    assert "claims" not in result
    assert any("skipped" in t.lower() for t in result["trace"])


def test_claim_grounding_node_skipped_on_refusal():
    service = _make_service(CLAIM_GROUNDING_ENABLED=True)
    state = {"found": False, "citations": [], "answer": "I cannot find the answer to this question in the provided documents."}
    result = service._claim_grounding(state)
    assert "claims" not in result
    assert any("skipped" in t.lower() for t in result["trace"])


def test_graph_execution_visits_every_node_in_order():
    service = _make_service(RERANK_ENABLED=True, RERANK_TOP_K=5, CROSS_ENCODER_ENABLED=False, CLAIM_GROUNDING_ENABLED=True)
    visited: list[str] = []

    def fake_classify(state):
        visited.append("classify_query")
        return {"query_type": QueryType.LOOKUP.value, "trace": ["classify_query: LOOKUP"], "stage_latencies": []}

    def fake_retrieve(state):
        visited.append("retrieve")
        return {"chunks": [_chunk("a::x::0", "some content")], "trace": ["retrieve: 1 chunk"], "stage_latencies": []}

    def fake_grade(state):
        visited.append("grade_chunks")
        return {
            "grade": {"sufficient": True, "relevant_chunk_ids": ["a::x::0"], "reason": "ok"},
            "trace": ["grade_chunks: sufficient"],
            "stage_latencies": [],
        }

    def fake_generate(state):
        visited.append("generate_answer")
        return {
            "found": True,
            "answer": "the answer",
            "citations": ["a::x::0"],
            "trace": ["generate_answer: found=True"],
            "stage_latencies": [],
        }

    def fake_validate(state):
        visited.append("validate_citations")
        return {
            "citations": [{"chunk_id": "a::x::0", "source_file": "a.md", "section": "x"}],
            "trace": ["validate_citations: 1 valid, 0 dropped"],
            "stage_latencies": [],
        }

    def fake_claim_grounding(state):
        visited.append("claim_grounding")
        return {
            "claims": [],
            "grounded_claim_rate": 1.0,
            "citation_precision": 1.0,
            "trace": ["claim_grounding: 0 claims"],
            "stage_latencies": [],
        }

    service._classify_query = fake_classify
    service._retrieve = fake_retrieve
    service._grade_chunks = fake_grade
    service._generate_answer = fake_generate
    service._validate_citations = fake_validate
    service._claim_grounding = fake_claim_grounding

    real_rerank = service._rerank

    def tracking_rerank(state):
        visited.append("rerank")
        return real_rerank(state)

    service._rerank = tracking_rerank

    graph = service._build_graph()
    graph.invoke(
        {
            "question": "q",
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
        },
        config={"recursion_limit": 20},
    )

    assert visited == [
        "classify_query",
        "retrieve",
        "rerank",
        "grade_chunks",
        "generate_answer",
        "validate_citations",
        "claim_grounding",
    ]


def test_graph_execution_unanswerable_query_skips_retrieval_entirely():
    service = _make_service()
    visited: list[str] = []

    def tracking_retrieve(state):
        visited.append("retrieve")  # should never run
        return {"chunks": [], "trace": [], "stage_latencies": []}

    service._retrieve = tracking_retrieve

    graph = service._build_graph()
    final = graph.invoke(
        {
            "question": "",
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
        },
        config={"recursion_limit": 20},
    )

    assert visited == []
    assert final["found"] is False
    assert any("refuse" in t for t in final["trace"])


# =============================================================================
# Table-reasoning wiring (Phase 6 completion audit, item 2): query ->
# classifier/planner -> retrieval -> table reasoning -> evidence ->
# generation -> citation validation -> claim grounding.
#
# The graph structure itself (route_after_grade/route_after_classify,
# tested above, plus the fixed node-edge wiring in _build_graph) already
# guarantees every query type reaches _generate_answer via the SAME
# single path — there is no separate/bypassed route a table-reasoning
# question could take instead. What these tests add: proof that
# _generate_answer's deterministic table_reasoning branch (a) actually
# fires for a real table chunk + a real aggregation question, (b)
# produces citation dicts that flow correctly through
# _validate_citations' pass-through branch, and (c) never triggers an
# LLM call to do so — verified by NOT setting service.chat, so any
# accidental fall-through to the LLM path would raise AttributeError
# and fail the test loudly rather than silently passing.
# =============================================================================


def _table_chunk(chunk_id: str, table_rows: list[list[str]], source_file: str = "revenue.pdf") -> dict:
    return {
        "chunk_id": chunk_id,
        "block_type": "table",
        "table_rows": table_rows,
        "source_file": source_file,
        "section": "Revenue by Region",
        "text": "\n".join(" | ".join(row) for row in table_rows),
        "score": 0.9,
        "page_start": 1,
        "page_end": 1,
        "source": "native",
        "confidence": 1.0,
    }


_REVENUE_TABLE_ROWS = [
    ["Region", "Q1 Revenue", "Q4 Revenue"],
    ["North America", "1200000", "1400000"],
    ["Europe", "800000", "900000"],
]


def test_generate_answer_node_takes_deterministic_table_path_without_any_llm_call():
    """service.chat is deliberately left unset (None from object.__new__) —
    if _generate_answer fell through to the LLM path, `llm.generate_answer`
    would immediately raise trying to use it, failing this test loudly.
    """
    service = _make_service()
    state = {
        "question": "Which region had the highest Q4 revenue?",
        "chunks": [_table_chunk("rev::table::0", _REVENUE_TABLE_ROWS)],
        "grade": {"sufficient": True, "relevant_chunk_ids": ["rev::table::0"], "reason": "table present"},
    }

    result = service._generate_answer(state)

    assert result["found"] is True
    assert "1400000" in result["answer"] or "1,400,000" in result["answer"] or "North America" in result["answer"]
    assert result["citations"] and isinstance(result["citations"][0], dict)
    assert result["citations"][0]["chunk_id"] == "rev::table::0"
    assert "deterministic" in result["trace"][0]


def test_generate_answer_node_falls_through_to_llm_path_for_non_computable_question():
    """A question the table-reasoning/figure-reasoning modules can't
    compute (no aggregation keyword) must fall through to the normal LLM
    path — proven here by asserting the LLM path IS reached (raises,
    since service.chat is unset) rather than silently short-circuiting.
    """
    service = _make_service()
    state = {
        "question": "What color is the sky?",
        "chunks": [_table_chunk("rev::table::0", _REVENUE_TABLE_ROWS)],
        "grade": {"sufficient": True, "relevant_chunk_ids": ["rev::table::0"], "reason": "table present"},
    }

    import pytest

    with pytest.raises(AttributeError):
        service._generate_answer(state)


def test_deterministic_table_citations_pass_through_validate_citations_unchanged():
    """_validate_citations must detect the already-dict citations from
    the deterministic path and skip re-running llm.validate_citations
    against them (which expects list[str], not list[dict], and would
    silently produce zero valid citations if it tried)."""
    service = _make_service()
    pre_validated = [{"chunk_id": "rev::table::0", "source_file": "revenue.pdf", "section": "Revenue by Region"}]
    state = {"citations": pre_validated, "chunks": [], "found": True}

    result = service._validate_citations(state)

    assert result["citations"] == pre_validated
    assert "pre-validated" in result["trace"][0]


def test_deterministic_table_path_reachable_through_full_graph_invocation():
    """End-to-end: classify_query -> retrieve -> rerank -> grade_chunks ->
    generate_answer (deterministic table path) -> validate_citations ->
    claim_grounding -> END, via a real graph.invoke() (not individual
    node calls) — the actual reachability guarantee item 2 asks for.
    """
    service = _make_service(RERANK_ENABLED=False, CLAIM_GROUNDING_ENABLED=True)
    # _claim_grounding runs for real in this test (see docstring below) —
    # unlike every other test in this file, it genuinely needs
    # `self.chat` even though the ANSWER comes from the deterministic
    # table path, not an LLM. See _FakeChatClient's docstring.
    service.chat = _FakeChatClient()

    def fake_classify(state):
        return {"query_type": QueryType.TABLE_QUERY.value, "trace": ["classify_query: TABLE_QUERY"], "stage_latencies": []}

    def fake_retrieve(state):
        return {
            "chunks": [_table_chunk("rev::table::0", _REVENUE_TABLE_ROWS)],
            "trace": ["retrieve: 1 chunk"],
            "stage_latencies": [],
        }

    def fake_grade(state):
        return {
            "grade": {"sufficient": True, "relevant_chunk_ids": ["rev::table::0"], "reason": "table present"},
            "trace": ["grade_chunks: sufficient"],
            "stage_latencies": [],
        }

    service._classify_query = fake_classify
    service._retrieve = fake_retrieve
    service._grade_chunks = fake_grade
    # _generate_answer, _validate_citations, _claim_grounding are the
    # REAL methods here — this is the point of the test.

    graph = service._build_graph()
    final = graph.invoke(
        {
            "question": "Which region had the highest Q4 revenue?",
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
        },
        config={"recursion_limit": 20},
    )

    assert final["found"] is True
    assert final["citations"] and final["citations"][0]["chunk_id"] == "rev::table::0"
    assert any("deterministic" in t for t in final["trace"])
    # claim_grounding must not have crashed or reset found/citations —
    # per its own docstring, it only adds claims/grounded_claim_rate/
    # citation_precision metadata.
    assert "claims" in final


# =============================================================================
# Citation-validation bypass (Phase 7 spec item 2/9, ablation config H
# vs E — app.core.config.Settings.citation_validation_enabled).
#
# The Settings-field-level check that H's override actually flips the
# right field already exists (tests/test_run_ablation.py::
# test_ablation_config_overrides_actually_apply, generic over every
# config including H). What's missing, and what these tests add: proof
# that _validate_citations ITSELF genuinely behaves differently when the
# flag is off — not just that the flag exists and toggles.
# =============================================================================


def test_validate_citations_normal_path_drops_and_refuses_on_bad_citation():
    """The baseline (citation_validation_enabled=True, the default)
    behavior H is being compared against: a cited_id that doesn't match
    any retrieved chunk is dropped, and — since nothing survives — the
    whole answer is forced to refusal."""
    service = _make_service(CITATION_VALIDATION_ENABLED=True)
    state = {
        "citations": ["fabricated::chunk::id"],
        "chunks": [_chunk("real::chunk::0", "actual content")],
        "found": True,
    }

    result = service._validate_citations(state)

    assert result["found"] is False
    assert result["citations"] == []
    assert "refus" in result["trace"][0].lower()


def test_validate_citations_bypass_path_keeps_fabricated_citation_marked_unverified():
    """With citation_validation_enabled=False, the SAME fabricated
    cited_id is NOT dropped and does NOT force a refusal — but it must
    be marked verified=False with empty/zero fields, never given fake
    -plausible real-looking data (the bypass exists to MEASURE the
    danger of skipping validation, not to paper over it)."""
    service = _make_service(CITATION_VALIDATION_ENABLED=False)
    state = {
        "citations": ["fabricated::chunk::id"],
        "chunks": [_chunk("real::chunk::0", "actual content")],
        "found": True,
    }

    result = service._validate_citations(state)

    assert result["citations"] != []  # NOT dropped, unlike the normal path
    assert len(result["citations"]) == 1
    fabricated = result["citations"][0]
    assert fabricated["chunk_id"] == "fabricated::chunk::id"
    assert fabricated["verified"] is False
    assert fabricated["source_file"] == ""  # no fake-plausible data filled in
    assert "DISABLED" in result["trace"][0]
    # And found is NOT forced to False either — this is the crux of the
    # ablation: config H's whole point is to show what a (worse, less
    # trustworthy) answer looks like without the safety gate.
    assert "found" not in result  # state["found"] (True) passes through unmodified


def test_validate_citations_bypass_path_still_correctly_resolves_genuine_citations():
    """The bypass doesn't degrade GOOD citations — a chunk_id that DOES
    match a real retrieved chunk still gets its real source_file/score/
    etc, same as the normal path would have given it."""
    service = _make_service(CITATION_VALIDATION_ENABLED=False)
    state = {
        "citations": ["real::chunk::0"],
        "chunks": [_chunk("real::chunk::0", "actual content")],
        "found": True,
    }

    result = service._validate_citations(state)

    assert len(result["citations"]) == 1
    real = result["citations"][0]
    assert real["chunk_id"] == "real::chunk::0"
    assert real.get("verified", True) is not False  # not marked unverified — it's genuine
    assert real["source_file"]  # real metadata, not blanked out


def test_config_e_vs_h_produce_genuinely_different_outcomes_for_the_same_fabricated_citation():
    """Direct A/B comparison proving E and H are not just differently
    -labeled but behaviorally distinct on the exact same input — the
    "does the configuration actually change pipeline behavior" check
    the Phase 7 spec explicitly asks every config to satisfy."""
    from evaluation.run_ablation import CONFIGS

    base = Settings(_env_file=None)
    e_settings = base.model_copy(update=CONFIGS["E"].overrides)
    h_settings = base.model_copy(update=CONFIGS["H"].overrides)
    assert e_settings.citation_validation_enabled is True
    assert h_settings.citation_validation_enabled is False

    state = {
        "citations": ["fabricated::chunk::id"],
        "chunks": [_chunk("real::chunk::0", "actual content")],
        "found": True,
    }

    service_e = _make_service(CITATION_VALIDATION_ENABLED=e_settings.citation_validation_enabled)
    service_h = _make_service(CITATION_VALIDATION_ENABLED=h_settings.citation_validation_enabled)

    result_e = service_e._validate_citations(state)
    result_h = service_h._validate_citations(state)

    assert result_e["found"] is False  # E: refused
    assert "found" not in result_h  # H: not refused (state's found=True passes through)
    assert result_e["citations"] == []
    assert len(result_h["citations"]) == 1