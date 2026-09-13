"""Offline unit tests for app.rag.grounding — no network call, no API key.

Uses the same fake-Groq-client pattern as tests/test_llm.py.
"""

import json

from app.rag.grounding import (
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    UNSUPPORTED,
    check_claim_support,
    citation_recall,
    extract_claims,
    ground_answer,
)

MODEL = "llama-3.1-8b-instant"


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
    def __init__(self, replies):
        # Accepts a single reply string (used for every call) or a list of
        # replies consumed in order — grounding.ground_answer makes more
        # than one LLM call per invocation (extraction + per-claim checks).
        self._replies = replies if isinstance(replies, list) else None
        self._single = replies if isinstance(replies, str) else None
        self._i = 0

    def create(self, **kwargs):
        if self._single is not None:
            return _FakeCompletion(self._single)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return _FakeCompletion(reply)


class _FakeChat:
    def __init__(self, replies="{}"):
        self.chat = type("_C", (), {})()
        self.chat.completions = _FakeCompletions(replies)


def test_extract_claims_parses_list():
    chat = _FakeChat(json.dumps({"claims": ["The trial lasts 14 days.", "The plan is called Pro."]}))
    claims = extract_claims(chat, MODEL, "The trial lasts 14 days on the Pro plan.")
    assert claims == ["The trial lasts 14 days.", "The plan is called Pro."]


def test_extract_claims_empty_answer_returns_empty_list_without_calling_llm():
    chat = _FakeChat("SHOULD_NOT_BE_PARSED")
    assert extract_claims(chat, MODEL, "") == []
    assert extract_claims(chat, MODEL, "   ") == []


def test_extract_claims_malformed_json_degrades_to_empty_list():
    chat = _FakeChat("not json at all")
    assert extract_claims(chat, MODEL, "Some answer.") == []


def test_check_claim_support_supported():
    # `chunks` is `list[dict]` per check_claim_support's signature — a
    # single evidence chunk is labeled EVIDENCE_1 by `_evidence_context`,
    # and the fake reply must name it in `supporting_evidence` for the
    # SUPPORTED label to resolve (see the fail-closed check at the end of
    # `check_claim_support`: an unexplained label degrades to UNSUPPORTED).
    chat = _FakeChat(
        json.dumps(
            {
                "label": "SUPPORTED",
                "reason": "directly stated",
                "supporting_evidence": ["EVIDENCE_1"],
            }
        )
    )
    chunks = [{"chunk_id": "a::x::0", "text": "Every account gets a 14-day trial."}]
    result = check_claim_support(chat, MODEL, "The trial lasts 14 days.", chunks)
    assert result["label"] == SUPPORTED
    assert result["supporting_chunk_ids"] == ["a::x::0"]


def test_check_claim_support_no_evidence_is_unsupported_without_calling_llm():
    chat = _FakeChat("SHOULD_NOT_BE_PARSED")
    result = check_claim_support(chat, MODEL, "claim", [])
    assert result["label"] == UNSUPPORTED
    assert "no evidence" in result["reason"]


def test_check_claim_support_invalid_label_degrades_to_unsupported():
    chat = _FakeChat(json.dumps({"label": "MAYBE", "reason": "unclear"}))
    chunks = [{"chunk_id": "a::x::0", "text": "some evidence text"}]
    result = check_claim_support(chat, MODEL, "claim", chunks)
    assert result["label"] == UNSUPPORTED


def test_check_claim_support_supported_label_without_resolvable_evidence_fails_closed():
    """A SUPPORTED/PARTIALLY_SUPPORTED label with no `supporting_evidence`
    that resolves to a real chunk is inconsistent model output — treated
    as UNSUPPORTED rather than trusted (see check_claim_support's
    docstring, "fail closed").
    """
    chat = _FakeChat(json.dumps({"label": "SUPPORTED", "reason": "stated"}))  # no supporting_evidence
    chunks = [{"chunk_id": "a::x::0", "text": "Every account gets a 14-day trial."}]
    result = check_claim_support(chat, MODEL, "The trial lasts 14 days.", chunks)
    assert result["label"] == UNSUPPORTED
    assert result["supporting_chunk_ids"] == []


def test_ground_answer_no_claims_is_vacuously_fully_grounded():
    chat = _FakeChat(json.dumps({"claims": []}))
    result = ground_answer(chat, MODEL, "I cannot find the answer.", cited_chunks=[])
    assert result["claims"] == []
    assert result["grounded_claim_rate"] == 1.0
    assert result["citation_precision"] == 1.0


def test_ground_answer_all_claims_supported():
    replies = [
        json.dumps({"claims": ["The trial lasts 14 days."]}),  # extraction
        json.dumps(
            {
                "label": "SUPPORTED",
                "reason": "stated",
                "supporting_evidence": ["EVIDENCE_1"],
            }
        ),  # per-claim check (single combined call — see ground_answer's "Cost" docstring)
    ]
    chat = _FakeChat(replies)
    cited = [{"chunk_id": "a::x::0", "text": "Every account gets a 14-day trial."}]

    result = ground_answer(chat, MODEL, "The trial lasts 14 days.", cited_chunks=cited)

    assert result["grounded_claim_rate"] == 1.0
    assert result["citation_precision"] == 1.0
    assert result["claims"][0]["label"] == SUPPORTED
    assert result["grounding_calls"] == 2  # 1 extraction + 1 per-claim, no separate precision pass


def test_ground_answer_partially_and_unsupported_claims_lower_the_rate():
    replies = [
        json.dumps({"claims": ["Claim A.", "Claim B."]}),  # extraction
        json.dumps(
            {"label": "SUPPORTED", "reason": "x", "supporting_evidence": ["EVIDENCE_1"]}
        ),  # claim A check
        json.dumps({"label": "UNSUPPORTED", "reason": "not stated", "supporting_evidence": []}),  # claim B check
    ]
    chat = _FakeChat(replies)
    cited = [{"chunk_id": "a::x::0", "text": "some evidence"}]

    result = ground_answer(chat, MODEL, "Claim A. Claim B.", cited_chunks=cited, max_claims=8)

    assert result["grounded_claim_rate"] == 0.5
    labels = [c["label"] for c in result["claims"]]
    assert SUPPORTED in labels and UNSUPPORTED in labels
    assert result["grounding_calls"] == 3  # 1 extraction + 1 per claim (2 claims)


def test_ground_answer_respects_max_claims_bound():
    many_claims = [f"Claim {i}." for i in range(20)]
    replies = [json.dumps({"claims": many_claims})] + [
        json.dumps({"label": "SUPPORTED", "reason": "x"})
    ] * 30
    chat = _FakeChat(replies)
    result = ground_answer(chat, MODEL, "many claims", cited_chunks=[{"chunk_id": "a", "text": "x"}], max_claims=3)
    assert len(result["claims"]) == 3


def test_citation_recall_no_ground_truth_returns_none():
    assert citation_recall(["a", "b"], required_chunk_ids=[]) is None


def test_citation_recall_computes_overlap_fraction():
    assert citation_recall(["a", "b"], required_chunk_ids=["a", "b", "c"]) == round(2 / 3, 4)


def test_citation_recall_full_match_is_one():
    assert citation_recall(["a", "b"], required_chunk_ids=["a", "b"]) == 1.0


def test_citation_recall_no_overlap_is_zero():
    assert citation_recall(["x", "y"], required_chunk_ids=["a", "b"]) == 0.0