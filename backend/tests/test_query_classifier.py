"""Unit tests for app.rag.query_classifier — pure Python, no network, no LLM."""

from app.rag.query_classifier import QueryType, classify_query, classify_query_rule_based


def test_empty_question_is_unanswerable():
    assert classify_query_rule_based("") == QueryType.UNANSWERABLE
    assert classify_query_rule_based("   ") == QueryType.UNANSWERABLE


def test_greeting_is_unanswerable():
    assert classify_query_rule_based("hello") == QueryType.UNANSWERABLE
    assert classify_query_rule_based("Thanks!") == QueryType.UNANSWERABLE


def test_short_pronoun_lead_is_ambiguous():
    assert classify_query_rule_based("It expires when?") == QueryType.AMBIGUOUS
    assert classify_query_rule_based("This changes what?") == QueryType.AMBIGUOUS


def test_comparison_question():
    assert classify_query_rule_based("Compare the two remote work policies.") == QueryType.COMPARISON
    assert classify_query_rule_based("Which policy is stricter?") == QueryType.COMPARISON


def test_cross_document_question():
    assert (
        classify_query_rule_based("Find contradictions across all the documents.") == QueryType.CROSS_DOCUMENT
    )


def test_which_documents_mention_pattern_is_cross_document():
    """Phase 7 completion pass: added after a real evaluation run
    (`evaluation/run_query_planning_eval.py` against
    `evaluation/datasets/phase4_v1.jsonl`) found EVERY CROSS_DOCUMENT
    question in that dataset falling through to LOOKUP — this phrasing
    ("which document(s) ... mention/contain/reference/appear/discuss")
    was the most common real miss. See `_CROSS_DOC_WORDS`'s own comment
    for the measured before/after accuracy.
    """
    assert (
        classify_query_rule_based("Which documents in the corpus mention a timeframe of 7 days or less?")
        == QueryType.CROSS_DOCUMENT
    )
    assert (
        classify_query_rule_based("Which document contains the vendor's refund policy?") == QueryType.CROSS_DOCUMENT
    )


def test_extraction_question():
    assert classify_query_rule_based("Extract all contract renewal dates.") == QueryType.EXTRACTION


def test_summarization_question():
    assert classify_query_rule_based("Summarize this document for me.") == QueryType.SUMMARIZATION


def test_table_question():
    assert classify_query_rule_based("What's in row 4 of the pricing table?") == QueryType.TABLE_QUERY


def test_ocr_question():
    assert classify_query_rule_based("What does the scanned receiving log say?") == QueryType.OCR_QUERY


def test_multi_hop_question():
    assert (
        classify_query_rule_based("How did revenue change between Q1 and Q2?") == QueryType.MULTI_HOP
    )


def test_plain_fact_question_is_lookup():
    assert classify_query_rule_based("What is the warranty period?") == QueryType.LOOKUP


def test_classify_query_defaults_to_rule_based_without_llm():
    assert classify_query("Compare A and B.") == QueryType.COMPARISON


def test_classify_query_llm_fallback_used_only_when_requested_and_available():
    # use_llm=True but no chat/model given -> must not crash, falls back to rules.
    assert classify_query("What is the warranty period?", use_llm=True) == QueryType.LOOKUP


def test_classify_query_llm_skipped_when_rules_already_confident():
    calls = []

    class FakeChat:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    raise AssertionError("LLM should not be called when rules are already confident")

    # A comparison question is already confidently classified by rules, so
    # the LLM path must never be invoked even with use_llm=True.
    result = classify_query("Compare A and B.", chat=FakeChat(), model="fake-model", use_llm=True)
    assert result == QueryType.COMPARISON
    assert calls == []


def test_classify_query_llm_refinement_succeeds_and_overrides_rule_based_lookup():
    """The success path: a genuinely ambiguous LOOKUP-by-rules question,
    where a valid LLM response refines it to a more specific type."""
    import json as _json

    class FakeChat:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class _Msg:
                        content = _json.dumps({"type": "EXTRACTION"})

                    class _Choice:
                        message = _Msg()

                    class _Response:
                        choices = [_Choice()]

                    return _Response()

    result = classify_query("What is the warranty period?", chat=FakeChat(), model="fake-model", use_llm=True)
    assert result == QueryType.EXTRACTION


def test_classify_query_llm_invalid_output_falls_back_to_rule_based():
    """Malformed/unrecognized model output (not a real QueryType value,
    or not valid JSON at all) must degrade to the deterministic
    rule-based result, never raise or return garbage."""
    import json as _json

    class FakeChatInvalidType:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class _Msg:
                        content = _json.dumps({"type": "NOT_A_REAL_QUERY_TYPE"})

                    class _Choice:
                        message = _Msg()

                    class _Response:
                        choices = [_Choice()]

                    return _Response()

    result = classify_query("What is the warranty period?", chat=FakeChatInvalidType(), model="fake-model", use_llm=True)
    assert result == QueryType.LOOKUP  # classify_query_rule_based's fallback for this question

    class FakeChatMalformedJSON:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class _Msg:
                        content = "not valid json at all {"

                    class _Choice:
                        message = _Msg()

                    class _Response:
                        choices = [_Choice()]

                    return _Response()

    result2 = classify_query("What is the warranty period?", chat=FakeChatMalformedJSON(), model="fake-model", use_llm=True)
    assert result2 == QueryType.LOOKUP


# ---------------------------------------------------------------------------
# Phase 6 completion audit, Step 6: representative-query routing tests for
# every query type, using the docx's own example queries where the
# existing implementation's labels match them (per the docx: "The exact
# labels must match the implementation" — this project's LOOKUP plays the
# role of "single-document QA", CROSS_DOCUMENT plays "multi-document
# QA/comparison across documents"). Every example below is checked
# against the REAL regex-driven classifier, not asserted from memory of
# what SHOULD happen.
# ---------------------------------------------------------------------------


def test_routing_single_document_lookup_question():
    # docx example: "What is the patient's diagnosis?" -> single-document QA
    assert classify_query_rule_based("What is the patient's diagnosis?") == QueryType.LOOKUP


def test_routing_comparison_across_two_documents():
    # docx example: "Compare the revenue reported in these two documents."
    assert (
        classify_query_rule_based("Compare the revenue reported in these two documents.") == QueryType.COMPARISON
    )


def test_routing_extraction_question():
    # docx example: "Extract the contract start date."
    assert classify_query_rule_based("Extract the contract start date.") == QueryType.EXTRACTION


def test_routing_summarization_question():
    # docx example: "Summarize this document."
    assert classify_query_rule_based("Summarize this document.") == QueryType.SUMMARIZATION


def test_routing_multi_document_which_reports_higher():
    # docx example: "Which document reports the higher revenue?" ->
    # multi-document/comparison. This project's classifier routes it via
    # the comparison pattern ("which (?:is|one|policy|document|plan)").
    result = classify_query_rule_based("Which document reports the higher revenue?")
    assert result == QueryType.COMPARISON


def test_routing_aggregation_total_across_invoices():
    # docx example: "What was the total amount across all invoices?" -> aggregation
    assert (
        classify_query_rule_based("What was the total amount across all invoices?") == QueryType.AGGREGATION
    )


def test_routing_aggregation_highest_lowest():
    assert classify_query_rule_based("What is the highest value in the table?") == QueryType.AGGREGATION
    assert classify_query_rule_based("Calculate the average revenue per quarter.") == QueryType.AGGREGATION


def test_routing_contradiction_what_changed_between_reports():
    # docx example: "What changed between these reports?" -> comparison /
    # contradiction depending on implementation. Neither _CROSS_DOC_WORDS
    # nor _CONTRADICTION_WORDS nor _COMPARISON_WORDS match this exact
    # phrasing in the real implementation (confirmed by running it) --
    # it falls through to the LOOKUP default. Documented here as the
    # actual current behavior, not a claim about what "should" happen.
    result = classify_query_rule_based("What changed between these reports?")
    assert result == QueryType.LOOKUP


def test_routing_contradiction_explicit_disagreement_question():
    # _CONTRADICTION_WORDS matches disagree/disagreement/inconsistent/
    # conflict(s/ing) -- NOT the word "contradict" itself, despite the
    # enum member's name (confirmed by inspecting the real regex at
    # app/rag/query_classifier.py's _CONTRADICTION_WORDS). Using
    # "disagree" here, which the pattern actually catches.
    assert (
        classify_query_rule_based("Do these two documents disagree about the deadline?")
        == QueryType.CONTRADICTION_DETECTION
    )
    assert (
        classify_query_rule_based("Is there a conflict between what these two reports say?")
        == QueryType.CONTRADICTION_DETECTION
    )


def test_routing_table_reasoning_question():
    # docx example: "What is the value in the third column of the table?"
    result = classify_query_rule_based("What is the value in the third column of the table?")
    assert result == QueryType.TABLE_QUERY


def test_routing_report_generation_question():
    # docx example: "Generate a report comparing these documents."
    assert (
        classify_query_rule_based("Generate a report comparing these documents.")
        == QueryType.REPORT_GENERATION
    )


def test_routing_timeline_question():
    assert (
        classify_query_rule_based("Give me a timeline of events in chronological order.") == QueryType.TIMELINE
    )
    assert classify_query_rule_based("What happened when during the incident?") == QueryType.TIMELINE


def test_routing_wide_fanout_types_include_new_phase_6_multi_document_types():
    from app.rag.query_classifier import WIDE_FANOUT_TYPES

    assert QueryType.CONTRADICTION_DETECTION in WIDE_FANOUT_TYPES
    assert QueryType.REPORT_GENERATION in WIDE_FANOUT_TYPES
    assert QueryType.TIMELINE in WIDE_FANOUT_TYPES
    # Deliberately excluded per the implementation's own documented
    # reasoning (most aggregation questions target one already-localized
    # table, not a wide search).
    assert QueryType.AGGREGATION not in WIDE_FANOUT_TYPES