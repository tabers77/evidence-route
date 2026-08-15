"""Tests for the structured answer contract."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evidence_route.generation.schema import (
    StructuredAnswer,
    extract_json_object,
    parse_structured_answer,
)
from evidence_route.workflows.actions import AbstentionReason


# ---------------------------------------------------------------------------
# The answer / abstention invariant
# ---------------------------------------------------------------------------
def test_plain_answer_is_valid():
    answer = StructuredAnswer(answer="$66,608 million", confidence=0.8)
    assert not answer.abstained
    assert answer.citations == []


def test_answer_with_citations():
    answer = StructuredAnswer(
        answer="$66,608 million",
        citations=[{"chunk_id": "D::c0001", "quoted_text": "Total revenues 66,608"}],
    )
    assert answer.cited_chunk_ids == {"D::c0001"}


def test_abstention_requires_a_reason():
    with pytest.raises(ValidationError, match="requires an abstention_reason"):
        StructuredAnswer(abstained=True)


def test_abstention_cannot_also_answer():
    """Both would let one workflow be scored as answering and as abstaining."""
    with pytest.raises(ValidationError, match="cannot also carry an answer"):
        StructuredAnswer(
            abstained=True,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            answer="but here it is anyway",
        )


def test_reason_without_abstaining_is_rejected():
    with pytest.raises(ValidationError, match="only valid when abstained"):
        StructuredAnswer(answer="something", abstention_reason=AbstentionReason.AMBIGUOUS_QUESTION)


def test_non_abstaining_answer_must_have_text():
    """A workflow that cannot answer should abstain, not return an empty answer."""
    with pytest.raises(ValidationError, match="must contain text"):
        StructuredAnswer()


def test_abstain_helper():
    answer = StructuredAnswer.abstain(AbstentionReason.RETRIEVAL_DISAGREEMENT)
    assert answer.abstained
    assert answer.abstention_reason is AbstentionReason.RETRIEVAL_DISAGREEMENT
    assert answer.answer is None


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", confidence=1.5)
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", confidence=-0.1)


def test_confidence_is_optional():
    assert StructuredAnswer(answer="x").confidence is None


def test_answers_are_frozen():
    answer = StructuredAnswer(answer="x")
    with pytest.raises(ValidationError):
        answer.answer = "y"  # type: ignore[misc]


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", confidance=0.5)  # typo


# ---------------------------------------------------------------------------
# Hallucinated citations
# ---------------------------------------------------------------------------
def test_citation_of_an_unretrieved_chunk_is_detected():
    """A model citing evidence it was never shown is a hallucinated citation."""
    answer = StructuredAnswer(
        answer="x",
        citations=[{"chunk_id": "D::c0001"}, {"chunk_id": "D::c9999"}],
    )
    assert answer.validate_citations({"D::c0001"}) == ["D::c9999"]


def test_valid_citations_report_nothing():
    answer = StructuredAnswer(answer="x", citations=[{"chunk_id": "D::c0001"}])
    assert answer.validate_citations({"D::c0001", "D::c0002"}) == []


def test_uncited_answer_passes_citation_validation():
    assert StructuredAnswer(answer="x").validate_citations(set()) == []


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------
def test_extracts_bare_json():
    assert extract_json_object('{"answer": "x"}') == {"answer": "x"}


def test_extracts_from_a_markdown_fence():
    raw = 'Here you go:\n```json\n{"answer": "x"}\n```\nHope that helps.'
    assert extract_json_object(raw) == {"answer": "x"}


def test_extracts_from_an_unlabelled_fence():
    assert extract_json_object('```\n{"answer": "x"}\n```') == {"answer": "x"}


def test_extracts_json_surrounded_by_prose():
    """A substantively correct answer wrapped in chatter is a correct answer,
    not a parse failure — otherwise the failure rate measures prompt
    formatting rather than the workflow."""
    raw = 'Sure! {"answer": "$66,608 million"} Let me know if you need more.'
    assert extract_json_object(raw)["answer"] == "$66,608 million"


def test_nested_objects_survive_brace_scanning():
    raw = 'text {"answer": "x", "citations": [{"chunk_id": "c1"}]} more text'
    assert extract_json_object(raw)["citations"] == [{"chunk_id": "c1"}]


def test_unparseable_response_raises_with_context():
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json_object("I cannot answer that question.")


def test_json_array_is_not_accepted_as_an_object():
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json_object('["not", "an", "object"]')


# ---------------------------------------------------------------------------
# Parsing model responses
# ---------------------------------------------------------------------------
def test_parses_a_full_response():
    raw = json.dumps(
        {
            "answer": "$66,608 million",
            "citations": [{"chunk_id": "D::c0001", "quoted_text": "Total revenues"}],
            "confidence": 0.9,
        }
    )
    answer = parse_structured_answer(raw)
    assert answer.answer == "$66,608 million"
    assert answer.confidence == 0.9
    assert answer.citations[0].quoted_text == "Total revenues"


def test_bare_string_citations_are_normalized():
    """Models routinely emit chunk ids as plain strings."""
    raw = json.dumps({"answer": "x", "citations": ["D::c0001", "D::c0002"]})
    assert parse_structured_answer(raw).cited_chunk_ids == {"D::c0001", "D::c0002"}


def test_citations_without_a_chunk_id_are_dropped():
    raw = json.dumps({"answer": "x", "citations": [{"quoted_text": "orphan"}]})
    assert parse_structured_answer(raw).citations == []


def test_reason_code_alone_implies_abstention():
    raw = json.dumps({"abstention_reason": "insufficient_evidence"})
    answer = parse_structured_answer(raw)
    assert answer.abstained
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_empty_string_answer_is_treated_as_absent():
    raw = json.dumps({"answer": "", "abstention_reason": "ambiguous_question"})
    assert parse_structured_answer(raw).abstained


def test_unknown_reason_code_is_rejected():
    """An unrecognised code must fail rather than silently becoming a category
    the abstention analysis does not know about."""
    raw = json.dumps({"abstained": True, "abstention_reason": "i_felt_like_it"})
    with pytest.raises(ValidationError):
        parse_structured_answer(raw)


def test_parse_failure_propagates():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_structured_answer("no json here")
