"""Tests for the A0 and A1 workflows.

Uses the scripted provider throughout, so what is checked is the workflow's own
behaviour — prompt construction, accounting, and above all what happens when
things go wrong.
"""

from __future__ import annotations

import json

import pytest

from evidence_route.documents.models import Chunk
from evidence_route.generation.client import (
    GenerationResponse,
    ProviderError,
    ScriptedProvider,
)
from evidence_route.generation.cost import PRICING, TokenPricing, register_pricing
from evidence_route.storage.records import QuestionRecord
from evidence_route.workflows.actions import AbstentionReason, Action
from evidence_route.workflows.base import WorkflowContext
from evidence_route.workflows.bm25_workflow import BM25Workflow
from evidence_route.workflows.direct import DirectAnswerWorkflow


@pytest.fixture(autouse=True)
def _clean_pricing():
    original = dict(PRICING)
    PRICING.clear()
    yield
    PRICING.clear()
    PRICING.update(original)


def _chunk(ordinal: int, text: str, document_id: str = "ACME_2022_10K") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::c{ordinal:04d}",
        document_id=document_id,
        text=text,
        page_numbers=(ordinal + 1,),
        ordinal=ordinal,
        token_estimate=max(1, len(text) // 4),
    )


def _context(**overrides) -> WorkflowContext:
    kwargs = {
        "question": QuestionRecord(
            question_id="fb-1",
            dataset="financebench",
            split="dev",
            question_text="What were total revenues?",
            document_ids=["ACME_2022_10K"],
            reference_answer="66,608",
        ),
        "experiment_id": "exp-1",
        "chunks": [
            _chunk(0, "Total revenues 66,608 for the fiscal year"),
            _chunk(1, "Research and development expense 2,852"),
            _chunk(2, "Unrelated discussion of weather"),
        ],
        "random_seed": 42,
        "code_commit": "abc123",
    }
    kwargs.update(overrides)
    return WorkflowContext(**kwargs)


def _answer_json(**overrides) -> str:
    payload = {
        "answer": "$66,608 million",
        "citations": [{"chunk_id": "ACME_2022_10K::c0000", "quoted_text": "Total revenues 66,608"}],
        "confidence": 0.9,
        "abstained": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# A0 — direct answer
# ---------------------------------------------------------------------------
def test_direct_answers_without_retrieving():
    provider = ScriptedProvider(responses=[_answer_json(citations=[])])
    outcome = DirectAnswerWorkflow(provider).run(_context())

    assert outcome.action_id is Action.DIRECT
    assert outcome.succeeded
    assert outcome.retrieved_items == []
    assert outcome.answer == "$66,608 million"


def test_direct_ignores_available_chunks():
    """A0's evidence list must be empty, not evidence it did not use."""
    provider = ScriptedProvider(responses=[_answer_json(citations=[])])
    outcome = DirectAnswerWorkflow(provider).run(_context())
    assert outcome.retrieved_items == []
    assert "Total revenues" not in provider.requests[0].user


def test_direct_prompt_contains_only_the_question():
    provider = ScriptedProvider(responses=[_answer_json(citations=[])])
    DirectAnswerWorkflow(provider).run(_context())
    assert provider.requests[0].user == "Question: What were total revenues?"
    assert provider.requests[0].prompt_version == "direct_v1"


def test_direct_retrieval_latency_is_zero():
    provider = ScriptedProvider(responses=[_answer_json(citations=[])])
    outcome = DirectAnswerWorkflow(provider).run(_context())
    assert outcome.latency_breakdown is not None
    assert outcome.latency_breakdown.retrieval_ms == pytest.approx(0.0, abs=5.0)


# ---------------------------------------------------------------------------
# A1 — BM25 retrieval plus generation
# ---------------------------------------------------------------------------
def test_bm25_retrieves_and_answers():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.action_id is Action.BM25
    assert outcome.succeeded
    assert outcome.retrieved_items
    assert outcome.retrieved_items[0].chunk_id == "ACME_2022_10K::c0000"


def test_bm25_prompt_carries_evidence_with_citable_ids():
    provider = ScriptedProvider(responses=[_answer_json()])
    BM25Workflow(provider).run(_context())

    user = provider.requests[0].user
    assert "[ACME_2022_10K::c0000]" in user
    assert "Total revenues 66,608" in user
    assert "What were total revenues?" in user
    assert provider.requests[0].prompt_version == "grounded_v1"


def test_bm25_respects_top_k():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider, top_k=1).run(_context())
    assert len(outcome.retrieved_items) == 1


def test_bm25_records_retrieval_latency():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())
    assert outcome.latency_breakdown is not None
    assert outcome.latency_breakdown.retrieval_ms >= 0
    assert outcome.latency_breakdown.total_ms >= outcome.latency_breakdown.retrieval_ms


def test_bm25_with_no_chunks_still_produces_an_outcome():
    """Nothing to search is a legitimate state, and the grounded prompt tells
    the model to abstain rather than invent an answer."""
    provider = ScriptedProvider(
        responses=[json.dumps({"abstention_reason": "insufficient_evidence"})]
    )
    outcome = BM25Workflow(provider).run(_context(chunks=[]))

    assert outcome.succeeded
    assert outcome.retrieved_items == []
    assert outcome.abstained
    assert "no evidence" in provider.requests[0].user.lower()


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------
def test_a_grounded_workflow_can_abstain():
    """Declining when evidence is absent is the behaviour being measured."""
    provider = ScriptedProvider(
        responses=[
            json.dumps(
                {"abstained": True, "abstention_reason": "insufficient_evidence", "confidence": 0.2}
            )
        ]
    )
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.succeeded
    assert outcome.action_id is Action.BM25
    assert outcome.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert outcome.answer is None
    assert outcome.confidence == 0.2


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def test_citations_gain_their_document_id():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.citations[0].chunk_id == "ACME_2022_10K::c0000"
    assert outcome.citations[0].document_id == "ACME_2022_10K"
    assert outcome.citations[0].quoted_text == "Total revenues 66,608"


def test_hallucinated_citations_are_counted_not_stored():
    """A citation naming a chunk the model was never shown is a hallucination.

    It is recorded for the failure taxonomy rather than kept as if it were
    genuine evidence, and rather than raising and losing the whole outcome.
    """
    provider = ScriptedProvider(responses=[_answer_json(citations=[{"chunk_id": "NEVER::c9999"}])])
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.succeeded
    assert outcome.citations == []
    assert outcome.metadata["hallucinated_citations"] == ["NEVER::c9999"]
    assert outcome.metadata["n_hallucinated_citations"] == 1


def test_valid_citations_are_not_flagged():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())
    assert outcome.metadata["n_hallucinated_citations"] == 0


# ---------------------------------------------------------------------------
# Failures become outcomes
# ---------------------------------------------------------------------------
class _FailingProvider:
    name = "failing"

    def complete(self, request):
        raise ProviderError("rate limited")


def test_provider_failure_is_recorded_not_raised():
    """Raising would abandon the run and bias the matrix toward easy questions."""
    outcome = BM25Workflow(_FailingProvider()).run(_context())

    assert not outcome.succeeded
    assert outcome.error_status == "provider_error"
    assert "rate limited" in (outcome.error_detail or "")
    assert outcome.answer is None
    # Retrieval still happened and is still recorded.
    assert outcome.retrieved_items


def test_unparseable_response_is_recorded_with_its_cost():
    """The call was billed even though the output was unusable.

    Treating a parse failure as free would make unreliable formatting look
    cheap and understate the true cost of a workflow.
    """
    register_pricing(
        "m", TokenPricing(input_usd_per_million=1.0, output_usd_per_million=1.0, source="t")
    )

    class Garbage:
        name = "garbage"

        def complete(self, request):
            return GenerationResponse(
                text="I cannot produce JSON.",
                input_tokens=1_000_000,
                output_tokens=0,
                model="m",
                finish_reason="stop",
            )

    outcome = BM25Workflow(Garbage()).run(_context())

    assert not outcome.succeeded
    assert outcome.error_status == "parse_error"
    assert outcome.token_usage.input_tokens == 1_000_000
    assert outcome.estimated_cost_usd == pytest.approx(1.0)


def test_truncation_is_named_in_the_error():
    """A truncated response is a configuration problem, not model refusal."""

    class Truncating:
        name = "truncating"

        def complete(self, request):
            return GenerationResponse(text='{"answer": "cut o', finish_reason="length")

    outcome = BM25Workflow(Truncating()).run(_context())
    assert outcome.error_status == "parse_error"
    assert "truncated" in (outcome.error_detail or "")


def test_failed_outcome_keeps_the_raw_response_for_diagnosis():
    class Garbage:
        name = "garbage"

        def complete(self, request):
            return GenerationResponse(text="not json at all", finish_reason="stop")

    outcome = BM25Workflow(Garbage()).run(_context())
    assert "not json at all" in outcome.metadata["raw_response"]


# ---------------------------------------------------------------------------
# Accounting and provenance
# ---------------------------------------------------------------------------
def test_tokens_and_retries_are_recorded():
    class Retried:
        name = "retried"

        def complete(self, request):
            return GenerationResponse(
                text=_answer_json(),
                input_tokens=100,
                output_tokens=20,
                retry_input_tokens=100,
                retry_output_tokens=5,
                model="m",
            )

    outcome = BM25Workflow(Retried()).run(_context())
    assert outcome.token_usage.total_tokens == 225


def test_unknown_pricing_is_flagged_not_silently_zero():
    """Without the flag, a zero cost is indistinguishable from an unpriced one
    and the budget ceiling stops working."""
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.estimated_cost_usd == 0.0
    assert outcome.metadata["pricing_known"] is False


def test_known_pricing_is_applied():
    register_pricing(
        "scripted",
        TokenPricing(input_usd_per_million=1.0, output_usd_per_million=1.0, source="t"),
    )
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.metadata["pricing_known"] is True
    assert outcome.estimated_cost_usd > 0


def test_outcome_carries_experiment_provenance():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider).run(_context())

    assert outcome.experiment_id == "exp-1"
    assert outcome.question_id == "fb-1"
    assert outcome.random_seed == 42
    assert outcome.code_commit == "abc123"


def test_model_configuration_records_what_produced_the_answer():
    provider = ScriptedProvider(responses=[_answer_json()])
    outcome = BM25Workflow(provider, top_k=7, temperature=0.0).run(_context())

    config = outcome.model_configuration
    assert config["prompt_version"] == "grounded_v1"
    assert config["temperature"] == 0.0
    assert config["top_k"] == 7
    assert config["model"] == "scripted"


def test_seed_is_passed_to_the_provider():
    provider = ScriptedProvider(responses=[_answer_json()])
    BM25Workflow(provider).run(_context())
    assert provider.requests[0].seed == 42


def test_both_actions_report_their_identity():
    provider = ScriptedProvider()
    assert DirectAnswerWorkflow(provider).action is Action.DIRECT
    assert BM25Workflow(provider).action is Action.BM25
    assert BM25Workflow(provider).version == "0.1.0"
