"""Tests for the experiment data model.

These schemas are the contract every later stage depends on, so the constraints
that protect the experiment's validity are tested explicitly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidence_route.storage.records import (
    BanditLogRecord,
    EvaluationRecord,
    LatencyBreakdown,
    QuestionRecord,
    RetrievedItem,
    TokenUsage,
    WorkflowOutcomeRecord,
)
from evidence_route.workflows.actions import AbstentionReason, Action


# ---------------------------------------------------------------------------
# QuestionRecord
# ---------------------------------------------------------------------------
def test_question_record_minimal():
    q = QuestionRecord(
        question_id="fb-0001",
        dataset="financebench",
        split="dev",
        question_text="What was 3M's FY2018 capital expenditure?",
    )
    assert q.document_ids == []
    assert q.reference_evidence == []


def test_question_record_rejects_unknown_split():
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="fb-0001",
            dataset="financebench",
            split="train",  # type: ignore[arg-type]
            question_text="...",
        )


def test_records_reject_unknown_fields():
    """extra='forbid' catches typos that would otherwise silently drop data."""
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="fb-0001",
            dataset="financebench",
            split="dev",
            question_text="...",
            questoin_type="extraction",  # typo
        )


# ---------------------------------------------------------------------------
# WorkflowOutcomeRecord
# ---------------------------------------------------------------------------
def _outcome(**overrides) -> WorkflowOutcomeRecord:
    kwargs = {
        "experiment_id": "exp-1",
        "question_id": "fb-0001",
        "action_id": Action.BM25,
        "answer": "USD 1,577 million",
    }
    kwargs.update(overrides)
    return WorkflowOutcomeRecord(**kwargs)


def test_outcome_defaults_to_success():
    outcome = _outcome()
    assert outcome.succeeded
    assert outcome.estimated_cost_usd == 0.0
    assert outcome.token_usage.total_tokens == 0


def test_abstain_requires_a_reason_code():
    """An abstention without a reason code is not auditable."""
    with pytest.raises(ValidationError, match="requires an abstention_reason"):
        _outcome(action_id=Action.ABSTAIN, answer=None)


def test_abstain_with_reason_is_valid():
    outcome = _outcome(
        action_id=Action.ABSTAIN,
        answer=None,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    assert outcome.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_reason_code_rejected_on_answering_action():
    with pytest.raises(ValidationError, match="only valid for ABSTAIN"):
        _outcome(abstention_reason=AbstentionReason.AMBIGUOUS_QUESTION)


def test_failed_outcome_is_retained_but_not_successful():
    """Errors are recorded, not dropped.

    Silently discarding failures would bias the outcome matrix toward actions
    that happen to fail loudly on hard questions.
    """
    outcome = _outcome(error_status="rate_limited", answer=None)
    assert not outcome.succeeded
    assert outcome.error_status == "rate_limited"


def test_token_usage_counts_retries():
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=200,
        retry_input_tokens=1000,
        retry_output_tokens=50,
    )
    # Retries cost real money; the total must reflect what was actually spent.
    assert usage.total_tokens == 2250


def test_negative_cost_is_rejected():
    with pytest.raises(ValidationError):
        _outcome(estimated_cost_usd=-1.0)


def test_latency_breakdown_includes_feature_extraction():
    """Feature-extraction cost is attributed to the router, so it has its own field."""
    latency = LatencyBreakdown(
        total_ms=2400.0, retrieval_ms=180.0, generation_ms=2100.0, feature_extraction_ms=120.0
    )
    assert latency.feature_extraction_ms == 120.0


def test_retrieved_item_rank_is_one_based():
    with pytest.raises(ValidationError):
        RetrievedItem(chunk_id="c1", document_id="d1", rank=0)


def test_outcome_records_are_frozen():
    outcome = _outcome()
    with pytest.raises(ValidationError):
        outcome.answer = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvaluationRecord
# ---------------------------------------------------------------------------
def test_normalized_score_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        EvaluationRecord(
            experiment_id="exp-1",
            question_id="fb-0001",
            action_id=Action.BM25,
            scorer_name="RetrievalRecallScorer",
            raw_metric=1.4,
            normalized_score=1.4,
        )


def test_raw_metric_is_unconstrained():
    """Raw metrics keep their natural units — latency in ms, cost in USD."""
    record = EvaluationRecord(
        experiment_id="exp-1",
        question_id="fb-0001",
        action_id=Action.HYBRID_RERANK,
        scorer_name="LatencyScorer",
        raw_metric=3412.7,
        normalized_score=0.42,
    )
    assert record.raw_metric == 3412.7
    assert not record.is_llm_judged


def test_llm_judged_records_are_identifiable():
    """Judge-derived scores must be separable from deterministic ones."""
    record = EvaluationRecord(
        experiment_id="exp-1",
        question_id="fb-0001",
        action_id=Action.BM25,
        scorer_name="CitationSupportScorer",
        raw_metric=0.8,
        normalized_score=0.8,
        judge_model="judge-deployment-a",
        judge_prompt_version="v3",
    )
    assert record.is_llm_judged


# ---------------------------------------------------------------------------
# BanditLogRecord
# ---------------------------------------------------------------------------
def _log(**overrides) -> BanditLogRecord:
    kwargs = {
        "round_id": 0,
        "question_id": "fb-0001",
        "context_vector": [0.2, 0.8, 1.0],
        "behavior_policy": "uniform",
        "selected_action": Action.BM25,
        "action_propensity": 1 / 7,
        "observed_reward": 0.63,
        "reward_profile": "balanced_enterprise:v1",
        "simulation_seed": 7,
    }
    kwargs.update(overrides)
    return BanditLogRecord(**kwargs)


def test_bandit_log_round_trip():
    log = _log(context_feature_names=["bm25_margin", "dense_margin", "has_numeral"])
    assert log.selected_action is Action.BM25
    assert len(log.context_feature_names) == len(log.context_vector)


@pytest.mark.parametrize("propensity", [0.0, -0.1])
def test_zero_or_negative_propensity_is_rejected(propensity: float):
    """IPS divides by the propensity.

    A zero propensity means the action could never have been logged, which makes
    the estimator undefined rather than merely high-variance — so it is rejected
    at write time instead of producing an infinity during analysis.
    """
    with pytest.raises(ValidationError):
        _log(action_propensity=propensity)


def test_propensity_above_one_is_rejected():
    with pytest.raises(ValidationError):
        _log(action_propensity=1.2)


def test_context_feature_names_must_match_vector_length():
    with pytest.raises(ValidationError, match="context_feature_names"):
        _log(context_feature_names=["only_one"])


def test_context_feature_names_may_be_omitted():
    assert _log().context_feature_names == []
