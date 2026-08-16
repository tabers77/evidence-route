"""Tests for the Evallab boundary: episode translation, scorers, pipeline.

Marked integration because they need Evallab installed. They skip cleanly when
it is absent, since it is a local editable dependency during development.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agent_eval", reason="Evallab not installed; run `pip install -e ../evallab`")

from evidence_route.evallab_adapters.episode import (
    episode_id_for,
    outcome_to_episode,
    outcomes_to_episodes,
)
from evidence_route.evallab_adapters.pipeline import (
    build_pipeline,
    evaluation_records_from,
    reward_from_score_vector,
    score_episodes,
)
from evidence_route.evaluation.rewards import get_profile
from evidence_route.evaluation.scorers import (
    AbstentionScorer,
    CitationSupportScorer,
    CostScorer,
    NumericalAnswerScorer,
    RetrievalRecallScorer,
    default_scorers,
    extract_numbers,
)
from evidence_route.storage.records import (
    Citation,
    LatencyBreakdown,
    QuestionRecord,
    RetrievedItem,
    TokenUsage,
    WorkflowOutcomeRecord,
)
from evidence_route.workflows.actions import AbstentionReason, Action

pytestmark = pytest.mark.integration


def _question(**overrides) -> QuestionRecord:
    kwargs = {
        "question_id": "fb-1",
        "dataset": "financebench",
        "split": "dev",
        "question_text": "What were total revenues?",
        "document_ids": ["ACME_10K"],
        "reference_answer": "$66,608 million",
        "reference_evidence": ["ACME_10K::p3"],
    }
    kwargs.update(overrides)
    return QuestionRecord(**kwargs)


def _item(ordinal: int) -> RetrievedItem:
    return RetrievedItem(
        chunk_id=f"ACME_10K::c{ordinal:04d}",
        document_id="ACME_10K",
        rank=ordinal + 1,
        text=f"chunk {ordinal}",
        score=1.0 / (ordinal + 1),
    )


def _outcome(**overrides) -> WorkflowOutcomeRecord:
    kwargs = {
        "experiment_id": "exp-1",
        "question_id": "fb-1",
        "action_id": Action.BM25,
        "model_configuration": {"model": "m", "prompt_version": "grounded_v1", "top_k": 10},
        "retrieved_items": [_item(0), _item(1)],
        "answer": "$66,608 million",
        "citations": [
            Citation(chunk_id="ACME_10K::c0000", document_id="ACME_10K", quoted_text="q")
        ],
        "confidence": 0.9,
        "latency_breakdown": LatencyBreakdown(total_ms=1500.0, retrieval_ms=20.0),
        "token_usage": TokenUsage(input_tokens=1000, output_tokens=50),
        "estimated_cost_usd": 0.01,
        "metadata": {
            "pricing_known": True,
            "hallucinated_citations": [],
            "n_hallucinated_citations": 0,
        },
    }
    kwargs.update(overrides)
    return WorkflowOutcomeRecord(**kwargs)


def _episode(**overrides):
    gold = overrides.pop("gold_chunk_ids", {"ACME_10K::c0000"})
    return outcome_to_episode(_outcome(**overrides), _question(), gold_chunk_ids=gold)


# ---------------------------------------------------------------------------
# Episode translation
# ---------------------------------------------------------------------------
def test_episode_id_identifies_a_matrix_cell():
    assert episode_id_for(_outcome()) == "exp-1::fb-1::A1_bm25"


def test_episode_carries_experimental_identity():
    metadata = _episode().metadata
    assert metadata["experiment_id"] == "exp-1"
    assert metadata["workflow_action"] == "A1_bm25"
    assert metadata["dataset"] == "financebench"
    assert metadata["split"] == "dev"


def test_episode_carries_ground_truth_so_scorers_need_no_side_channel():
    """Evallab's Scorer protocol takes only an Episode."""
    metadata = _episode().metadata
    assert metadata["reference_answer"] == "$66,608 million"
    assert metadata["gold_chunk_ids"] == ["ACME_10K::c0000"]


def test_retrieval_is_expressed_as_a_tool_call():
    """So agent-specific scorers can be applied to deterministic workflows too."""
    from agent_eval.core.models import StepKind

    episode = _episode()
    kinds = [s.kind for s in episode.steps]
    assert StepKind.TOOL_CALL in kinds
    assert StepKind.TOOL_RESULT in kinds

    result_step = next(s for s in episode.steps if s.kind is StepKind.TOOL_RESULT)
    assert len(result_step.tool_result) == 2
    assert result_step.tool_succeeded is True


def test_generation_step_carries_token_counts():
    from agent_eval.core.models import StepKind

    step = next(s for s in _episode().steps if s.kind is StepKind.LLM_CALL)
    assert step.prompt_tokens == 1000
    assert step.completion_tokens == 50


def test_direct_action_has_no_retrieval_steps():
    from agent_eval.core.models import StepKind

    episode = _episode(action_id=Action.DIRECT, retrieved_items=[], citations=[])
    assert StepKind.TOOL_CALL not in [s.kind for s in episode.steps]


def test_abstention_appears_as_a_step():
    episode = _episode(
        answer=None,
        citations=[],
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    assert episode.metadata["abstained"] is True
    assert any(s.metadata.get("phase") == "abstention" for s in episode.steps)


def test_batch_skips_outcomes_without_their_question():
    """An outcome with no question has no reference answer, so it would score
    zero on every dimension rather than being absent."""
    episodes = outcomes_to_episodes([_outcome(question_id="ghost")], {"fb-1": _question()})
    assert episodes == []


# ---------------------------------------------------------------------------
# Scorers implement Evallab's protocol structurally
# ---------------------------------------------------------------------------
def test_scorers_satisfy_the_protocol_without_inheriting():
    """The property that keeps the two repositories decoupled."""
    from agent_eval.core.protocols import Scorer

    for scorer in default_scorers():
        assert isinstance(scorer, Scorer)
        assert type(scorer).__mro__[1:] == (object,), "should not inherit from Evallab"


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------
def test_retrieval_recall_is_measured():
    dimensions = {d.name: d.value for d in RetrievalRecallScorer().score(_episode())}
    assert dimensions["retrieval_recall"] == 1.0
    assert dimensions["retrieval_hit"] == 1.0


def test_missed_evidence_scores_zero_and_raises_an_issue():
    episode = _episode(gold_chunk_ids={"ACME_10K::c9999"})
    dimensions = {d.name: d.value for d in RetrievalRecallScorer().score(episode)}
    assert dimensions["retrieval_recall"] == 0.0
    assert RetrievalRecallScorer().detect_issues(episode)[0].category == "retrieval_failure"


def test_unmeasurable_retrieval_emits_nothing_rather_than_zero():
    """Zero would blend "retrieval failed" with "we could not tell"."""
    episode = _episode(gold_chunk_ids=set())
    assert RetrievalRecallScorer().score(episode) == []
    assert RetrievalRecallScorer().detect_issues(episode) == []


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def test_hallucinated_citation_is_critical():
    episode = _episode(
        metadata={
            "pricing_known": True,
            "hallucinated_citations": ["GHOST::c1"],
            "n_hallucinated_citations": 1,
        }
    )
    issues = CitationSupportScorer().detect_issues(episode)
    assert issues[0].category == "hallucinated_citation"
    assert issues[0].severity.value == "CRITICAL"


def test_citation_precision_reflects_hallucinations():
    episode = _episode(
        metadata={
            "pricing_known": True,
            "hallucinated_citations": ["GHOST::c1"],
            "n_hallucinated_citations": 1,
        }
    )
    dimensions = {d.name: d.value for d in CitationSupportScorer().score(episode)}
    assert dimensions["citation_precision"] == pytest.approx(0.5)


def test_abstention_is_not_penalised_for_having_no_citations():
    """Penalising it would discourage the reliable behaviour A6 exists for."""
    episode = _episode(
        answer=None, citations=[], abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE
    )
    assert CitationSupportScorer().score(episode) == []


def test_uncited_grounded_answer_is_an_issue():
    episode = _episode(citations=[])
    categories = [i.category for i in CitationSupportScorer().detect_issues(episode)]
    assert "uncited_answer" in categories


def test_direct_action_is_not_faulted_for_lacking_citations():
    """A0 has no evidence to cite by construction."""
    episode = _episode(action_id=Action.DIRECT, retrieved_items=[], citations=[])
    categories = [i.category for i in CitationSupportScorer().detect_issues(episode)]
    assert "uncited_answer" not in categories


# ---------------------------------------------------------------------------
# Numeric correctness
# ---------------------------------------------------------------------------
def test_numbers_are_extracted_with_scale():
    assert extract_numbers("$1,577 million")[0] == pytest.approx(1_577_000_000)
    assert extract_numbers("66,608")[0] == pytest.approx(66608)


def test_percentages_are_not_scaled():
    assert extract_numbers("23.4%")[0] == pytest.approx(23.4)


def test_accounting_negatives_are_negative():
    assert extract_numbers("(3,547)")[0] == pytest.approx(-3547)


def test_matching_answer_scores_one():
    dimensions = NumericalAnswerScorer().score(_episode())
    assert dimensions[0].value == 1.0


def test_wrong_number_scores_zero_with_an_issue():
    episode = _episode(answer="$12,345 million")
    assert NumericalAnswerScorer().score(episode)[0].value == 0.0
    assert NumericalAnswerScorer().detect_issues(episode)[0].category == (
        "incorrect_numeric_answer"
    )


def test_tolerance_is_relative_and_declared():
    scorer = NumericalAnswerScorer(relative_tolerance=0.01)
    assert scorer._within_tolerance(66608, 66600)  # 0.01% — rounding
    assert not scorer._within_tolerance(66608, 65000)  # 2.4% — a different figure


def test_tolerance_boundary_is_explicit():
    """The default 1% is generous on large figures: it accepts ±666 on 66,608.

    Pinned here so the choice is visible rather than incidental. The value is a
    protocol decision recorded in protocol_v1.md, not an implementation detail —
    tightening it later changes which answers count as correct.
    """
    scorer = NumericalAnswerScorer(relative_tolerance=0.01)
    assert scorer._within_tolerance(66608, 66000)  # 0.91% — inside
    assert not scorer._within_tolerance(66608, 65900)  # 1.06% — outside

    strict = NumericalAnswerScorer(relative_tolerance=0.001)
    assert not strict._within_tolerance(66608, 66000)


def test_non_numeric_reference_is_out_of_scope():
    """Semantic comparison belongs to a judge, not this scorer."""
    episode = outcome_to_episode(
        _outcome(answer="Yes, margins improved."),
        _question(reference_answer="Yes, margins improved."),
    )
    assert NumericalAnswerScorer().score(episode) == []


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------
def test_abstaining_without_evidence_is_correct():
    episode = _episode(
        answer=None,
        citations=[],
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        gold_chunk_ids={"ACME_10K::c9999"},
    )
    assert AbstentionScorer().score(episode)[0].value == 1.0


def test_abstaining_with_evidence_present_is_over_cautious():
    episode = _episode(
        answer=None, citations=[], abstention_reason=AbstentionReason.POLICY_LOW_CONFIDENCE
    )
    assert AbstentionScorer().score(episode)[0].value == 0.0
    assert AbstentionScorer().detect_issues(episode)[0].category == ("over_cautious_abstention")


def test_answering_without_retrieved_evidence_is_critical():
    """The failure mode abstention exists to prevent."""
    episode = _episode(gold_chunk_ids={"ACME_10K::c9999"})
    issues = AbstentionScorer().detect_issues(episode)
    assert issues[0].category == "unsupported_answer"
    assert issues[0].severity.value == "CRITICAL"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def test_unknown_pricing_emits_no_cost_score():
    """A free-looking score would make expensive workflows cheap in the reward."""
    episode = _episode(metadata={"pricing_known": False})
    assert CostScorer().score(episode) == []
    assert CostScorer().detect_issues(episode)[0].category == "cost_unmeasured"


def test_known_pricing_scores_cost():
    dimension = CostScorer(reference_usd=0.05).score(_episode())[0]
    assert dimension.value == pytest.approx(1.0 - 0.01 / 0.05)


# ---------------------------------------------------------------------------
# End to end through Evallab
# ---------------------------------------------------------------------------
def test_pipeline_produces_a_score_vector():
    scored = score_episodes([_episode()], pipeline=build_pipeline())[0]

    assert scored.episode_id == "exp-1::fb-1::A1_bm25"
    assert scored.score_vector.dimensions
    assert scored.grade
    assert "retrieval_recall" in scored.dimensions


def test_pipeline_computes_a_reward_under_a_profile():
    scored = score_episodes([_episode()], profile=get_profile("balanced_enterprise"))[0]

    assert scored.reward is not None
    assert scored.reward.profile_name == "balanced_enterprise:v1"
    assert -3.0 <= scored.reward.total <= 1.0


def test_reward_is_not_evallabs_grade():
    """Evallab's overall score is a generic trace grade, not a policy value."""
    scored = score_episodes([_episode()], profile=get_profile("quality_first"))[0]
    overall = scored.score_vector.dimension_by_name("overall_score")
    assert overall is not None
    assert scored.reward is not None
    assert scored.reward.total != overall.value


def test_cost_score_is_inverted_back_into_a_penalty():
    """A goodness score fed in as a penalty would flip the cost term's sign."""
    cheap = _episode(estimated_cost_usd=0.001)
    dear = _episode(estimated_cost_usd=0.05)
    profile = get_profile("cost_sensitive")

    cheap_reward = reward_from_score_vector(score_episodes([cheap])[0].score_vector, profile)
    dear_reward = reward_from_score_vector(score_episodes([dear])[0].score_vector, profile)
    assert cheap_reward.cost_penalty < dear_reward.cost_penalty


def test_failed_outcomes_are_scored_not_dropped():
    """A workflow that errors on hard questions must not look better for it."""
    episode = _episode(
        answer=None, citations=[], error_status="provider_error", error_detail="rate limited"
    )
    scored = score_episodes([episode])[0]
    assert scored.dimensions["policy_compliance"] == 0.0
    assert any(i.category == "workflow_provider_error" for i in scored.score_vector.issues)


def test_evaluation_records_carry_raw_and_normalized_values():
    """Re-normalizing later must not require re-scoring."""
    scored = score_episodes([_episode()])[0]
    records = evaluation_records_from(scored)

    assert records
    for record in records:
        assert record.experiment_id == "exp-1"
        assert record.action_id is Action.BM25
        assert 0.0 <= record.normalized_score <= 1.0
        assert "dimension" in record.evaluation_metadata
