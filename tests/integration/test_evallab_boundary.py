"""Tests for the EvidenceRoute <-> Evallab boundary.

Evallab is a local editable dependency during development (spec section 2.3), so
these tests skip cleanly when it is absent rather than failing a fresh clone.
What they assert is that the contract EvidenceRoute relies on actually holds:
the canonical types exist and have the shape the adapters will target.
"""

from __future__ import annotations

import pytest

agent_eval = pytest.importorskip(
    "agent_eval",
    reason="Evallab not installed; run `pip install -e ../evallab`",
)

pytestmark = pytest.mark.integration


def test_canonical_types_are_importable():
    """The canonical types live in `agent_eval.core`, not the top-level package.

    Pinned here because the adapters import from this exact path; if Evallab
    ever re-exports them at the top level, that is a widening, not a break.
    """
    from agent_eval.core import Episode, ScoreDimension, ScoreVector, Step, StepKind

    assert Episode is not None
    assert Step is not None
    assert StepKind is not None
    assert ScoreVector is not None
    assert ScoreDimension is not None


def test_scorer_protocol_is_available():
    """EvidenceRoute's domain scorers implement this protocol structurally."""
    from agent_eval.core import RewardFunction, Scorer, TraceAdapter

    assert Scorer is not None
    assert TraceAdapter is not None
    assert RewardFunction is not None


def test_episode_carries_arbitrary_metadata():
    """Experimental identity (question_id, action, seed, ...) rides in metadata.

    Evallab does not model EvidenceRoute's experiment fields, so the adapter
    depends on metadata being a free-form dict.
    """
    from agent_eval.core import Episode

    from evidence_route.workflows.actions import Action

    episode = Episode(
        episode_id="exp-1::fb-0001::A1_bm25",
        steps=[],
        source_framework="evidence_route",
        metadata={
            "question_id": "fb-0001",
            "dataset": "financebench",
            "split": "dev",
            "workflow_action": Action.BM25.value,
            "experiment_id": "exp-1",
            "random_seed": 20260815,
        },
    )
    assert episode.metadata["workflow_action"] == "A1_bm25"
    assert episode.agents == set()


def test_step_kinds_cover_the_agentic_workflow_trace():
    """The A5 trace needs tool calls and results to compute agent metrics."""
    from agent_eval.core import StepKind

    for required in ("TOOL_CALL", "TOOL_RESULT", "LLM_CALL"):
        assert hasattr(StepKind, required)


def test_score_vector_aggregates_dimensions():
    from agent_eval.core import ScoreDimension, ScoreVector

    vector = ScoreVector(
        episode_id="exp-1::fb-0001::A1_bm25",
        dimensions=[
            ScoreDimension(name="answer_correctness", value=0.9),
            ScoreDimension(name="evidence_support", value=1.0),
        ],
    )
    assert vector.dimension_by_name("answer_correctness") is not None
    assert 0.0 <= vector.overall <= 1.0
