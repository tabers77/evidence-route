"""Tests for the routing action space."""

from __future__ import annotations

import pytest

from evidence_route.workflows.actions import (
    ACTION_SPACE,
    AbstentionReason,
    Action,
    action_from_index,
    action_index,
)


def test_action_space_covers_every_action():
    assert set(ACTION_SPACE) == set(Action)
    assert len(ACTION_SPACE) == 7


def test_action_values_are_stable_identifiers():
    """Action values are written into outcome and bandit-log files.

    Renaming one silently invalidates every stored experiment, so the exact
    strings are pinned here as a tripwire.
    """
    assert [a.value for a in ACTION_SPACE] == [
        "A0_direct",
        "A1_bm25",
        "A2_dense",
        "A3_hybrid",
        "A4_hybrid_rerank",
        "A5_agentic",
        "A6_abstain",
    ]


def test_index_round_trip():
    for action in ACTION_SPACE:
        assert action_from_index(action_index(action)) is action


def test_direct_action_does_not_retrieve():
    assert not Action.DIRECT.uses_retrieval
    assert Action.DIRECT.produces_answer


def test_abstain_is_the_only_non_answering_action():
    non_answering = [a for a in Action if not a.produces_answer]
    assert non_answering == [Action.ABSTAIN]


def test_answering_actions_excludes_abstain():
    assert Action.ABSTAIN not in Action.answering_actions()
    assert len(Action.answering_actions()) == 6


@pytest.mark.parametrize(
    "action",
    [Action.BM25, Action.DENSE, Action.HYBRID, Action.HYBRID_RERANK, Action.AGENTIC],
)
def test_retrieving_actions(action: Action):
    assert action.uses_retrieval


def test_abstention_reasons_are_machine_readable():
    assert AbstentionReason.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
    # All reason codes are lowercase snake_case so they survive a round trip
    # through Parquet, JSON and report tables without normalization.
    for reason in AbstentionReason:
        assert reason.value == reason.value.lower()
        assert " " not in reason.value
