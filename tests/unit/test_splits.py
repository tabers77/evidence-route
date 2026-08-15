"""Tests for deterministic grouped split generation.

The leakage property is the one that protects every downstream comparison, so
it is tested directly rather than inferred from the algorithm.
"""

from __future__ import annotations

import json

import pytest

from evidence_route.datasets.base import RawQuestion
from evidence_route.datasets.splits import (
    assert_no_group_leakage,
    generate_grouped_splits,
    load_frozen_splits,
    write_splits,
)

FRACTIONS = {"dev": 0.3, "validation": 0.25, "test": 0.45}


def _questions(groups: dict[str, int], dataset: str = "financebench") -> list[RawQuestion]:
    """Build questions from a {group_key: question_count} mapping."""
    out: list[RawQuestion] = []
    for group, count in groups.items():
        for i in range(count):
            out.append(
                RawQuestion(
                    question_id=f"{group}-{i}",
                    dataset=dataset,
                    question_text=f"question {i} about {group}",
                    group_key=group,
                    document_ids=(f"{group}_10K",),
                    reference_answer="42",
                    reference_evidence=(f"{group}_10K::p1",),
                )
            )
    return out


def _even_groups(n: int, per_group: int = 4) -> list[RawQuestion]:
    return _questions({f"company{i:02d}": per_group for i in range(n)})


# ---------------------------------------------------------------------------
# The core guarantee
# ---------------------------------------------------------------------------
def test_no_group_spans_two_splits():
    """The property the whole design exists for.

    Questions about one company must never appear in two splits — otherwise a
    model can learn company-specific patterns in training and be rewarded for
    them at test time, and no confidence interval would reveal it.
    """
    questions = _even_groups(20)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=1)
    assert_no_group_leakage(questions, assignment)

    for group in {q.group_key for q in questions}:
        splits = {
            assignment.split_of_question[q.question_id] for q in questions if q.group_key == group
        }
        assert len(splits) == 1, f"{group} spans {splits}"


def test_leakage_assertion_actually_catches_leakage():
    """A guard that cannot fail is not a guard."""
    questions = _even_groups(10)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=1)

    # Move one question of a group into a different split by hand.
    victim = questions[0]
    other = next(
        s
        for s in ("dev", "validation", "test")
        if s != assignment.split_of_question[victim.question_id]
    )
    assignment.split_of_question[victim.question_id] = other  # type: ignore[assignment]

    with pytest.raises(AssertionError, match="Group leakage detected"):
        assert_no_group_leakage(questions, assignment)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_splits():
    questions = _even_groups(20)
    a = generate_grouped_splits(questions, fractions=FRACTIONS, seed=20260815)
    b = generate_grouped_splits(questions, fractions=FRACTIONS, seed=20260815)
    assert a.split_of_question == b.split_of_question
    assert a.split_of_group == b.split_of_group


def test_input_order_does_not_change_the_result():
    """Splits must not depend on the order rows happened to appear in the file."""
    questions = _even_groups(20)
    shuffled = list(reversed(questions))
    a = generate_grouped_splits(questions, fractions=FRACTIONS, seed=7)
    b = generate_grouped_splits(shuffled, fractions=FRACTIONS, seed=7)
    assert a.split_of_question == b.split_of_question


def test_different_seeds_generally_differ():
    questions = _even_groups(30)
    a = generate_grouped_splits(questions, fractions=FRACTIONS, seed=1)
    b = generate_grouped_splits(questions, fractions=FRACTIONS, seed=2)
    assert a.split_of_group != b.split_of_group


# ---------------------------------------------------------------------------
# Coverage and proportions
# ---------------------------------------------------------------------------
def test_every_question_assigned_exactly_once():
    questions = _even_groups(20)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=3)
    assert set(assignment.split_of_question) == {q.question_id for q in questions}
    assert sum(assignment.counts().values()) == len(questions)


def test_all_requested_splits_are_populated():
    questions = _even_groups(20)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=3)
    assert set(assignment.counts()) == set(FRACTIONS)
    assert all(count > 0 for count in assignment.counts().values())


def test_proportions_are_approximately_respected():
    """Approximate by construction — a group cannot be divided without leaking."""
    questions = _even_groups(50, per_group=4)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=11)
    total = len(questions)
    for split, target in FRACTIONS.items():
        actual = assignment.counts()[split] / total
        assert abs(actual - target) < 0.10, f"{split}: {actual:.2f} vs target {target}"


def test_uneven_group_sizes_still_balance_by_question_count():
    """Balancing on questions, not groups — one huge group must not skew a split."""
    questions = _questions({"mega": 40, **{f"small{i:02d}": 2 for i in range(30)}})
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=5)
    assert_no_group_leakage(questions, assignment)
    total = len(questions)
    for split, target in FRACTIONS.items():
        actual = assignment.counts()[split] / total
        assert abs(actual - target) < 0.20, f"{split}: {actual:.2f} vs target {target}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty question set"):
        generate_grouped_splits([], fractions=FRACTIONS, seed=1)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        generate_grouped_splits(_even_groups(10), fractions={"dev": 0.3, "test": 0.3}, seed=1)


def test_negative_fraction_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        generate_grouped_splits(_even_groups(10), fractions={"dev": 1.5, "test": -0.5}, seed=1)


def test_too_few_groups_for_the_requested_splits():
    """Two companies cannot produce three non-empty grouped splits."""
    questions = _questions({"only_one": 5, "and_two": 5})
    with pytest.raises(ValueError, match="cannot produce a non-empty split"):
        generate_grouped_splits(questions, fractions=FRACTIONS, seed=1)


def test_duplicate_question_ids_are_rejected():
    questions = _even_groups(10)
    questions.append(questions[0])
    with pytest.raises(ValueError, match="Duplicate question_ids"):
        generate_grouped_splits(questions, fractions=FRACTIONS, seed=1)


# ---------------------------------------------------------------------------
# Freezing and reloading
# ---------------------------------------------------------------------------
def test_write_and_reload_round_trip(tmp_path):
    questions = _even_groups(20)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=42)
    path = write_splits(assignment, tmp_path, dataset="financebench", dataset_version="v1")

    reloaded = load_frozen_splits(path)
    assert reloaded.split_of_question == assignment.split_of_question
    assert reloaded.split_of_group == assignment.split_of_group
    assert reloaded.seed == 42
    assert reloaded.group_by == assignment.group_by


def test_frozen_file_is_sorted_and_reviewable(tmp_path):
    """These files go into git; a reviewer has to be able to diff them."""
    questions = _even_groups(12)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=9)
    path = write_splits(assignment, tmp_path, dataset="financebench", dataset_version="v1")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["seed"] == 9
    assert payload["dataset"] == "financebench"
    for ids in payload["splits"].values():
        assert ids == sorted(ids)


def test_question_ids_helper_is_sorted():
    questions = _even_groups(15)
    assignment = generate_grouped_splits(questions, fractions=FRACTIONS, seed=4)
    ids = assignment.question_ids("test")
    assert ids == sorted(ids)
