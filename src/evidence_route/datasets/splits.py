"""Deterministic grouped split generation (spec section 7.4).

Splits are grouped by company or document, never sampled per question.

The reason is specific rather than stylistic: several FinanceBench questions
reference the same filing. Under question-level random splitting, a model can
learn document-specific patterns during training and be rewarded for them at
test time. The split leaks, the held-out estimate is optimistic, and *no
confidence interval reveals it* — the uncertainty would be computed over the
same contaminated data. Grouping is the only place this can be prevented.

Determinism matters equally. A split regenerated with a different seed is a
different experiment, so the seed is recorded, the assignment is reproducible
from it, and frozen test identifiers are committed to git.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from evidence_route.datasets.base import RawQuestion
from evidence_route.storage.records import SplitName

__all__ = [
    "SplitAssignment",
    "assert_no_group_leakage",
    "generate_grouped_splits",
    "load_frozen_splits",
    "write_splits",
]


@dataclass(frozen=True)
class SplitAssignment:
    """Which split each question and each group landed in."""

    #: question_id -> split
    split_of_question: dict[str, SplitName]
    #: group_key -> split
    split_of_group: dict[str, SplitName]
    seed: int
    group_by: str

    def question_ids(self, split: SplitName) -> list[str]:
        """Question identifiers in one split, sorted for stable output."""
        return sorted(q for q, s in self.split_of_question.items() if s == split)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for split in self.split_of_question.values():
            counts[split] += 1
        return dict(counts)

    def group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for split in self.split_of_group.values():
            counts[split] += 1
        return dict(counts)


def generate_grouped_splits(
    questions: list[RawQuestion],
    *,
    fractions: dict[str, float],
    seed: int,
    group_by: str = "company",
) -> SplitAssignment:
    """Assign whole groups to splits, approximating the requested proportions.

    Groups are shuffled with a seeded RNG, then assigned greedily to whichever
    split is furthest below its target *question* count. Balancing on questions
    rather than groups matters because group sizes are uneven — one company with
    forty questions and ten with two would otherwise skew the split badly.

    The proportions are approximate by construction. A group cannot be divided
    without reintroducing the leakage this function exists to prevent, so exact
    fractions and leak-free splits are mutually exclusive; the leak-free
    property wins.
    """
    if not questions:
        raise ValueError("Cannot generate splits from an empty question set.")
    if not fractions:
        raise ValueError("No split fractions were provided.")

    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_fraction:.6f} from {fractions}."
        )
    if any(v <= 0 for v in fractions.values()):
        raise ValueError(f"Split fractions must all be positive, got {fractions}.")

    duplicates = _duplicate_ids(questions)
    if duplicates:
        raise ValueError(
            f"Duplicate question_ids would be assigned to different splits: "
            f"{sorted(duplicates)[:5]}"
        )

    groups: dict[str, list[RawQuestion]] = defaultdict(list)
    for question in questions:
        groups[question.group_key].append(question)

    split_names = sorted(fractions)
    if len(groups) < len(split_names):
        raise ValueError(
            f"Only {len(groups)} group(s) available for {len(split_names)} splits. "
            f"Grouping by {group_by!r} cannot produce a non-empty split for each."
        )

    # Sort before shuffling so the ordering does not depend on dict insertion
    # order, which varies with the order rows appeared in the source file.
    ordered_groups = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered_groups)
    # The shuffled position becomes the tie-breaker below. Using the group name
    # instead would make the seed inert whenever groups are the same size —
    # every seed would produce an identical split.
    shuffled_rank = {group: i for i, group in enumerate(ordered_groups)}

    total_questions = len(questions)
    targets = {name: fractions[name] * total_questions for name in split_names}
    assigned: dict[str, int] = dict.fromkeys(split_names, 0)

    split_of_group: dict[str, SplitName] = {}
    # Largest groups first: placing the big, hard-to-balance groups while every
    # split still has room produces noticeably closer proportions than
    # processing them in arbitrary order. Ties break on the seeded shuffle, so
    # the result is both size-balanced and genuinely seed-dependent.
    for group_key in sorted(groups, key=lambda g: (-len(groups[g]), shuffled_rank[g])):
        deficits = {name: targets[name] - assigned[name] for name in split_names}
        # Ties resolve by split name, keeping the result deterministic.
        chosen = max(split_names, key=lambda name: (deficits[name], name))
        split_of_group[group_key] = chosen  # type: ignore[assignment]
        assigned[chosen] += len(groups[group_key])

    split_of_question: dict[str, SplitName] = {}
    for group_key, members in groups.items():
        split = split_of_group[group_key]
        for question in members:
            split_of_question[question.question_id] = split

    return SplitAssignment(
        split_of_question=split_of_question,
        split_of_group=split_of_group,
        seed=seed,
        group_by=group_by,
    )


def _duplicate_ids(questions: list[RawQuestion]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for question in questions:
        if question.question_id in seen:
            duplicates.add(question.question_id)
        seen.add(question.question_id)
    return duplicates


def assert_no_group_leakage(questions: list[RawQuestion], assignment: SplitAssignment) -> None:
    """Verify no group has questions in more than one split.

    This is the property the whole grouped-splitting design exists to guarantee,
    so it is asserted explicitly rather than assumed from the algorithm. It is
    cheap, and it turns a subtle statistical error into a loud failure.
    """
    splits_per_group: dict[str, set[str]] = defaultdict(set)
    for question in questions:
        split = assignment.split_of_question.get(question.question_id)
        if split is None:
            raise AssertionError(
                f"Question {question.question_id!r} was not assigned to any split."
            )
        splits_per_group[question.group_key].add(split)

    leaking = {g: sorted(s) for g, s in splits_per_group.items() if len(s) > 1}
    if leaking:
        raise AssertionError(
            f"Group leakage detected — these groups span multiple splits: {leaking}"
        )


def write_splits(
    assignment: SplitAssignment,
    output_dir: Path,
    *,
    dataset: str,
    dataset_version: str,
) -> Path:
    """Freeze the assignment to disk as committed, human-reviewable JSON.

    Written sorted and indented on purpose: these files go into git, and a
    reviewer needs to be able to diff them and see that the test set did not
    change between runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "dataset_version": dataset_version,
        "group_by": assignment.group_by,
        "seed": assignment.seed,
        "counts": assignment.counts(),
        "group_counts": assignment.group_counts(),
        "splits": {
            split: assignment.question_ids(split)
            for split in sorted(set(assignment.split_of_question.values()))
        },
        "groups": {
            split: sorted(g for g, s in assignment.split_of_group.items() if s == split)
            for split in sorted(set(assignment.split_of_group.values()))
        },
    }
    path = output_dir / "splits.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_frozen_splits(path: Path) -> SplitAssignment:
    """Read a frozen split file back.

    Experiments load the committed assignment rather than regenerating it, so a
    change in the splitting code cannot silently reshuffle a benchmark whose
    results have already been reported.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_of_question: dict[str, SplitName] = {}
    for split, ids in payload["splits"].items():
        for question_id in ids:
            split_of_question[question_id] = split
    split_of_group: dict[str, SplitName] = {}
    for split, group_keys in payload.get("groups", {}).items():
        for group_key in group_keys:
            split_of_group[group_key] = split
    return SplitAssignment(
        split_of_question=split_of_question,
        split_of_group=split_of_group,
        seed=payload["seed"],
        group_by=payload.get("group_by", "unknown"),
    )
