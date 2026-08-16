"""Paired retrieval comparison (spec section 21, week 4 exit criteria).

This module answers the question the whole project depends on: **do different
retrievers fail on different questions?**

If they succeed and fail together, routing has nothing to exploit and the honest
conclusion is to pick the best single retriever. If they disagree substantially,
there is signal a policy could learn. Either answer is a finding; the point is
to measure it rather than assume it.

The comparison is **paired** — the same questions through every retriever —
which is what makes the disagreement counts meaningful. Knowing that BM25 scored
0.82 and dense 0.79 across different samples tells you almost nothing; knowing
*which* questions each one won tells you whether a router could beat both.

The disagreement counts form a McNemar contingency table, so the week-11
statistical protocol can test the difference properly rather than eyeballing two
means.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ComparisonReport",
    "PairwiseDisagreement",
    "QuestionResult",
    "compare_retrievers",
]


@dataclass
class QuestionResult:
    """One retriever's outcome on one question."""

    question_id: str
    retriever: str
    recall: float
    hit: bool
    n_gold: int
    measurable: bool = True


@dataclass
class PairwiseDisagreement:
    """A McNemar contingency table for two retrievers on hit@k.

    ``a_only`` and ``b_only`` are the discordant pairs — the only cells McNemar's
    test uses, and the only ones that tell you whether routing could help.
    """

    retriever_a: str
    retriever_b: str
    both: int = 0
    a_only: int = 0
    b_only: int = 0
    neither: int = 0

    @property
    def total(self) -> int:
        return self.both + self.a_only + self.b_only + self.neither

    @property
    def discordant(self) -> int:
        """Questions where exactly one retriever succeeded."""
        return self.a_only + self.b_only

    @property
    def disagreement_rate(self) -> float:
        """Share of questions where the two disagree.

        The headline number for whether routing has anything to exploit. Near
        zero means the retrievers are interchangeable and a router cannot help.
        """
        return self.discordant / self.total if self.total else 0.0

    @property
    def complementarity(self) -> float:
        """Gain an oracle choosing the better retriever per question would get.

        The ceiling on what routing between exactly these two could achieve —
        not what any real router will reach, but the bound it works within.
        """
        if not self.total:
            return 0.0
        best_single = max(self.both + self.a_only, self.both + self.b_only)
        oracle = self.both + self.a_only + self.b_only
        return (oracle - best_single) / self.total

    def summary(self) -> str:
        return (
            f"{self.retriever_a} vs {self.retriever_b}: "
            f"both {self.both}, {self.retriever_a} only {self.a_only}, "
            f"{self.retriever_b} only {self.b_only}, neither {self.neither} "
            f"(disagreement {self.disagreement_rate:.1%}, "
            f"oracle gain {self.complementarity:+.1%})"
        )


@dataclass
class ComparisonReport:
    """Paired comparison across several retrievers."""

    k: int
    results: list[QuestionResult] = field(default_factory=list)
    disagreements: list[PairwiseDisagreement] = field(default_factory=list)
    #: Set when any retriever used a non-semantic embedder. Conclusions about
    #: dense retrieval are not valid in that case.
    warnings: list[str] = field(default_factory=list)

    @property
    def retrievers(self) -> list[str]:
        return sorted({r.retriever for r in self.results})

    @property
    def n_questions(self) -> int:
        return len({r.question_id for r in self.results})

    @property
    def n_measurable(self) -> int:
        return len({r.question_id for r in self.results if r.measurable})

    def mean_recall(self, retriever: str) -> float | None:
        values = [r.recall for r in self.results if r.retriever == retriever and r.measurable]
        return sum(values) / len(values) if values else None

    def hit_rate(self, retriever: str) -> float | None:
        values = [r.hit for r in self.results if r.retriever == retriever and r.measurable]
        return sum(values) / len(values) if values else None

    def oracle_hit_rate(self) -> float:
        """Hit rate if the best retriever were chosen per question.

        The upper bound on routing over this action set — useful as a sanity
        check on any later router, which cannot exceed it.
        """
        by_question: dict[str, bool] = {}
        for result in self.results:
            if not result.measurable:
                continue
            by_question[result.question_id] = (
                by_question.get(result.question_id, False) or result.hit
            )
        return sum(by_question.values()) / len(by_question) if by_question else 0.0

    def summary(self) -> str:
        lines = [
            f"Paired retrieval comparison — {self.n_measurable}/{self.n_questions} "
            f"questions measurable, k={self.k}"
        ]
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        lines.append("")
        for retriever in self.retrievers:
            recall = self.mean_recall(retriever)
            hit = self.hit_rate(retriever)
            lines.append(
                f"  {retriever:<24} recall@{self.k} "
                f"{'n/a' if recall is None else f'{recall:.3f}'}   "
                f"hit@{self.k} {'n/a' if hit is None else f'{hit:.3f}'}"
            )
        lines.append(
            f"  {'ORACLE (best per question)':<24} hit@{self.k} {self.oracle_hit_rate():.3f}"
        )
        lines.append("")
        for disagreement in self.disagreements:
            lines.append(f"  {disagreement.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_questions": self.n_questions,
            "n_measurable": self.n_measurable,
            "warnings": self.warnings,
            "oracle_hit_rate": round(self.oracle_hit_rate(), 4),
            "retrievers": {
                name: {
                    "mean_recall": self.mean_recall(name),
                    "hit_rate": self.hit_rate(name),
                }
                for name in self.retrievers
            },
            "disagreements": [
                {
                    "a": d.retriever_a,
                    "b": d.retriever_b,
                    "both": d.both,
                    "a_only": d.a_only,
                    "b_only": d.b_only,
                    "neither": d.neither,
                    "disagreement_rate": round(d.disagreement_rate, 4),
                    "complementarity": round(d.complementarity, 4),
                }
                for d in self.disagreements
            ],
        }


def compare_retrievers(
    results: list[QuestionResult],
    *,
    k: int,
    non_semantic_retrievers: list[str] | None = None,
) -> ComparisonReport:
    """Build a paired comparison from per-question results.

    Only questions measurable by *every* retriever are compared. Including a
    question one retriever could not be scored on would make the pairing
    unbalanced, and an unbalanced pairing is exactly what paired analysis exists
    to avoid.
    """
    report = ComparisonReport(k=k, results=results)

    for name in sorted(non_semantic_retrievers or []):
        report.warnings.append(
            f"{name!r} used a non-semantic embedder. It measures literal token "
            f"overlap, which is what BM25 already does — so any conclusion here "
            f"about dense retrieval is an artifact of the stub, not a finding."
        )

    retrievers = report.retrievers
    measurable_counts = Counter(r.question_id for r in results if r.measurable)
    complete = {q for q, count in measurable_counts.items() if count == len(retrievers)}

    lookup: dict[tuple[str, str], QuestionResult] = {
        (r.question_id, r.retriever): r for r in results
    }

    for i, a in enumerate(retrievers):
        for b in retrievers[i + 1 :]:
            table = PairwiseDisagreement(retriever_a=a, retriever_b=b)
            for question_id in complete:
                hit_a = lookup[(question_id, a)].hit
                hit_b = lookup[(question_id, b)].hit
                if hit_a and hit_b:
                    table.both += 1
                elif hit_a:
                    table.a_only += 1
                elif hit_b:
                    table.b_only += 1
                else:
                    table.neither += 1
            report.disagreements.append(table)

    return report
