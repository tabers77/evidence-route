"""Dataset validation.

Excluded examples are counted and reported, never silently dropped. A loader
that quietly discards the questions it cannot parse produces a benchmark that
looks cleaner than it is, and the discarded ones are rarely random — they tend
to be the hard cases with awkward evidence, which is precisely the population
the routing question cares about.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evidence_route.datasets.base import RawQuestion

__all__ = ["ValidationIssue", "ValidationReport", "validate_questions"]


@dataclass(frozen=True)
class ValidationIssue:
    """One reason one question failed validation."""

    question_id: str
    code: str
    detail: str


@dataclass
class ValidationReport:
    """The outcome of validating a parsed dataset."""

    dataset: str
    total: int
    valid: list[RawQuestion] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def excluded_ids(self) -> set[str]:
        return {issue.question_id for issue in self.issues}

    @property
    def excluded_count(self) -> int:
        return len(self.excluded_ids)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def counts_by_code(self) -> dict[str, int]:
        return dict(Counter(issue.code for issue in self.issues))

    def summary(self) -> str:
        """A one-glance summary for the CLI and for run logs."""
        lines = [
            f"{self.dataset}: {len(self.valid)} valid / {self.total} parsed"
            f" ({self.excluded_count} excluded)"
        ]
        for code, count in sorted(self.counts_by_code().items()):
            lines.append(f"  {code}: {count}")
        return "\n".join(lines)


def validate_questions(
    questions: list[RawQuestion],
    *,
    dataset: str,
    require_reference_answer: bool = True,
    require_reference_evidence: bool = True,
) -> ValidationReport:
    """Check parsed questions against the dataset's declared requirements.

    A question failing any check is excluded from ``valid`` but recorded in
    ``issues``, so the count of what was dropped and why survives into the run
    log and the data card.
    """
    report = ValidationReport(dataset=dataset, total=len(questions))

    seen: dict[str, int] = Counter(q.question_id for q in questions)

    for question in questions:
        problems: list[ValidationIssue] = []

        if seen[question.question_id] > 1:
            problems.append(
                ValidationIssue(
                    question.question_id,
                    "duplicate_question_id",
                    f"appears {seen[question.question_id]} times",
                )
            )
        if not question.question_text.strip():
            problems.append(ValidationIssue(question.question_id, "empty_question_text", "blank"))
        if not question.group_key.strip():
            problems.append(
                ValidationIssue(
                    question.question_id,
                    "missing_group_key",
                    "no grouping key; the question cannot be split without leaking",
                )
            )
        if not question.document_ids:
            problems.append(
                ValidationIssue(question.question_id, "no_document_ids", "no source document")
            )
        if require_reference_answer and not (question.reference_answer or "").strip():
            problems.append(
                ValidationIssue(question.question_id, "missing_reference_answer", "no gold answer")
            )
        if require_reference_evidence and not question.reference_evidence:
            problems.append(
                ValidationIssue(
                    question.question_id,
                    "missing_reference_evidence",
                    "no evidence spans; retrieval recall would be unmeasurable",
                )
            )

        if problems:
            report.issues.extend(problems)
        else:
            report.valid.append(question)

    return report
