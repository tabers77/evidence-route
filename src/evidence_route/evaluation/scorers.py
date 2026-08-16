"""EvidenceRoute's domain scorers (spec section 13.2).

These implement Evallab's ``Scorer`` protocol *structurally* — no base class, no
inheritance, just ``name``, ``score()`` and ``detect_issues()``. That is the
property that lets the two repositories stay genuinely decoupled, and it is
worth pointing at in the technical report: EvidenceRoute needs Evallab's types
to describe results, but Evallab needs to know nothing at all about financial
question answering.

What lives here rather than upstream: anything that understands retrieval
evidence, financial figures or abstention. A scorer that turns out to be
domain-independent can later be proposed for Evallab, but the default direction
is that domain logic stays put.

**A scorer that cannot measure something emits nothing.** Not zero. A question
whose evidence could not be resolved has no retrieval recall to report, and
recording 0.0 would blend "retrieval failed" with "we could not tell", which are
different findings. Absence is visible in the aggregate; a fabricated zero is
not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent_eval.core.models import Episode
from agent_eval.core.score import Issue, ScoreDimension, Severity

__all__ = [
    "AbstentionScorer",
    "CitationSupportScorer",
    "CostScorer",
    "LatencyScorer",
    "NumericalAnswerScorer",
    "RetrievalRecallScorer",
    "WorkflowViolationScorer",
    "default_scorers",
    "extract_numbers",
]

#: Scale words as they appear in filings and in FinanceBench answers.
_SCALES = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
}

_NUMBER_RE = re.compile(
    r"(?P<sign>-|\()?\s*[$€£]?\s*(?P<number>\d[\d,]*(?:\.\d+)?)\s*\)?\s*"
    r"(?P<scale>thousands?|millions?|billions?|trillions?|mm|bn)?",
    re.IGNORECASE,
)


def extract_numbers(text: str | None) -> list[float]:
    """Pull comparable magnitudes out of free text.

    Scale words are applied, so ``$1,577 million`` and ``1577000000`` become the
    same quantity. Without that, a correct answer phrased in millions would be
    scored wrong against a reference written in full — a units bug masquerading
    as a model error, and one that would land in the wrong failure category.

    Percentages are deliberately *not* scaled: ``23.4%`` stays 23.4, because a
    reference answer expressing a margin writes it the same way.
    """
    if not text:
        return []

    numbers: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("number").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue

        scale = match.group("scale")
        if scale:
            value *= _SCALES.get(scale.lower(), 1.0)

        # Accounting parentheses and a leading minus both mean negative.
        if match.group("sign"):
            value = -value

        numbers.append(value)
    return numbers


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass
class RetrievalRecallScorer:
    """Evidence recall, measured independently of answer quality.

    Kept separate on purpose (spec section 11.1): a correct answer can conceal
    poor retrieval when the model already knew the fact, and a wrong answer can
    hide successful retrieval when generation failed on good evidence. Scoring
    them together would make both failures invisible.
    """

    k: int = 10

    @property
    def name(self) -> str:
        return "RetrievalRecallScorer"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        gold = set(episode.metadata.get("gold_chunk_ids") or [])
        if not gold:
            # Unmeasurable, not zero — see the module docstring.
            return []

        retrieved = list(episode.metadata.get("retrieved_chunk_ids") or [])
        top_k = set(retrieved[: self.k])
        found = gold & top_k

        return [
            ScoreDimension(
                name="retrieval_recall",
                value=len(found) / len(gold),
                source=self.name,
            ),
            ScoreDimension(
                name="retrieval_hit",
                value=1.0 if found else 0.0,
                source=self.name,
            ),
            ScoreDimension(
                name="retrieval_precision",
                value=len(found) / len(top_k) if top_k else 0.0,
                source=self.name,
            ),
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        gold = set(episode.metadata.get("gold_chunk_ids") or [])
        if not gold:
            return []
        retrieved = set(list(episode.metadata.get("retrieved_chunk_ids") or [])[: self.k])
        if gold & retrieved:
            return []
        return [
            Issue(
                severity=Severity.ERROR,
                category="retrieval_failure",
                description=(
                    f"No gold evidence chunk appeared in the top {self.k}. Any "
                    f"answer produced here cannot have been grounded in the "
                    f"cited evidence."
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
@dataclass
class CitationSupportScorer:
    """Whether the answer's citations point at evidence it was actually given."""

    @property
    def name(self) -> str:
        return "CitationSupportScorer"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        if episode.metadata.get("abstained"):
            # An abstention has nothing to cite; scoring it would penalise the
            # reliable behaviour the abstention action exists to encourage.
            return []

        cited = list(episode.metadata.get("cited_chunk_ids") or [])
        hallucinated = list(episode.metadata.get("hallucinated_citations") or [])
        total = len(cited) + len(hallucinated)
        if total == 0:
            return [ScoreDimension(name="citation_coverage", value=0.0, source=self.name)]

        return [
            ScoreDimension(
                name="citation_precision",
                value=len(cited) / total,
                source=self.name,
            ),
            ScoreDimension(name="citation_coverage", value=1.0, source=self.name),
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        hallucinated = list(episode.metadata.get("hallucinated_citations") or [])
        if hallucinated:
            issues.append(
                Issue(
                    severity=Severity.CRITICAL,
                    category="hallucinated_citation",
                    description=(
                        f"Cited {len(hallucinated)} chunk(s) never retrieved: "
                        f"{', '.join(hallucinated[:3])}. The model referenced "
                        f"evidence it was not shown."
                    ),
                )
            )

        answered = not episode.metadata.get("abstained") and episode.final_answer
        # A0 is exempt: it has no evidence to cite by construction.
        grounded = episode.metadata.get("workflow_action") != "A0_direct"
        if answered and grounded and not episode.metadata.get("cited_chunk_ids"):
            issues.append(
                Issue(
                    severity=Severity.ERROR,
                    category="uncited_answer",
                    description=(
                        "A grounded workflow answered without citing any "
                        "evidence, so the claim cannot be verified."
                    ),
                )
            )
        return issues


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------
@dataclass
class NumericalAnswerScorer:
    """Numeric correctness under a declared relative tolerance.

    The tolerance is explicit because "correct" for a financial figure is a
    judgement, not a fact: filings round, and an answer of 66,608 against a
    reference of 66,608.0 is right while 66,000 is not.

    Known limitation, stated rather than hidden: matching looks for the
    reference magnitude anywhere among the answer's numbers. An answer
    containing many figures can therefore match by coincidence. That is why
    section 15 requires a human-reviewed sample — this scorer is a cheap filter,
    not the final word on correctness.
    """

    relative_tolerance: float = 0.01

    @property
    def name(self) -> str:
        return "NumericalAnswerScorer"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        reference = episode.metadata.get("reference_answer")
        if not reference or episode.metadata.get("abstained"):
            return []

        expected = extract_numbers(reference)
        if not expected:
            # A non-numeric reference is out of this scorer's scope. Semantic
            # comparison belongs to an LLM judge, not here.
            return []

        actual = extract_numbers(episode.final_answer)
        if not actual:
            return [ScoreDimension(name="numeric_correctness", value=0.0, source=self.name)]

        target = expected[0]
        matched = any(self._within_tolerance(target, value) for value in actual)
        return [
            ScoreDimension(
                name="numeric_correctness", value=1.0 if matched else 0.0, source=self.name
            )
        ]

    def _within_tolerance(self, expected: float, actual: float) -> bool:
        if expected == 0:
            return abs(actual) <= self.relative_tolerance
        return abs(actual - expected) / abs(expected) <= self.relative_tolerance

    def detect_issues(self, episode: Episode) -> list[Issue]:
        dimensions = self.score(episode)
        if dimensions and dimensions[0].value == 0.0:
            return [
                Issue(
                    severity=Severity.ERROR,
                    category="incorrect_numeric_answer",
                    description=(
                        f"Expected ~{episode.metadata.get('reference_answer')!r}, "
                        f"answered {episode.final_answer!r}."
                    ),
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------
@dataclass
class AbstentionScorer:
    """Whether declining to answer was the right call.

    Judged against whether the evidence was actually retrievable, which gives
    four cases:

    ==================  ==================  ==========================
    abstained           gold retrieved      verdict
    ==================  ==================  ==========================
    yes                 no                  appropriate — the right call
    yes                 yes                 over-cautious
    no                  no                  unsupported answer
    no                  yes                 normal
    ==================  ==================  ==========================

    Only the first two produce a score, because abstention precision is a
    property of abstentions. The third produces an issue: answering when the
    evidence was never retrieved is the failure mode abstention exists to
    prevent.
    """

    k: int = 10

    @property
    def name(self) -> str:
        return "AbstentionScorer"

    def _gold_retrieved(self, episode: Episode) -> bool | None:
        gold = set(episode.metadata.get("gold_chunk_ids") or [])
        if not gold:
            return None
        retrieved = set(list(episode.metadata.get("retrieved_chunk_ids") or [])[: self.k])
        return bool(gold & retrieved)

    def score(self, episode: Episode) -> list[ScoreDimension]:
        if not episode.metadata.get("abstained"):
            return []
        gold_retrieved = self._gold_retrieved(episode)
        if gold_retrieved is None:
            return []
        # Abstaining when the evidence was not there is correct; abstaining
        # when it was there is over-caution.
        return [
            ScoreDimension(
                name="abstention_precision",
                value=0.0 if gold_retrieved else 1.0,
                source=self.name,
            )
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        gold_retrieved = self._gold_retrieved(episode)
        if gold_retrieved is None:
            return []
        abstained = bool(episode.metadata.get("abstained"))

        if abstained and gold_retrieved:
            return [
                Issue(
                    severity=Severity.WARNING,
                    category="over_cautious_abstention",
                    description=(
                        "Declined to answer although gold evidence was retrieved. "
                        "Coverage was lost without a reliability gain."
                    ),
                )
            ]
        if not abstained and not gold_retrieved and episode.final_answer:
            return [
                Issue(
                    severity=Severity.CRITICAL,
                    category="unsupported_answer",
                    description=(
                        "Answered although no gold evidence was retrieved. This is "
                        "the failure mode the abstention action exists to prevent."
                    ),
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------
@dataclass
class CostScorer:
    """Cost, normalized against a declared reference so it enters the reward.

    Emits nothing when pricing was unknown. A cost score of 1.0 (meaning free)
    for an unpriced call would make expensive workflows look cheap in exactly
    the reward term meant to penalise them.
    """

    reference_usd: float = 0.05

    @property
    def name(self) -> str:
        return "CostScorer"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        if episode.metadata.get("pricing_known") is not True:
            return []
        cost = float(episode.metadata.get("estimated_cost_usd") or 0.0)
        normalized = min(cost / self.reference_usd, 1.0) if self.reference_usd else 0.0
        return [
            ScoreDimension(name="cost_score", value=1.0 - normalized, source=self.name),
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        if episode.metadata.get("pricing_known") is not True:
            return [
                Issue(
                    severity=Severity.INFO,
                    category="cost_unmeasured",
                    description=(
                        "No pricing registered for this model, so cost is unknown "
                        "rather than zero and is excluded from the reward."
                    ),
                )
            ]
        return []


@dataclass
class LatencyScorer:
    """Latency, normalized against a declared reference."""

    reference_ms: float = 30_000.0

    @property
    def name(self) -> str:
        return "LatencyScorer"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        latency = episode.metadata.get("latency_ms")
        if latency is None:
            return []
        normalized = min(float(latency) / self.reference_ms, 1.0) if self.reference_ms else 0.0
        return [ScoreDimension(name="latency_score", value=1.0 - normalized, source=self.name)]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        latency = episode.metadata.get("latency_ms")
        if latency is not None and float(latency) > self.reference_ms:
            return [
                Issue(
                    severity=Severity.WARNING,
                    category="slow_response",
                    description=f"{float(latency):.0f} ms exceeds the {self.reference_ms:.0f} ms reference.",
                )
            ]
        return []


@dataclass
class WorkflowViolationScorer:
    """Schema and execution compliance.

    Failures are scored rather than discarded, because a workflow that errors on
    hard questions and is then dropped from the matrix would look better than
    one that answers them badly.
    """

    @property
    def name(self) -> str:
        return "WorkflowViolationScorer"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        status = episode.metadata.get("error_status", "ok")
        return [
            ScoreDimension(
                name="policy_compliance",
                value=1.0 if status == "ok" else 0.0,
                source=self.name,
            )
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        status = episode.metadata.get("error_status", "ok")
        if status == "ok":
            return []
        severity = Severity.WARNING if status in {"rate_limited", "timeout"} else Severity.ERROR
        return [
            Issue(
                severity=severity,
                category=f"workflow_{status}",
                description=str(episode.metadata.get("error_detail") or status),
            )
        ]


def default_scorers(
    *, k: int = 10, cost_reference_usd: float = 0.05, latency_reference_ms: float = 30_000.0
) -> list:
    """The deterministic scorer set for the MVP.

    No LLM judge: every scorer here is free, offline and reproducible, so the
    whole evaluation path runs in CI. Judge-based scorers are added separately
    and validated against these (spec section 15.5).
    """
    return [
        RetrievalRecallScorer(k=k),
        CitationSupportScorer(),
        NumericalAnswerScorer(),
        AbstentionScorer(k=k),
        CostScorer(reference_usd=cost_reference_usd),
        LatencyScorer(reference_ms=latency_reference_ms),
        WorkflowViolationScorer(),
    ]
