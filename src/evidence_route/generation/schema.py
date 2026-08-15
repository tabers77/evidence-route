"""The structured answer contract.

Every workflow returns this shape, whether it retrieved anything or not.

Free text would make three of the project's headline metrics uncomputable:
citation precision needs to know which chunks the model claims support which
statement, abstention precision needs a machine-readable decline, and
calibration needs a confidence value that exists before the answer is graded.
None of those can be recovered reliably by parsing prose after the fact.

The schema also enforces the invariants that keep the evaluation honest — most
importantly that a citation must name a chunk the workflow actually retrieved.
A model citing a chunk it was never shown is a hallucinated citation, and
catching it here turns a silent scoring error into a validation failure.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_route.workflows.actions import AbstentionReason

__all__ = [
    "AnswerCitation",
    "StructuredAnswer",
    "extract_json_object",
    "parse_structured_answer",
]


class AnswerCitation(BaseModel):
    """A claim-to-evidence link asserted by the generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    #: The span the model says supports its claim. Optional because not every
    #: prompt elicits it, but citation precision is much weaker without it —
    #: a chunk-level citation only proves the model pointed at the right page.
    quoted_text: str | None = None
    claim: str | None = None


class StructuredAnswer(BaseModel):
    """What a workflow returns.

    Either an answer with citations, or an abstention with a reason code. Never
    both, never neither.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str | None = None
    citations: list[AnswerCitation] = Field(default_factory=list)

    #: The model's own confidence, before any scoring. Whether it means anything
    #: is an empirical question the calibration analysis answers (spec 14.6) —
    #: it is collected precisely so that can be tested rather than assumed.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    abstained: bool = False
    abstention_reason: AbstentionReason | None = None

    #: Free-text rationale. Recorded for failure analysis, never scored — an
    #: LLM judge rewarding a fluent rationale is exactly the metric-gaming
    #: failure mode section 16 warns about.
    reasoning: str | None = None

    @model_validator(mode="after")
    def _check_answer_or_abstention(self) -> StructuredAnswer:
        if self.abstained:
            if self.abstention_reason is None:
                raise ValueError(
                    "An abstention requires an abstention_reason; without a "
                    "machine-readable code it cannot be audited."
                )
            if self.answer:
                raise ValueError(
                    "An abstention cannot also carry an answer. Producing both "
                    "would let a workflow be scored as answering and as "
                    "abstaining on the same question."
                )
        else:
            if self.abstention_reason is not None:
                raise ValueError("abstention_reason is only valid when abstained is true.")
            if self.answer is None:
                raise ValueError(
                    "A non-abstaining answer must contain text. If the workflow "
                    "could not answer, it should abstain with a reason code."
                )
        return self

    @property
    def cited_chunk_ids(self) -> set[str]:
        return {c.chunk_id for c in self.citations}

    def validate_citations(self, retrieved_chunk_ids: set[str]) -> list[str]:
        """Return citations naming chunks that were never retrieved.

        A non-empty result is a hallucinated citation: the model referenced
        evidence it was not shown. Reported rather than raised, so the outcome
        is still recorded and the failure is counted in the taxonomy instead of
        aborting the run.
        """
        return sorted(self.cited_chunk_ids - retrieved_chunk_ids)

    @classmethod
    def abstain(
        cls, reason: AbstentionReason, *, confidence: float | None = None
    ) -> StructuredAnswer:
        """Construct an abstention."""
        return cls(abstained=True, abstention_reason=reason, confidence=confidence)


#: Models frequently wrap JSON in prose or a markdown fence despite instructions.
_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Tries the whole string, then a fenced block, then the outermost brace pair.
    This tolerance is deliberate: a response that is substantively correct but
    wrapped in "Here is the answer:" should be recorded as a correct answer, not
    as a parse failure, or the parse-failure rate becomes a measure of prompt
    formatting rather than of the workflow.

    Raises ``ValueError`` when no object can be recovered; the caller records
    that as a ``parse_error`` outcome with its cost, rather than discarding it.
    """
    candidates: list[str] = [raw.strip()]

    fenced = _FENCE_RE.search(raw)
    if fenced:
        candidates.append(fenced.group("body").strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No JSON object found in model response (first 200 chars): {raw[:200]!r}")


def parse_structured_answer(raw: str) -> StructuredAnswer:
    """Parse a model response into a :class:`StructuredAnswer`.

    Normalizes two shapes models commonly emit: citations given as bare chunk-id
    strings rather than objects, and an abstention expressed only by a reason
    code without the ``abstained`` flag.
    """
    payload = extract_json_object(raw)

    citations = payload.get("citations") or []
    normalized: list[dict[str, Any]] = []
    for citation in citations:
        if isinstance(citation, str):
            normalized.append({"chunk_id": citation})
        elif isinstance(citation, dict) and citation.get("chunk_id"):
            normalized.append(
                {
                    "chunk_id": str(citation["chunk_id"]),
                    "quoted_text": citation.get("quoted_text"),
                    "claim": citation.get("claim"),
                }
            )
    payload["citations"] = normalized

    # A reason code implies an abstention even if the flag was omitted.
    if payload.get("abstention_reason") and not payload.get("abstained"):
        payload["abstained"] = True

    # Models sometimes emit the empty string instead of null for a non-answer.
    if payload.get("answer") == "":
        payload["answer"] = None

    return StructuredAnswer.model_validate(payload)
