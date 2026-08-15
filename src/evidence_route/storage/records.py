"""The experiment data model (spec section 20).

Four record types carry every experiment through the pipeline:

    QuestionRecord        one benchmark question and its reference evidence
    WorkflowOutcomeRecord what one action did on one question — *raw* outputs
    EvaluationRecord      what one scorer said about one outcome — *derived*
    BanditLogRecord       one simulated round of logged partial feedback

Two invariants shape these schemas:

1. **Raw and derived stay separate.** ``WorkflowOutcomeRecord`` holds what the
   system produced; ``EvaluationRecord`` holds what a scorer concluded. Scorers,
   normalization rules and reward weights all change over the project's life, and
   when they do, the derived layer is recomputed while the expensive raw layer is
   reused. Merging them would force re-running paid LLM calls to change a metric.

2. **Enough provenance to recompute without re-running.** Every record carries
   the experiment, code commit, seed and configuration that produced it
   (spec section 25). A result whose provenance is missing cannot be defended.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_route.workflows.actions import AbstentionReason, Action

QuestionType = Literal[
    "extraction",
    "comparison",
    "arithmetic",
    "synthesis",
    "unanswerable",
]

SplitName = Literal["dev", "validation", "test", "challenge"]

ErrorStatus = Literal[
    "ok",
    "provider_error",
    "timeout",
    "rate_limited",
    "parse_error",
    "budget_exceeded",
    "step_limit_exceeded",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _Record(BaseModel):
    """Shared configuration for every persisted record."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# 20.1 Question record
# ---------------------------------------------------------------------------
class QuestionRecord(_Record):
    """One benchmark question with its reference answer and evidence."""

    question_id: str
    dataset: str
    split: SplitName
    question_text: str
    question_type: QuestionType | None = None

    #: Documents the question is asked against. Splits are grouped by document
    #: or company rather than sampled per question, because several questions
    #: often share a source and random splitting would leak document-specific
    #: patterns into training (spec section 7.4).
    document_ids: list[str] = Field(default_factory=list)

    reference_answer: str | None = None

    #: Identifiers of the evidence spans/chunks that support the reference
    #: answer. Retrieval recall is measured against these, independently of
    #: whether the final answer happened to be correct (spec section 11.1).
    reference_evidence: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 20.2 Workflow outcome record
# ---------------------------------------------------------------------------
class RetrievedItem(BaseModel):
    """One retrieved chunk, with the scores that ranked it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str
    rank: int = Field(ge=1)
    text: str | None = None
    #: Raw retriever score. Its scale depends on the retriever, so it is stored
    #: as produced and normalized only at analysis time.
    score: float | None = None
    #: Present for hybrid actions: the per-retriever ranks that were fused.
    component_ranks: dict[str, int] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Per-stage wall-clock cost in milliseconds (spec section 11.5).

    Broken down rather than totalled because the routing question is partly
    "which stage is worth its latency" — a single end-to-end number cannot
    answer that.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_ms: float = Field(ge=0)
    retrieval_ms: float = Field(default=0.0, ge=0)
    reranking_ms: float = Field(default=0.0, ge=0)
    generation_ms: float = Field(default=0.0, ge=0)
    #: Cost of computing routing features. Counted against the router, not
    #: against the selected action — a router is not cheap if it secretly runs
    #: expensive probes before choosing (spec section 8.2).
    feature_extraction_ms: float = Field(default=0.0, ge=0)


class TokenUsage(BaseModel):
    """Token counts, including tokens burned by retries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    #: Retries are recorded rather than silently discarded, so reported cost
    #: reflects what was actually spent (spec section 26, item 8).
    retry_input_tokens: int = Field(default=0, ge=0)
    retry_output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.retry_input_tokens
            + self.retry_output_tokens
        )


class Citation(BaseModel):
    """A claim-to-evidence link asserted by the generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str
    quoted_text: str | None = None
    claim: str | None = None


class WorkflowOutcomeRecord(_Record):
    """What one action produced on one question. Raw, not scored.

    This is the expensive layer: producing it costs API calls and wall-clock.
    Nothing here is a judgement about quality — that lives in
    :class:`EvaluationRecord`, which is cheap to recompute from this.
    """

    experiment_id: str
    question_id: str
    action_id: Action

    #: Bumped whenever a workflow's behaviour changes, so outcomes produced by
    #: different implementations are never silently pooled.
    workflow_version: str = "0.1.0"

    #: Deployment names, model versions, temperature, top-k and so on. Recorded
    #: because Azure deployments can be re-pointed at new model versions without
    #: any change in this repository.
    model_configuration: dict[str, Any] = Field(default_factory=dict)

    retrieved_items: list[RetrievedItem] = Field(default_factory=list)
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)

    #: Required when ``action_id`` is ABSTAIN, forbidden otherwise. An
    #: abstention without a reason code is not auditable.
    abstention_reason: AbstentionReason | None = None

    #: Path to the full trace (tool calls, prompts, raw provider responses).
    #: Kept out of the record itself so the outcome table stays queryable.
    raw_trace_path: str | None = None

    latency_breakdown: LatencyBreakdown | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = Field(default=0.0, ge=0)

    error_status: ErrorStatus = "ok"
    error_detail: str | None = None

    random_seed: int | None = None
    code_commit: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _check_abstention_consistency(self) -> WorkflowOutcomeRecord:
        if self.action_id is Action.ABSTAIN and self.abstention_reason is None:
            raise ValueError(
                "action_id=ABSTAIN requires an abstention_reason; an abstention "
                "without a machine-readable reason code cannot be audited."
            )
        if self.action_id is not Action.ABSTAIN and self.abstention_reason is not None:
            raise ValueError(f"abstention_reason is only valid for ABSTAIN, got {self.action_id}.")
        return self

    @property
    def succeeded(self) -> bool:
        """Whether this outcome is usable, as opposed to an explicit error state.

        Failed outcomes are kept, not dropped: silently discarding them would
        bias the outcome matrix toward actions that fail loudly on hard
        questions (spec section 21, week 7 exit criteria).
        """
        return self.error_status == "ok"


# ---------------------------------------------------------------------------
# 20.3 Evaluation record
# ---------------------------------------------------------------------------
class EvaluationRecord(_Record):
    """What one scorer concluded about one workflow outcome.

    One row per (outcome, scorer). Both the raw measurement and the normalized
    score are stored, because normalization rules are versioned and will change
    — and when they do, re-normalizing must not require re-scoring.
    """

    experiment_id: str
    question_id: str
    action_id: Action

    scorer_name: str
    scorer_version: str = "0.1.0"

    #: The measurement in its natural units (e.g. recall@10 = 0.6, latency
    #: = 2140 ms, cost = 0.0031 USD).
    raw_metric: float

    #: The measurement mapped into [0, 1] under a versioned normalization rule.
    normalized_score: float = Field(ge=0.0, le=1.0)
    normalization_version: str = "0.1.0"

    #: Set only for LLM-judge scorers. Judge results are tracked separately
    #: from deterministic ones so judge bias can be quantified against human
    #: labels later (spec sections 15.5 and 27).
    judge_model: str | None = None
    judge_prompt_version: str | None = None

    evaluation_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    @property
    def is_llm_judged(self) -> bool:
        return self.judge_model is not None


# ---------------------------------------------------------------------------
# 20.4 Bandit log record
# ---------------------------------------------------------------------------
class BanditLogRecord(_Record):
    """One round of simulated logged bandit feedback (spec section 10).

    Constructed by replaying a behavior policy over the full-information outcome
    matrix. This is a *controlled logged-feedback simulation*, not production
    user feedback and not a deployed online RL system.
    """

    round_id: int = Field(ge=0)
    question_id: str

    #: The context the behavior policy saw. Must contain only features available
    #: *before* the answer is known, or the routing claim is invalid
    #: (spec sections 8 and 27).
    context_vector: list[float]
    context_feature_names: list[str] = Field(default_factory=list)

    behavior_policy: str
    selected_action: Action

    #: P(selected_action | context) under the behavior policy. Strictly positive:
    #: IPS divides by it, and a zero propensity means the action could never have
    #: been logged, which makes the estimator undefined rather than merely noisy.
    action_propensity: float = Field(gt=0.0, le=1.0)

    observed_reward: float
    reward_profile: str
    simulation_seed: int

    @model_validator(mode="after")
    def _check_context_names(self) -> BanditLogRecord:
        names = self.context_feature_names
        if names and len(names) != len(self.context_vector):
            raise ValueError(
                f"context_feature_names has {len(names)} entries but "
                f"context_vector has {len(self.context_vector)}."
            )
        return self


__all__ = [
    "BanditLogRecord",
    "Citation",
    "ErrorStatus",
    "EvaluationRecord",
    "LatencyBreakdown",
    "QuestionRecord",
    "QuestionType",
    "RetrievedItem",
    "SplitName",
    "TokenUsage",
    "WorkflowOutcomeRecord",
]
