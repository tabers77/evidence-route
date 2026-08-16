"""Running EvidenceRoute outcomes through Evallab's evaluation pipeline.

This is the seam described in spec section 13: EvidenceRoute supplies the domain
scorers and the episode translation, Evallab supplies the orchestration,
``ScoreVector`` construction and reporting.

**Evallab's overall score is not the reward.** ``EvalPipeline`` always applies
its own ``RuleBasedScorer``, producing an ``overall_score`` and a letter grade.
That is Evallab's generic notion of trace quality and it is useful for
inspection, but it is not what the router optimises. The reward comes from the
declared profiles in :mod:`evidence_route.evaluation.rewards`, which weigh
correctness against cost, latency and hallucination in ways a generic grader
knows nothing about. :func:`score_vector_to_dimensions` extracts the dimensions
the reward needs; the grade is carried alongside for reporting and never fed
into a policy value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_eval.core.models import Episode
from agent_eval.core.score import ScoreVector
from agent_eval.pipeline.runner import EvalPipeline, EvalResult

from evidence_route.evaluation.rewards import RewardBreakdown, RewardProfile
from evidence_route.evaluation.scorers import default_scorers
from evidence_route.storage.records import EvaluationRecord
from evidence_route.workflows.actions import Action

__all__ = [
    "ScoredOutcome",
    "build_pipeline",
    "evaluation_records_from",
    "reward_from_score_vector",
    "score_episodes",
    "score_vector_to_dimensions",
]


class _NoOpAdapter:
    """Satisfies ``EvalPipeline``'s adapter slot.

    EvidenceRoute builds episodes directly from its own outcome records rather
    than loading traces from a file, so the pipeline's loading path is never
    used. The adapter exists only because the constructor requires one.
    """

    @property
    def framework_name(self) -> str:
        return "evidence_route"

    def load_episode(self, source: str, **kwargs: Any) -> Episode:
        del source, kwargs  # signature required by the protocol; never called
        raise NotImplementedError(
            "EvidenceRoute builds episodes from outcome records; use "
            "outcome_to_episode() rather than loading from a source path."
        )

    def load_episodes(self, source: str, **kwargs: Any) -> list[Episode]:
        del source, kwargs  # signature required by the protocol; never called
        raise NotImplementedError(
            "EvidenceRoute builds episodes from outcome records; use "
            "outcomes_to_episodes() rather than loading from a source path."
        )


@dataclass
class ScoredOutcome:
    """One episode after scoring, with its reward under a chosen profile."""

    episode: Episode
    score_vector: ScoreVector
    #: Evallab's generic trace grade. Reported, never used as a policy value.
    grade: str
    summary: str
    reward: RewardBreakdown | None = None

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def dimensions(self) -> dict[str, float]:
        return score_vector_to_dimensions(self.score_vector)


def build_pipeline(
    *,
    k: int = 10,
    cost_reference_usd: float = 0.05,
    latency_reference_ms: float = 30_000.0,
    scorers: list | None = None,
) -> EvalPipeline:
    """Construct Evallab's pipeline with EvidenceRoute's scorers attached."""
    return EvalPipeline(
        adapter=_NoOpAdapter(),
        scorers=scorers
        if scorers is not None
        else default_scorers(
            k=k,
            cost_reference_usd=cost_reference_usd,
            latency_reference_ms=latency_reference_ms,
        ),
    )


def score_vector_to_dimensions(vector: ScoreVector) -> dict[str, float]:
    """Flatten a score vector into ``{dimension_name: normalized_value}``.

    Later dimensions of the same name overwrite earlier ones; in practice each
    scorer owns distinct names, so this is a flatten rather than a merge.
    """
    return {d.name: d.normalized for d in vector.dimensions}


def reward_from_score_vector(vector: ScoreVector, profile: RewardProfile) -> RewardBreakdown:
    """Compose the scalar reward from a scored episode.

    Quality is read from ``answer_correctness`` when an LLM judge supplied it,
    falling back to the deterministic ``numeric_correctness``. Cost and latency
    scores are *goodness* scores where higher is better, so they are inverted
    back into penalties here — the reward expects normalized cost, not a cost
    score, and conflating the two would flip the sign of the whole cost term.
    """
    dimensions = score_vector_to_dimensions(vector)

    quality: dict[str, float] = {}
    correctness = dimensions.get("answer_correctness", dimensions.get("numeric_correctness"))
    if correctness is not None:
        quality["answer_correctness"] = correctness
    support = dimensions.get("evidence_support", dimensions.get("retrieval_recall"))
    if support is not None:
        quality["evidence_support"] = support

    cost_score = dimensions.get("cost_score")
    latency_score = dimensions.get("latency_score")

    hallucination = 0.0
    if dimensions.get("citation_precision") is not None:
        hallucination = 1.0 - dimensions["citation_precision"]

    violation = 1.0 - dimensions.get("policy_compliance", 1.0)

    return profile.compute(
        quality_dimensions=quality,
        normalized_cost=(1.0 - cost_score) if cost_score is not None else 0.0,
        normalized_latency=(1.0 - latency_score) if latency_score is not None else 0.0,
        hallucination_rate=hallucination,
        policy_violation=violation,
    )


def score_episodes(
    episodes: list[Episode],
    *,
    pipeline: EvalPipeline | None = None,
    profile: RewardProfile | None = None,
) -> list[ScoredOutcome]:
    """Score episodes and, when a profile is given, compute their rewards."""
    pipeline = pipeline or build_pipeline()
    scored: list[ScoredOutcome] = []
    for episode in episodes:
        result: EvalResult = pipeline.evaluate(episode)
        scored.append(
            ScoredOutcome(
                episode=result.episode,
                score_vector=result.score_vector,
                grade=result.grade,
                summary=result.summary,
                reward=(
                    reward_from_score_vector(result.score_vector, profile) if profile else None
                ),
            )
        )
    return scored


def evaluation_records_from(
    scored: ScoredOutcome, *, scorer_version: str = "0.1.0"
) -> list[EvaluationRecord]:
    """Turn a scored episode into the persisted evaluation rows (spec 20.3).

    One row per dimension, carrying both the raw measurement and its normalized
    value, so re-normalizing later never requires re-scoring.
    """
    metadata = scored.episode.metadata
    records: list[EvaluationRecord] = []

    for dimension in scored.score_vector.dimensions:
        records.append(
            EvaluationRecord(
                experiment_id=str(metadata.get("experiment_id", "")),
                question_id=str(metadata.get("question_id", "")),
                action_id=Action(metadata["workflow_action"]),
                scorer_name=dimension.source or "unknown",
                scorer_version=scorer_version,
                raw_metric=dimension.value,
                normalized_score=dimension.normalized,
                evaluation_metadata={
                    "dimension": dimension.name,
                    "max_value": dimension.max_value,
                    "grade": scored.grade,
                },
            )
        )
    return records
