"""Reward profiles and scalar reward composition (spec section 12).

A scalar reward is required for routing and bandit learning, but it is a lossy
projection of a multi-dimensional evaluation. Two workflows with identical
rewards can differ completely: one accurate but expensive, another slightly
worse but far faster, a third answering fewer questions with fewer
hallucinations. This module therefore always carries the underlying dimensions
alongside the scalar (:class:`RewardBreakdown`), so a reported result can be
decomposed rather than taken on faith.

The general form for question *i* and action *a*:

    R(i,a) = Q - λc·C - λl·L - λh·H - λv·V

where Q is answer quality and evidence support, C normalized cost, L normalized
latency, H the hallucination / unsupported-claim penalty and V a hard
reliability or policy violation penalty.

Because the preferred policy can depend on the weights, several profiles are
declared up front and conclusions are reported under all of them. A finding that
survives only one weighting is a finding about the weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "REWARD_PROFILES",
    "RewardBreakdown",
    "RewardProfile",
    "get_profile",
]


@dataclass(frozen=True)
class RewardBreakdown:
    """A scalar reward together with the terms that produced it.

    Kept so that any reward appearing in a table or plot can be traced back to
    its components without recomputation.
    """

    total: float
    quality: float
    cost_penalty: float
    latency_penalty: float
    hallucination_penalty: float
    violation_penalty: float
    profile_name: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "profile": self.profile_name,
            "total": self.total,
            "quality": self.quality,
            "cost_penalty": self.cost_penalty,
            "latency_penalty": self.latency_penalty,
            "hallucination_penalty": self.hallucination_penalty,
            "violation_penalty": self.violation_penalty,
        }


@dataclass(frozen=True)
class RewardProfile:
    """A declared set of reward weights with a stated intent.

    ``version`` is part of the identity: changing a weight changes every reward
    downstream, so profiles are versioned rather than edited in place.
    """

    name: str
    description: str
    lambda_cost: float
    lambda_latency: float
    lambda_hallucination: float
    lambda_violation: float
    version: str = "v1"
    #: Weights applied to the quality sub-dimensions before penalties.
    quality_weights: dict[str, float] = field(
        default_factory=lambda: {"answer_correctness": 0.6, "evidence_support": 0.4}
    )

    def compose_quality(self, dimensions: dict[str, float]) -> float:
        """Combine quality sub-dimensions into a single Q term in [0, 1].

        Missing dimensions are treated as absent rather than as zero: scoring a
        workflow as maximally wrong because a scorer failed to run would
        confuse a measurement gap with a quality failure. Weights are
        renormalized over whatever is present.
        """
        present = {k: w for k, w in self.quality_weights.items() if k in dimensions}
        if not present:
            return 0.0
        total_weight = sum(present.values())
        if total_weight == 0:
            return 0.0
        return sum(dimensions[k] * w for k, w in present.items()) / total_weight

    def compute(
        self,
        *,
        quality_dimensions: dict[str, float],
        normalized_cost: float,
        normalized_latency: float,
        hallucination_rate: float,
        policy_violation: float = 0.0,
    ) -> RewardBreakdown:
        """Compute the scalar reward and its breakdown.

        All inputs are expected in [0, 1]. Cost and latency must already be
        normalized against the declared reference scale (see the experiment
        config) rather than passed in raw units — otherwise a change in the
        cost of one provider would silently reweight the reward.
        """
        quality = self.compose_quality(quality_dimensions)
        cost_penalty = self.lambda_cost * normalized_cost
        latency_penalty = self.lambda_latency * normalized_latency
        hallucination_penalty = self.lambda_hallucination * hallucination_rate
        violation_penalty = self.lambda_violation * policy_violation

        total = quality - cost_penalty - latency_penalty - hallucination_penalty - violation_penalty
        return RewardBreakdown(
            total=total,
            quality=quality,
            cost_penalty=cost_penalty,
            latency_penalty=latency_penalty,
            hallucination_penalty=hallucination_penalty,
            violation_penalty=violation_penalty,
            profile_name=f"{self.name}:{self.version}",
        )


#: The declared profiles (spec section 12.3). Results are reported under every
#: profile; the primary comparison names one of them in the frozen protocol.
REWARD_PROFILES: dict[str, RewardProfile] = {
    "quality_first": RewardProfile(
        name="quality_first",
        description=(
            "Correctness and evidence support dominate. Cost and latency are "
            "acknowledged but rarely decisive. Represents a setting where a "
            "wrong answer is far more expensive than a slow one."
        ),
        lambda_cost=0.05,
        lambda_latency=0.05,
        lambda_hallucination=0.60,
        lambda_violation=1.00,
    ),
    "balanced_enterprise": RewardProfile(
        name="balanced_enterprise",
        description=(
            "Quality, cost and latency all matter materially. The default "
            "operating point for enterprise decision support, and the intended "
            "primary profile unless the protocol says otherwise."
        ),
        lambda_cost=0.25,
        lambda_latency=0.20,
        lambda_hallucination=0.40,
        lambda_violation=1.00,
    ),
    "cost_sensitive": RewardProfile(
        name="cost_sensitive",
        description=(
            "Minimize spend while holding an acceptable quality floor. Expect "
            "this profile to favour A1/A2 and to make the agentic action hard "
            "to justify."
        ),
        lambda_cost=0.60,
        lambda_latency=0.35,
        lambda_hallucination=0.30,
        lambda_violation=1.00,
    ),
    "regulated": RewardProfile(
        name="regulated",
        description=(
            "High-risk setting where abstention is preferred to an uncertain "
            "answer. Hallucination and policy violations carry the heaviest "
            "penalties in the suite (spec section 12.3, optional fourth profile)."
        ),
        lambda_cost=0.10,
        lambda_latency=0.10,
        lambda_hallucination=1.00,
        lambda_violation=2.00,
    ),
}


def get_profile(name: str) -> RewardProfile:
    """Look up a declared reward profile by name.

    Raises rather than falling back to a default: silently substituting a
    profile would attribute results to the wrong weighting.
    """
    try:
        return REWARD_PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(REWARD_PROFILES))
        raise KeyError(f"Unknown reward profile {name!r}. Declared profiles: {known}") from None
