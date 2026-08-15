"""Tests for reward composition and the declared profiles."""

from __future__ import annotations

import pytest

from evidence_route.evaluation.rewards import REWARD_PROFILES, RewardProfile, get_profile


def _full_quality() -> dict[str, float]:
    return {"answer_correctness": 1.0, "evidence_support": 1.0}


def test_all_declared_profiles_are_present():
    assert set(REWARD_PROFILES) == {
        "quality_first",
        "balanced_enterprise",
        "cost_sensitive",
        "regulated",
    }


def test_get_profile_rejects_unknown_name():
    """Falling back to a default would attribute results to the wrong weighting."""
    with pytest.raises(KeyError, match="Unknown reward profile"):
        get_profile("does_not_exist")


def test_perfect_free_instant_answer_scores_one():
    profile = get_profile("balanced_enterprise")
    reward = profile.compute(
        quality_dimensions=_full_quality(),
        normalized_cost=0.0,
        normalized_latency=0.0,
        hallucination_rate=0.0,
    )
    assert reward.total == pytest.approx(1.0)
    assert reward.quality == pytest.approx(1.0)


def test_breakdown_sums_back_to_total():
    profile = get_profile("cost_sensitive")
    reward = profile.compute(
        quality_dimensions={"answer_correctness": 0.8, "evidence_support": 0.6},
        normalized_cost=0.4,
        normalized_latency=0.3,
        hallucination_rate=0.1,
        policy_violation=0.0,
    )
    recomposed = (
        reward.quality
        - reward.cost_penalty
        - reward.latency_penalty
        - reward.hallucination_penalty
        - reward.violation_penalty
    )
    assert recomposed == pytest.approx(reward.total)


def test_quality_weights_are_applied():
    profile = get_profile("balanced_enterprise")
    # Default weights: correctness 0.6, evidence support 0.4.
    quality = profile.compose_quality({"answer_correctness": 1.0, "evidence_support": 0.0})
    assert quality == pytest.approx(0.6)


def test_missing_dimension_renormalizes_rather_than_scoring_zero():
    """A scorer that did not run is a measurement gap, not a quality failure."""
    profile = get_profile("balanced_enterprise")
    quality = profile.compose_quality({"answer_correctness": 0.9})
    assert quality == pytest.approx(0.9)


def test_no_known_dimensions_yields_zero():
    profile = get_profile("balanced_enterprise")
    assert profile.compose_quality({"unrelated_metric": 1.0}) == 0.0
    assert profile.compose_quality({}) == 0.0


def test_regulated_profile_punishes_hallucination_hardest():
    """The regulated profile must make an unsupported answer worse than abstaining."""
    rates = {name: profile.lambda_hallucination for name, profile in REWARD_PROFILES.items()}
    assert rates["regulated"] == max(rates.values())
    assert rates["regulated"] > rates["cost_sensitive"]


def test_cost_sensitive_profile_punishes_cost_hardest():
    costs = {name: profile.lambda_cost for name, profile in REWARD_PROFILES.items()}
    assert costs["cost_sensitive"] == max(costs.values())


def test_profiles_can_reorder_actions():
    """The preferred action is weighting-dependent — the reason profiles exist.

    An expensive-but-accurate workflow beats a cheap-but-weaker one under
    quality_first, and loses under cost_sensitive. A conclusion that holds under
    only one profile is a conclusion about the profile.
    """
    expensive_accurate = {
        "quality_dimensions": {"answer_correctness": 1.0, "evidence_support": 1.0},
        "normalized_cost": 0.9,
        "normalized_latency": 0.9,
        "hallucination_rate": 0.0,
    }
    cheap_weaker = {
        "quality_dimensions": {"answer_correctness": 0.75, "evidence_support": 0.75},
        "normalized_cost": 0.05,
        "normalized_latency": 0.05,
        "hallucination_rate": 0.0,
    }

    quality_first = get_profile("quality_first")
    assert (
        quality_first.compute(**expensive_accurate).total
        > quality_first.compute(**cheap_weaker).total
    )

    cost_sensitive = get_profile("cost_sensitive")
    assert (
        cost_sensitive.compute(**expensive_accurate).total
        < cost_sensitive.compute(**cheap_weaker).total
    )


def test_violation_penalty_can_drive_reward_negative():
    """A hard policy violation must not be recoverable by good quality alone."""
    profile = get_profile("regulated")
    reward = profile.compute(
        quality_dimensions=_full_quality(),
        normalized_cost=0.0,
        normalized_latency=0.0,
        hallucination_rate=0.0,
        policy_violation=1.0,
    )
    assert reward.total < 0


def test_profile_identity_includes_version():
    profile = get_profile("balanced_enterprise")
    reward = profile.compute(
        quality_dimensions=_full_quality(),
        normalized_cost=0.1,
        normalized_latency=0.1,
        hallucination_rate=0.0,
    )
    # Rewards are only comparable within a profile version, so the version
    # travels with every computed value.
    assert reward.profile_name == "balanced_enterprise:v1"


def test_profiles_are_immutable():
    profile = get_profile("balanced_enterprise")
    with pytest.raises(Exception):  # noqa: B017 - dataclass raises FrozenInstanceError
        profile.lambda_cost = 0.99  # type: ignore[misc]


def test_breakdown_as_dict_is_serialisable():
    reward = get_profile("quality_first").compute(
        quality_dimensions=_full_quality(),
        normalized_cost=0.2,
        normalized_latency=0.1,
        hallucination_rate=0.05,
    )
    payload = reward.as_dict()
    assert payload["profile"] == "quality_first:v1"
    assert set(payload) == {
        "profile",
        "total",
        "quality",
        "cost_penalty",
        "latency_penalty",
        "hallucination_penalty",
        "violation_penalty",
    }


def test_custom_profile_can_declare_its_own_quality_weights():
    profile = RewardProfile(
        name="retrieval_focused",
        description="test profile",
        lambda_cost=0.1,
        lambda_latency=0.1,
        lambda_hallucination=0.5,
        lambda_violation=1.0,
        quality_weights={"retrieval_recall": 1.0},
    )
    assert profile.compose_quality({"retrieval_recall": 0.5}) == pytest.approx(0.5)
