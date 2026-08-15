"""Domain-specific evaluation: scorers, reward composition and provenance.

Generic evaluation machinery (canonical episodes, scorer protocols, score
vectors, reporting) lives in Evallab. This package holds only what is specific
to EvidenceRoute — the scorers that understand financial questions, evidence
citations and abstention, plus the reward profiles that turn a score vector into
the scalar the router optimizes.

Planned scorer modules (spec section 13.2), each implementing Evallab's
``Scorer`` protocol:
    ``RetrievalRecallScorer``     evidence recall, independent of answer quality
    ``CitationSupportScorer``     do the citations actually support the claims
    ``NumericalAnswerScorer``     numeric equality under a declared tolerance
    ``AbstentionScorer``          was declining the right call
    ``CostScorer`` / ``LatencyScorer``   operational dimensions
    ``WorkflowViolationScorer``   step limits, tool allowlist, schema compliance

A scorer that turns out to be domain-independent should be proposed for Evallab
rather than duplicated there.
"""

from evidence_route.evaluation.rewards import (
    REWARD_PROFILES,
    RewardBreakdown,
    RewardProfile,
    get_profile,
)

__all__ = [
    "REWARD_PROFILES",
    "RewardBreakdown",
    "RewardProfile",
    "get_profile",
]
