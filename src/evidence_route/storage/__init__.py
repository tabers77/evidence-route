"""Persistence for questions, outcomes, evaluations and bandit logs.

Raw model outputs and derived metrics are stored in separate tables so that
scorers, normalization rules and reward weights can be revised without
re-running paid LLM calls (spec section 18, repository-structure principles).

Outcome and evaluation tables are written as Parquet: they are read far more
often than written, are analysed column-wise, and benefit from a typed schema
that survives a round-trip. Full traces stay as JSON on disk, referenced from
the outcome record by path.

Planned modules:
    ``records``   the typed record schemas (implemented)
    ``parquet``   read/write outcome and evaluation tables
    ``traces``    raw trace serialisation and retrieval
    ``cache``     content-addressed response cache for cost control
"""

from evidence_route.storage.records import (
    BanditLogRecord,
    Citation,
    EvaluationRecord,
    LatencyBreakdown,
    QuestionRecord,
    RetrievedItem,
    TokenUsage,
    WorkflowOutcomeRecord,
)

__all__ = [
    "BanditLogRecord",
    "Citation",
    "EvaluationRecord",
    "LatencyBreakdown",
    "QuestionRecord",
    "RetrievedItem",
    "TokenUsage",
    "WorkflowOutcomeRecord",
]
