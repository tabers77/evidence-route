"""Candidate question-answering workflows (the router's action space).

Each workflow in this package executes one :class:`~evidence_route.workflows.actions.Action`
and returns a :class:`~evidence_route.storage.records.WorkflowOutcomeRecord`.
All actions share that outcome schema so they remain directly comparable on the
same question — this is what makes the full-information outcome matrix possible.

Planned modules (spec section 6):
    ``direct``          A0 — no retrieval, parametric answer only.
    ``bm25``            A1 — lexical retrieval + generation.
    ``dense``           A2 — embedding retrieval + generation.
    ``hybrid``          A3 — explicit rank fusion + generation.
    ``hybrid_rerank``   A4 — fusion + cross-encoder rerank + generation.
    ``agentic``         A5 — bounded multi-step agent loop.
    ``abstain``         A6 — decline with a machine-readable reason code.
    ``base``            shared protocol every workflow implements.
"""

from evidence_route.workflows.actions import (
    ACTION_SPACE,
    AbstentionReason,
    Action,
    action_from_index,
    action_index,
)

__all__ = [
    "ACTION_SPACE",
    "AbstentionReason",
    "Action",
    "action_from_index",
    "action_index",
]
