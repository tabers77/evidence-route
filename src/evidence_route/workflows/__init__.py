"""Candidate question-answering workflows (the router's action space).

Each workflow in this package executes one :class:`~evidence_route.workflows.actions.Action`
and returns a :class:`~evidence_route.storage.records.WorkflowOutcomeRecord`.
All actions share that outcome schema so they remain directly comparable on the
same question — this is what makes the full-information outcome matrix possible.

Modules (spec section 6):
    ``base``            shared execution: timing, accounting, error handling
    ``direct``          A0 — no retrieval, parametric answer only (implemented)
    ``bm25_workflow``   A1 — lexical retrieval + generation (implemented)
    ``dense``           A2 — embedding retrieval + generation (planned)
    ``hybrid``          A3 — explicit rank fusion + generation (planned)
    ``hybrid_rerank``   A4 — fusion + cross-encoder rerank + generation (planned)
    ``agentic``         A5 — bounded multi-step agent loop (planned)
    ``abstain``         A6 — decline with a machine-readable reason code (planned)
"""

from evidence_route.workflows.actions import (
    ACTION_SPACE,
    AbstentionReason,
    Action,
    action_from_index,
    action_index,
)
from evidence_route.workflows.base import GenerativeWorkflow, Workflow, WorkflowContext
from evidence_route.workflows.bm25_workflow import BM25Workflow
from evidence_route.workflows.dense_workflow import DenseWorkflow, HybridWorkflow
from evidence_route.workflows.direct import DirectAnswerWorkflow

__all__ = [
    "ACTION_SPACE",
    "AbstentionReason",
    "Action",
    "BM25Workflow",
    "DenseWorkflow",
    "DirectAnswerWorkflow",
    "GenerativeWorkflow",
    "HybridWorkflow",
    "Workflow",
    "WorkflowContext",
    "action_from_index",
    "action_index",
]
