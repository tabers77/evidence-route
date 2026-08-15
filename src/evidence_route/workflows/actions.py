"""The routing action space.

The policy's action space is deliberately small and interpretable in the MVP
(spec section 6). Every candidate workflow must produce outcomes that share a
common schema so that actions are directly comparable on the same question.

Adding an action is a protocol change, not an implementation detail: it widens
the outcome matrix, changes the propensity denominators in the logged-feedback
simulation, and invalidates previously trained routers. Record any change in
``experiments/protocols/``.
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """A candidate question-answering workflow the router may select.

    Values are stable identifiers written into the outcome matrix and bandit
    logs. They must not be renamed once experiment data exists.
    """

    #: Answer directly from parametric memory, with no retrieval. Establishes
    #: the no-retrieval baseline and quantifies plausible-but-unsupported
    #: answers, including pretraining contamination.
    DIRECT = "A0_direct"

    #: BM25 lexical retrieval plus answer generation. Strong classical baseline
    #: for queries with explicit terminology, names, figures and exact phrases.
    BM25 = "A1_bm25"

    #: Dense embedding retrieval plus answer generation. Handles paraphrases and
    #: terminology mismatch.
    DENSE = "A2_dense"

    #: Hybrid retrieval (lexical + dense, fused explicitly) plus generation.
    #: The fusion method must be documented, not implied.
    HYBRID = "A3_hybrid"

    #: Hybrid retrieval, cross-encoder reranking, then generation. Buys evidence
    #: quality with additional latency and cost.
    HYBRID_RERANK = "A4_hybrid_rerank"

    #: Bounded multi-step agentic retrieval with an allowlisted tool set.
    #: Included as a candidate, not assumed superior.
    AGENTIC = "A5_agentic"

    #: Decline to answer and route to human review. A valid reliability
    #: decision, evaluated through risk-coverage analysis.
    ABSTAIN = "A6_abstain"

    @property
    def uses_retrieval(self) -> bool:
        """Whether this action retrieves evidence before answering."""
        return self in _RETRIEVING_ACTIONS

    @property
    def produces_answer(self) -> bool:
        """Whether this action is expected to emit an answer at all."""
        return self is not Action.ABSTAIN

    @classmethod
    def answering_actions(cls) -> tuple[Action, ...]:
        """Every action except abstention, in declaration order."""
        return tuple(a for a in cls if a.produces_answer)


_RETRIEVING_ACTIONS = frozenset(
    {
        Action.BM25,
        Action.DENSE,
        Action.HYBRID,
        Action.HYBRID_RERANK,
        Action.AGENTIC,
    }
)


class AbstentionReason(str, Enum):
    """Machine-readable reason codes attached to an abstention (spec 6, A6).

    An abstention without a reason code is not auditable, so the outcome schema
    requires one whenever the selected action is :attr:`Action.ABSTAIN`.
    """

    #: Retrieval returned nothing that supports an answer.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    #: Lexical and dense retrieval point at materially different evidence.
    RETRIEVAL_DISAGREEMENT = "retrieval_disagreement"

    #: The question admits multiple readings that imply different answers.
    AMBIGUOUS_QUESTION = "ambiguous_question"

    #: Retrieved documents contradict one another on the queried fact.
    CONFLICTING_DOCUMENTS = "conflicting_documents"

    #: An arithmetic result could not be verified against the evidence.
    CALCULATION_UNVERIFIED = "calculation_unverified"

    #: The router's own confidence fell below the validated threshold.
    POLICY_LOW_CONFIDENCE = "policy_low_confidence"


#: Ordered action space used as the canonical column order of the
#: full-information outcome matrix and as the index basis for bandit policies.
ACTION_SPACE: tuple[Action, ...] = tuple(Action)


def action_index(action: Action) -> int:
    """Position of ``action`` in :data:`ACTION_SPACE`.

    Bandit policies operate on integer arms; this is the single place that maps
    between the arm index and the semantic action.
    """
    return ACTION_SPACE.index(action)


def action_from_index(index: int) -> Action:
    """Inverse of :func:`action_index`."""
    return ACTION_SPACE[index]
