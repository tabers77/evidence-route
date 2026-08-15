"""EvidenceRoute.

Statistical evaluation and contextual-bandit routing for reliable enterprise
question answering.

The research question this package exists to answer:

    Can an adaptive routing policy select the least expensive sufficient
    question-answering workflow while preserving or improving evidence-grounded
    answer quality relative to fixed RAG and fixed agentic baselines?

EvidenceRoute owns the domain: datasets, retrieval, generation, agentic
workflows, routing features, bandit policies, offline policy evaluation and the
statistical protocol. Evallab (distribution ``agent-eval``) owns the generic
evaluation layer: canonical episodes, scorer protocols, score vectors, reward
composition and reporting.

See ``docs/EVIDENCEROUTE_PROJECT_SPECIFICATION.md`` for the specification and
current implementation status, and ``experiments/protocols/protocol_v1.md`` for
the research protocol.
"""

from evidence_route.workflows.actions import AbstentionReason, Action

__version__ = "0.1.0"

__all__ = ["AbstentionReason", "Action", "__version__"]
