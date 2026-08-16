"""Translating workflow outcomes into Evallab episodes.

Evallab's ``Scorer`` protocol takes an ``Episode`` and nothing else, so anything
a scorer needs has to be *on* the episode. That shapes this module: the adapter
does not merely reformat the trace, it attaches the ground truth — reference
answer, reference evidence, resolved gold chunk ids — that domain scorers need
to compute retrieval recall or judge whether an abstention was appropriate.

Without that enrichment the scorers would need a side channel back to the
dataset, and the clean protocol boundary that makes Evallab reusable would
disappear.

Evallab is imported at module level here rather than lazily. This module is
*about* Evallab, so it is not importable without it — and the rest of
EvidenceRoute stays importable because nothing else imports this eagerly.
"""

from __future__ import annotations

from typing import Any

from agent_eval.core.models import Episode, Step, StepKind

from evidence_route.storage.records import QuestionRecord, WorkflowOutcomeRecord

__all__ = ["episode_id_for", "outcome_to_episode", "outcomes_to_episodes"]

SOURCE_FRAMEWORK = "evidence_route"


def episode_id_for(outcome: WorkflowOutcomeRecord) -> str:
    """Stable identifier for one question-action pair within an experiment.

    The three parts together are what make a cell of the outcome matrix unique,
    so the id can be joined back to the matrix without a lookup table.
    """
    return f"{outcome.experiment_id}::{outcome.question_id}::{outcome.action_id.value}"


def outcome_to_episode(
    outcome: WorkflowOutcomeRecord,
    question: QuestionRecord,
    *,
    gold_chunk_ids: set[str] | None = None,
    router_name: str | None = None,
    router_version: str | None = None,
) -> Episode:
    """Convert one workflow outcome into an Evallab episode.

    ``gold_chunk_ids`` comes from resolving the question's evidence references
    against the corpus that was searched. It is passed in rather than computed
    here because resolution needs the corpus, and an adapter that reached for a
    corpus would couple translation to retrieval.
    """
    steps = _build_steps(outcome, question)

    metadata: dict[str, Any] = {
        # --- experimental identity (spec section 13.1) ---
        "experiment_id": outcome.experiment_id,
        "question_id": outcome.question_id,
        "dataset": question.dataset,
        "split": question.split,
        "document_ids": list(question.document_ids),
        "workflow_action": outcome.action_id.value,
        "workflow_version": outcome.workflow_version,
        "router_name": router_name,
        "router_version": router_version,
        "random_seed": outcome.random_seed,
        "code_commit": outcome.code_commit,
        # --- model configuration ---
        "model_name": outcome.model_configuration.get("model"),
        "prompt_version": outcome.model_configuration.get("prompt_version"),
        "embedding_model": outcome.model_configuration.get("embedding_model"),
        "reranker_model": outcome.model_configuration.get("reranker_model"),
        # --- ground truth, so scorers need no side channel ---
        "reference_answer": question.reference_answer,
        "reference_evidence": list(question.reference_evidence),
        "gold_chunk_ids": sorted(gold_chunk_ids) if gold_chunk_ids else [],
        "question_type": question.question_type,
        # --- what the workflow actually did ---
        "retrieved_chunk_ids": [item.chunk_id for item in outcome.retrieved_items],
        "retrieved_document_ids": sorted({i.document_id for i in outcome.retrieved_items}),
        "cited_chunk_ids": [c.chunk_id for c in outcome.citations],
        "abstained": outcome.abstained,
        "abstention_reason": (
            outcome.abstention_reason.value if outcome.abstention_reason else None
        ),
        "confidence": outcome.confidence,
        # --- operational ---
        "estimated_cost_usd": outcome.estimated_cost_usd,
        "pricing_known": outcome.metadata.get("pricing_known"),
        "total_tokens": outcome.token_usage.total_tokens,
        "input_tokens": outcome.token_usage.input_tokens,
        "output_tokens": outcome.token_usage.output_tokens,
        "latency_ms": (outcome.latency_breakdown.total_ms if outcome.latency_breakdown else None),
        "retrieval_ms": (
            outcome.latency_breakdown.retrieval_ms if outcome.latency_breakdown else None
        ),
        # --- failure signals ---
        "error_status": outcome.error_status,
        "error_detail": outcome.error_detail,
        "hallucinated_citations": outcome.metadata.get("hallucinated_citations", []),
        "n_hallucinated_citations": outcome.metadata.get("n_hallucinated_citations", 0),
    }

    return Episode(
        episode_id=episode_id_for(outcome),
        steps=steps,
        source_framework=SOURCE_FRAMEWORK,
        task_description=question.question_text,
        final_answer=outcome.answer,
        metadata=metadata,
    )


def _build_steps(outcome: WorkflowOutcomeRecord, question: QuestionRecord) -> list[Step]:
    """Render the workflow's execution as canonical steps.

    Retrieval is expressed as a tool call and its result. That is not cosmetic:
    it means agent-specific scorers, which count tool calls and inspect their
    results, can be applied to deterministic workflows too — so A1 and A5 are
    measured on the same footing rather than through different instruments.
    """
    action = outcome.action_id.value
    steps: list[Step] = [
        Step(
            kind=StepKind.MESSAGE,
            agent_id=action,
            agent_name=action,
            content=question.question_text,
            metadata={"phase": "query_received"},
        )
    ]

    if outcome.retrieved_items:
        steps.append(
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id=action,
                agent_name=action,
                tool_name="retrieve",
                tool_args={
                    "query": question.question_text,
                    "top_k": outcome.model_configuration.get("top_k"),
                },
                metadata={"phase": "retrieval_request"},
            )
        )
        steps.append(
            Step(
                kind=StepKind.TOOL_RESULT,
                agent_id=action,
                agent_name=action,
                tool_name="retrieve",
                tool_result=[
                    {
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "rank": item.rank,
                        "score": item.score,
                    }
                    for item in outcome.retrieved_items
                ],
                tool_succeeded=True,
                metadata={
                    "phase": "retrieval_response",
                    "n_retrieved": len(outcome.retrieved_items),
                },
            )
        )

    steps.append(
        Step(
            kind=StepKind.LLM_CALL,
            agent_id=action,
            agent_name=action,
            content=outcome.answer,
            model=outcome.model_configuration.get("model"),
            prompt_tokens=outcome.token_usage.input_tokens,
            completion_tokens=outcome.token_usage.output_tokens,
            metadata={
                "phase": "answer_generation",
                "prompt_version": outcome.model_configuration.get("prompt_version"),
                "temperature": outcome.model_configuration.get("temperature"),
                "error_status": outcome.error_status,
            },
        )
    )

    if outcome.abstained:
        steps.append(
            Step(
                kind=StepKind.CUSTOM,
                agent_id=action,
                agent_name=action,
                content=(outcome.abstention_reason.value if outcome.abstention_reason else None),
                metadata={"phase": "abstention"},
            )
        )
    elif outcome.citations:
        steps.append(
            Step(
                kind=StepKind.CUSTOM,
                agent_id=action,
                agent_name=action,
                content=f"{len(outcome.citations)} citation(s)",
                metadata={
                    "phase": "citation",
                    "citations": [
                        {"chunk_id": c.chunk_id, "quoted_text": c.quoted_text}
                        for c in outcome.citations
                    ],
                },
            )
        )

    return steps


def outcomes_to_episodes(
    outcomes: list[WorkflowOutcomeRecord],
    questions: dict[str, QuestionRecord],
    *,
    gold_chunk_ids: dict[str, set[str]] | None = None,
) -> list[Episode]:
    """Convert many outcomes, skipping any whose question is unknown.

    An outcome without its question cannot be scored — there is no reference
    answer to compare against — so it is skipped rather than turned into an
    episode that would silently score zero on every dimension.
    """
    gold = gold_chunk_ids or {}
    episodes: list[Episode] = []
    for outcome in outcomes:
        question = questions.get(outcome.question_id)
        if question is None:
            continue
        episodes.append(
            outcome_to_episode(outcome, question, gold_chunk_ids=gold.get(outcome.question_id))
        )
    return episodes
