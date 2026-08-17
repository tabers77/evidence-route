"""Action A5: bounded multi-step agentic retrieval.

The agent is a candidate, not the project's identity. It exists to answer one
question — does decomposition and repeated searching improve complex questions
enough to justify its cost — and the honest answer may well be no.

This workflow does not inherit the shared generation path, because A5 does not
make one call: it runs a loop, and its outcome is assembled from a trace rather
than a single response. What it does keep is the *outcome schema*. A5 produces
the same ``WorkflowOutcomeRecord`` as every other action, which is the week-5
exit criterion and the thing that lets the outcome matrix compare them at all.

Every limit breach becomes an explicit error status. An agent that exhausts its
steps and returns what it had would be scored as an ordinary wrong answer, and
resource exhaustion would disappear from the failure taxonomy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from evidence_route.agents.loop import AgentLimits, AgentRun, BoundedAgent, TerminationReason
from evidence_route.agents.tools import ToolBox
from evidence_route.generation.client import GenerationProvider
from evidence_route.generation.cost import estimate_cost
from evidence_route.generation.schema import StructuredAnswer
from evidence_route.retrieval.bm25 import BM25Config, BM25Retriever
from evidence_route.storage.records import (
    Citation,
    ErrorStatus,
    LatencyBreakdown,
    RetrievedItem,
    TokenUsage,
    WorkflowOutcomeRecord,
)
from evidence_route.workflows.actions import Action
from evidence_route.workflows.base import WorkflowContext

__all__ = ["AgenticWorkflow"]

#: Limit breaches mapped onto the outcome record's error vocabulary.
_TERMINATION_TO_ERROR: dict[str, ErrorStatus] = {
    TerminationReason.STEP_LIMIT: "step_limit_exceeded",
    TerminationReason.TOOL_CALL_LIMIT: "step_limit_exceeded",
    TerminationReason.TOKEN_LIMIT: "budget_exceeded",
    TerminationReason.TIME_LIMIT: "timeout",
    TerminationReason.PROVIDER_ERROR: "provider_error",
    TerminationReason.INVALID_TERMINATION: "parse_error",
}


@dataclass
class AgenticWorkflow:
    """A5 — a bounded tool loop that produces the standard outcome record."""

    provider: GenerationProvider
    limits: AgentLimits = field(default_factory=AgentLimits)
    bm25_config: BM25Config = field(default_factory=BM25Config)
    max_search_results: int = 5
    deployment_ref: str = "chat"
    workflow_version: str = "0.1.0"

    @property
    def action(self) -> Action:
        return Action.AGENTIC

    @property
    def version(self) -> str:
        return self.workflow_version

    def run(self, context: WorkflowContext) -> WorkflowOutcomeRecord:
        started = time.perf_counter()

        if not context.chunks:
            # Nothing to search. Recorded as an outcome rather than an
            # exception, consistent with every other action.
            return self._build(
                context,
                run=AgentRun(termination_reason=TerminationReason.INVALID_TERMINATION),
                answer=None,
                items=[],
                latency=LatencyBreakdown(total_ms=(time.perf_counter() - started) * 1000),
                error_status="ok",
                abstention="insufficient_evidence",
            )

        toolbox = ToolBox(
            chunks=context.chunks,
            retriever=BM25Retriever(config=self.bm25_config).index(context.chunks),
            max_search_results=self.max_search_results,
        )
        agent = BoundedAgent(
            provider=self.provider,
            limits=self.limits,
            deployment_ref=self.deployment_ref,
        )
        run = agent.run(context.question.question_text, toolbox, seed=context.random_seed)

        latency = LatencyBreakdown(
            total_ms=(time.perf_counter() - started) * 1000,
            generation_ms=run.elapsed_seconds * 1000,
        )
        items = self._retrieved_items(run, context)

        if not run.succeeded:
            return self._build(
                context,
                run=run,
                answer=None,
                items=items,
                latency=latency,
                error_status=_TERMINATION_TO_ERROR.get(run.termination_reason, "parse_error"),
                error_detail=run.error_detail or run.termination_reason,
            )

        try:
            answer = StructuredAnswer.model_validate(self._normalise(run.submission or {}))
        except Exception as exc:
            return self._build(
                context,
                run=run,
                answer=None,
                items=items,
                latency=latency,
                error_status="parse_error",
                error_detail=f"submit_answer payload was invalid: {exc}",
            )

        return self._build(context, run=run, answer=answer, items=items, latency=latency)

    # -- helpers ------------------------------------------------------------
    def _normalise(self, submission: dict) -> dict:
        """Coerce the submit_answer payload into the structured answer schema."""
        payload = dict(submission)
        citations = payload.get("citations") or []
        payload["citations"] = [
            {"chunk_id": c} if isinstance(c, str) else c
            for c in citations
            if isinstance(c, (str, dict))
        ]
        if payload.get("abstention_reason") and not payload.get("abstained"):
            payload["abstained"] = True
        if payload.get("answer") == "":
            payload["answer"] = None
        payload.pop("thought", None)
        return payload

    def _retrieved_items(self, run: AgentRun, context: WorkflowContext) -> list[RetrievedItem]:
        """The chunks the agent actually read, in the order it found them.

        This is A5's evidence set. Using anything else — say, a fresh BM25 run —
        would score the agent against evidence it never saw, and citation
        validation would stop meaning anything.
        """
        by_id = {chunk.chunk_id: chunk for chunk in context.chunks}
        items: list[RetrievedItem] = []
        for rank, chunk_id in enumerate(run.chunks_seen, start=1):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            items.append(
                RetrievedItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    rank=rank,
                    text=chunk.text,
                )
            )
        return items

    def _build(
        self,
        context: WorkflowContext,
        *,
        run: AgentRun,
        answer: StructuredAnswer | None,
        items: list[RetrievedItem],
        latency: LatencyBreakdown,
        error_status: ErrorStatus = "ok",
        error_detail: str | None = None,
        abstention: str | None = None,
    ) -> WorkflowOutcomeRecord:
        usage = TokenUsage(input_tokens=run.input_tokens, output_tokens=run.output_tokens)
        cost = estimate_cost(usage, None)

        retrieved_ids = {item.chunk_id for item in items}
        hallucinated: list[str] = []
        citations: list[Citation] = []
        if answer is not None:
            hallucinated = answer.validate_citations(retrieved_ids)
            document_for = {item.chunk_id: item.document_id for item in items}
            citations = [
                Citation(
                    chunk_id=c.chunk_id,
                    document_id=document_for[c.chunk_id],
                    quoted_text=c.quoted_text,
                    claim=c.claim,
                )
                for c in answer.citations
                if c.chunk_id in document_for
            ]

        abstention_reason = answer.abstention_reason if answer else None
        if abstention_reason is None and abstention is not None:
            from evidence_route.workflows.actions import AbstentionReason

            abstention_reason = AbstentionReason(abstention)

        metadata: dict[str, Any] = {
            # Agent-specific metrics (spec section 11.6), computed from the trace
            # while it is still in hand — they cannot be recovered from the
            # outcome record alone.
            "termination_reason": run.termination_reason,
            "n_steps": len(run.steps),
            "n_tool_calls": run.n_tool_calls,
            "valid_tool_call_rate": round(run.valid_tool_call_rate, 4),
            "repeated_call_rate": round(run.repeated_call_rate, 4),
            "evidence_per_call": round(run.evidence_per_call, 4),
            "n_chunks_seen": len(run.chunks_seen),
            "tools_used": [s.call.name for s in run.steps if s.call is not None],
            "hallucinated_citations": hallucinated,
            "n_hallucinated_citations": len(hallucinated),
            "pricing_known": cost.pricing_known,
            "limits": {
                "max_steps": self.limits.max_steps,
                "max_tool_calls": self.limits.max_tool_calls,
                "max_total_tokens": self.limits.max_total_tokens,
                "max_wall_clock_seconds": self.limits.max_wall_clock_seconds,
            },
        }

        return WorkflowOutcomeRecord(
            experiment_id=context.experiment_id,
            question_id=context.question.question_id,
            action_id=Action.AGENTIC,
            workflow_version=self.workflow_version,
            model_configuration={
                "deployment_ref": self.deployment_ref,
                "prompt_version": "agentic_v1",
                "provider": self.provider.name,
                "model": None,
                "max_steps": self.limits.max_steps,
            },
            retrieved_items=items,
            answer=answer.answer if answer else None,
            citations=citations,
            abstention_reason=abstention_reason,
            confidence=answer.confidence if answer else None,
            latency_breakdown=latency,
            token_usage=usage,
            estimated_cost_usd=cost.usd_or_zero,
            error_status=error_status,
            error_detail=error_detail,
            random_seed=context.random_seed,
            code_commit=context.code_commit,
            metadata=metadata,
        )
