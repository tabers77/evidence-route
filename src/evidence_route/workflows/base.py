"""Shared workflow execution.

Every action follows the same shape — retrieve, prompt, generate, parse, record
— and differs only in the retrieval step and the prompt. Putting that shape in
one place is what makes the actions comparable: identical timing points,
identical token accounting, identical error handling. If each workflow measured
itself, differences between actions would partly reflect differences in
instrumentation.

**Failures become outcomes, never exceptions.** A provider error, a truncated
response, an unparseable answer — each produces a ``WorkflowOutcomeRecord`` with
an explicit ``error_status`` and whatever cost was already incurred. Raising
would abandon the run and, worse, bias the outcome matrix: the questions that
fail are the hard ones, and dropping them makes every action look better than it
is (spec section 21, week 7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evidence_route.documents.models import Chunk
from evidence_route.generation.client import (
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderError,
)
from evidence_route.generation.cost import estimate_cost
from evidence_route.generation.prompts import PromptTemplate, format_evidence
from evidence_route.generation.schema import StructuredAnswer, parse_structured_answer
from evidence_route.storage.records import (
    Citation,
    LatencyBreakdown,
    QuestionRecord,
    RetrievedItem,
    TokenUsage,
    WorkflowOutcomeRecord,
)
from evidence_route.workflows.actions import Action

__all__ = ["GenerativeWorkflow", "Workflow", "WorkflowContext"]


@dataclass
class WorkflowContext:
    """Everything a workflow needs to answer one question."""

    question: QuestionRecord
    experiment_id: str
    #: Candidate chunks for retrieval, already scoped to the documents the
    #: question is asked against. Empty for actions that do not retrieve.
    chunks: list[Chunk] = field(default_factory=list)
    random_seed: int | None = None
    code_commit: str | None = None


@runtime_checkable
class Workflow(Protocol):
    """One candidate action."""

    @property
    def action(self) -> Action: ...

    @property
    def version(self) -> str: ...

    def run(self, context: WorkflowContext) -> WorkflowOutcomeRecord: ...


@dataclass
class GenerativeWorkflow:
    """Base for actions that call a model.

    Subclasses override :meth:`retrieve`; everything else is shared so the
    actions stay measurably comparable.
    """

    provider: GenerationProvider
    template: PromptTemplate
    action_id: Action
    top_k: int = 10
    temperature: float = 0.0
    max_output_tokens: int = 512
    deployment_ref: str = "chat"
    workflow_version: str = "0.1.0"

    @property
    def action(self) -> Action:
        return self.action_id

    @property
    def version(self) -> str:
        return self.workflow_version

    # -- overridden by subclasses -------------------------------------------
    def retrieve(self, context: WorkflowContext) -> list[RetrievedItem]:
        """Return evidence for the question. Default: none (action A0)."""
        return []

    def build_user_prompt(self, context: WorkflowContext, items: list[RetrievedItem]) -> str:
        """Render the user message. Default assumes an evidence template."""
        return self.template.render_user(
            evidence=format_evidence(items), question=context.question.question_text
        )

    # -- shared execution ---------------------------------------------------
    def run(self, context: WorkflowContext) -> WorkflowOutcomeRecord:
        started = time.perf_counter()

        retrieval_started = time.perf_counter()
        try:
            items = self.retrieve(context)
        except Exception as exc:
            return self._failed(
                context,
                items=[],
                error_status="provider_error",
                detail=f"retrieval failed: {exc}",
                latency=self._latency(started, 0.0, 0.0),
            )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        request = GenerationRequest(
            system=self.template.system,
            user=self.build_user_prompt(context, items),
            deployment_ref=self.deployment_ref,
            prompt_version=self.template.version,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            seed=context.random_seed,
        )

        generation_started = time.perf_counter()
        try:
            response = self.provider.complete(request)
        except ProviderError as exc:
            return self._failed(
                context,
                items=items,
                error_status="provider_error",
                detail=str(exc),
                latency=self._latency(
                    started, retrieval_ms, (time.perf_counter() - generation_started) * 1000
                ),
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        latency = self._latency(started, retrieval_ms, generation_ms)

        usage = TokenUsage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            retry_input_tokens=response.retry_input_tokens,
            retry_output_tokens=response.retry_output_tokens,
        )

        try:
            answer = parse_structured_answer(response.text)
        except Exception as exc:
            # The call was billed even though the response was unusable, so the
            # cost is recorded. Treating a parse failure as free would make
            # unreliable formatting look cheap.
            return self._failed(
                context,
                items=items,
                error_status="parse_error",
                detail=(
                    f"{'response truncated at max_output_tokens; ' if response.was_truncated else ''}"
                    f"{exc}"
                ),
                latency=latency,
                usage=usage,
                response=response,
            )

        return self._succeeded(context, items, answer, usage, latency, response)

    # -- record construction ------------------------------------------------
    def _latency(
        self, started: float, retrieval_ms: float, generation_ms: float
    ) -> LatencyBreakdown:
        return LatencyBreakdown(
            total_ms=(time.perf_counter() - started) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def _model_configuration(self, response: GenerationResponse | None = None) -> dict[str, Any]:
        return {
            "deployment_ref": self.deployment_ref,
            "prompt_version": self.template.version,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "provider": self.provider.name,
            # The model an Azure deployment resolved to. Recorded because a
            # deployment can be re-pointed with no change here, and results
            # from different underlying models must not be pooled.
            "model": response.model if response else None,
        }

    def _succeeded(
        self,
        context: WorkflowContext,
        items: list[RetrievedItem],
        answer: StructuredAnswer,
        usage: TokenUsage,
        latency: LatencyBreakdown,
        response: GenerationResponse,
    ) -> WorkflowOutcomeRecord:
        retrieved_ids = {item.chunk_id for item in items}
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

        cost = estimate_cost(usage, response.model)

        return WorkflowOutcomeRecord(
            experiment_id=context.experiment_id,
            question_id=context.question.question_id,
            action_id=self.action_id,
            workflow_version=self.workflow_version,
            model_configuration=self._model_configuration(response),
            retrieved_items=items,
            answer=answer.answer,
            citations=citations,
            abstention_reason=answer.abstention_reason,
            confidence=answer.confidence,
            latency_breakdown=latency,
            token_usage=usage,
            estimated_cost_usd=cost.usd_or_zero,
            random_seed=context.random_seed,
            code_commit=context.code_commit,
            metadata={
                # A citation naming a chunk the model was never shown. Counted
                # rather than raised, so it lands in the failure taxonomy.
                "hallucinated_citations": hallucinated,
                "n_hallucinated_citations": len(hallucinated),
                # Without this flag a zero cost is indistinguishable from an
                # unpriced one, and the budget ceiling silently stops working.
                "pricing_known": cost.pricing_known,
                "cost_note": cost.note,
                "from_cache": response.from_cache,
                "finish_reason": response.finish_reason,
                "reasoning": answer.reasoning,
            },
        )

    def _failed(
        self,
        context: WorkflowContext,
        *,
        items: list[RetrievedItem],
        error_status: Any,
        detail: str,
        latency: LatencyBreakdown,
        usage: TokenUsage | None = None,
        response: GenerationResponse | None = None,
    ) -> WorkflowOutcomeRecord:
        """Record a failure with whatever was already spent on it."""
        usage = usage or TokenUsage()
        cost = estimate_cost(usage, response.model if response else None)
        return WorkflowOutcomeRecord(
            experiment_id=context.experiment_id,
            question_id=context.question.question_id,
            action_id=self.action_id,
            workflow_version=self.workflow_version,
            model_configuration=self._model_configuration(response),
            retrieved_items=items,
            answer=None,
            latency_breakdown=latency,
            token_usage=usage,
            estimated_cost_usd=cost.usd_or_zero,
            error_status=error_status,
            error_detail=detail,
            random_seed=context.random_seed,
            code_commit=context.code_commit,
            metadata={
                "pricing_known": cost.pricing_known,
                "raw_response": (response.text[:500] if response else None),
                "finish_reason": response.finish_reason if response else None,
            },
        )
