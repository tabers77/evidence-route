"""Action A0: direct answer, no retrieval.

The baseline that makes every other action interpretable. Without it, a high
score from A1 could mean retrieval worked — or that the model already knew the
answer and the retrieved evidence was decoration.

Three things this action measures (spec section 6, A0):

*Parametric-memory hallucination.* How often the model produces a confident,
plausible, wrong answer with nothing behind it.

*Correct answers without evidence.* A right answer here is not a success. These
are public filings from well-known companies published before the model's
training cutoff, so a correct A0 answer is most likely **contamination**, and it
sets a floor that the retrieval actions must clear to demonstrate they add
anything at all.

*The cheapest possible workflow.* Its cost and latency anchor the low end of the
Pareto frontier the router navigates.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_route.generation.client import GenerationProvider
from evidence_route.generation.prompts import DIRECT_V1, PromptTemplate
from evidence_route.storage.records import RetrievedItem
from evidence_route.workflows.actions import Action
from evidence_route.workflows.base import GenerativeWorkflow, WorkflowContext

__all__ = ["DirectAnswerWorkflow"]


@dataclass
class DirectAnswerWorkflow(GenerativeWorkflow):
    """Answers from the model's own knowledge, with no documents."""

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        template: PromptTemplate = DIRECT_V1,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        deployment_ref: str = "chat",
        workflow_version: str = "0.1.0",
    ) -> None:
        super().__init__(
            provider=provider,
            template=template,
            action_id=Action.DIRECT,
            top_k=0,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            deployment_ref=deployment_ref,
            workflow_version=workflow_version,
        )

    def retrieve(self, context: WorkflowContext) -> list[RetrievedItem]:
        """No retrieval, by definition.

        Returning nothing rather than ignoring a corpus keeps the accounting
        honest: A0's retrieval latency is genuinely zero, and its outcome record
        shows an empty evidence list rather than evidence it did not use.
        """
        return []

    def build_user_prompt(self, context: WorkflowContext, items: list[RetrievedItem]) -> str:
        # The direct template takes only the question — there is no evidence
        # slot to fill.
        return self.template.render_user(question=context.question.question_text)
