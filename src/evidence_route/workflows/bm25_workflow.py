"""Action A1: BM25 retrieval plus answer generation.

The strong classical baseline. On this corpus it is not a straw man: most
FinanceBench questions name a specific line item and expect a specific figure,
which is exactly the shape lexical retrieval handles well — provided the
tokenizer keeps ``66,608`` intact, which is why
:mod:`evidence_route.retrieval.tokenization` exists.

Treating A1 seriously matters for the project's central claim. If the learned
router cannot beat "always BM25", that is a real finding, and it can only be a
real finding if BM25 was implemented properly rather than handicapped.

The index is built per question over the chunks in ``context``, which are
already scoped to the documents the question names. That mirrors how
FinanceBench is posed — the question tells you which filing to look in — and it
keeps A1 honest about its own cost: index construction time is counted in the
retrieval latency, not amortised away.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_route.generation.client import GenerationProvider
from evidence_route.generation.prompts import GROUNDED_V1, PromptTemplate
from evidence_route.retrieval.bm25 import BM25Config, BM25Retriever
from evidence_route.storage.records import RetrievedItem
from evidence_route.workflows.actions import Action
from evidence_route.workflows.base import GenerativeWorkflow, WorkflowContext

__all__ = ["BM25Workflow"]


@dataclass
class BM25Workflow(GenerativeWorkflow):
    """Retrieves lexically, then answers strictly from what it retrieved."""

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        template: PromptTemplate = GROUNDED_V1,
        bm25_config: BM25Config | None = None,
        top_k: int = 10,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        deployment_ref: str = "chat",
        workflow_version: str = "0.1.0",
        retriever: BM25Retriever | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            template=template,
            action_id=Action.BM25,
            top_k=top_k,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            deployment_ref=deployment_ref,
            workflow_version=workflow_version,
        )
        self.bm25_config = bm25_config or BM25Config()
        #: A prebuilt index, when the caller has one for the whole corpus.
        #: Supplying it avoids rebuilding per question, at the cost of losing
        #: document scoping — so it is opt-in rather than the default.
        self.retriever = retriever

    def retrieve(self, context: WorkflowContext) -> list[RetrievedItem]:
        """Rank the question's candidate chunks with BM25.

        Returns an empty list when there is nothing to search. The workflow then
        prompts with no evidence, and the grounded template instructs the model
        to abstain — which is the correct outcome, and one that gets recorded
        rather than turning into a fabricated answer.
        """
        if self.retriever is not None:
            return self.retriever.search(context.question.question_text, k=self.top_k)

        if not context.chunks:
            return []

        retriever = BM25Retriever(config=self.bm25_config).index(context.chunks)
        return retriever.search(context.question.question_text, k=self.top_k)
