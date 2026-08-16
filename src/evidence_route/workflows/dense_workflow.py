"""Actions A2 and A3: dense and hybrid retrieval plus generation.

A2 tests whether semantic matching finds evidence that lexical matching misses —
the question asking about "revenue" where the filing says "net sales". A3 tests
whether fusing the two beats either alone, and at what cost.

The two live together because A3 is A2 plus A1 plus a documented fusion rule,
and separating them would duplicate the index-construction logic that dominates
both.

Index construction happens per question over the chunks the question names, and
its time is counted in retrieval latency rather than amortised away. That is not
a detail: embedding a corpus is the expensive part of dense retrieval, and an
action that hid it would look artificially cheap in exactly the cost comparison
the router is meant to arbitrate.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence_route.generation.client import GenerationProvider
from evidence_route.generation.prompts import GROUNDED_V1, PromptTemplate
from evidence_route.retrieval.bm25 import BM25Config, BM25Retriever
from evidence_route.retrieval.dense import DenseConfig, DenseRetriever
from evidence_route.retrieval.embeddings import Embedder, HashingEmbedder
from evidence_route.retrieval.hybrid import HybridConfig, HybridRetriever
from evidence_route.storage.records import RetrievedItem
from evidence_route.workflows.actions import Action
from evidence_route.workflows.base import GenerativeWorkflow, WorkflowContext

__all__ = ["DenseWorkflow", "HybridWorkflow"]


@dataclass
class DenseWorkflow(GenerativeWorkflow):
    """A2 — embedding retrieval, then answer strictly from what was retrieved."""

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        embedder: Embedder | None = None,
        template: PromptTemplate = GROUNDED_V1,
        dense_config: DenseConfig | None = None,
        top_k: int = 10,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        deployment_ref: str = "chat",
        workflow_version: str = "0.1.0",
        retriever: DenseRetriever | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            template=template,
            action_id=Action.DENSE,
            top_k=top_k,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            deployment_ref=deployment_ref,
            workflow_version=workflow_version,
        )
        self.embedder = embedder or HashingEmbedder()
        self.dense_config = dense_config or DenseConfig()
        #: A prebuilt index over the whole corpus. Supplying it avoids
        #: re-embedding per question, at the cost of losing document scoping.
        self.retriever = retriever

    def retrieve(self, context: WorkflowContext) -> list[RetrievedItem]:
        if self.retriever is not None:
            return self.retriever.search(context.question.question_text, k=self.top_k)
        if not context.chunks:
            return []
        retriever = DenseRetriever(embedder=self.embedder, config=self.dense_config).index(
            context.chunks
        )
        return retriever.search(context.question.question_text, k=self.top_k)

    def _model_configuration(self, response=None) -> dict:
        config = super()._model_configuration(response)
        # Recorded because an index built with a different embedding model is a
        # different index, and results must not be pooled across them.
        config["embedding_model"] = self.embedder.model_version
        config["embedding_is_semantic"] = self.embedder.is_semantic
        return config


@dataclass
class HybridWorkflow(GenerativeWorkflow):
    """A3 — reciprocal rank fusion of BM25 and dense, then generation."""

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        embedder: Embedder | None = None,
        template: PromptTemplate = GROUNDED_V1,
        bm25_config: BM25Config | None = None,
        dense_config: DenseConfig | None = None,
        hybrid_config: HybridConfig | None = None,
        top_k: int = 10,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        deployment_ref: str = "chat",
        workflow_version: str = "0.1.0",
    ) -> None:
        super().__init__(
            provider=provider,
            template=template,
            action_id=Action.HYBRID,
            top_k=top_k,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            deployment_ref=deployment_ref,
            workflow_version=workflow_version,
        )
        self.embedder = embedder or HashingEmbedder()
        self.bm25_config = bm25_config or BM25Config()
        self.dense_config = dense_config or DenseConfig()
        self.hybrid_config = hybrid_config or HybridConfig()

    def retrieve(self, context: WorkflowContext) -> list[RetrievedItem]:
        if not context.chunks:
            return []

        hybrid = HybridRetriever(
            components={
                "bm25": BM25Retriever(config=self.bm25_config).index(context.chunks),
                "dense": DenseRetriever(embedder=self.embedder, config=self.dense_config).index(
                    context.chunks
                ),
            },
            config=self.hybrid_config,
        )
        return hybrid.search(context.question.question_text, k=self.top_k)

    def _model_configuration(self, response=None) -> dict:
        config = super()._model_configuration(response)
        config["embedding_model"] = self.embedder.model_version
        config["embedding_is_semantic"] = self.embedder.is_semantic
        config["fusion"] = "reciprocal_rank_fusion"
        config["rrf_k"] = self.hybrid_config.rrf_k
        config["candidate_k"] = self.hybrid_config.candidate_k
        return config
