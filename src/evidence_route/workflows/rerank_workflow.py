"""Action A4: hybrid retrieval, reranking, then generation.

A4 buys evidence quality with latency and cost. Whether that trade pays is the
whole question, so the implementation is careful about two things.

**A wider candidate set, then a narrower final one.** Retrieval fetches
``candidate_k`` chunks and reranking keeps ``top_k``. Reranking the same ten
chunks retrieval already ranked would mostly reorder them, and the interesting
case is a chunk retrieval placed 25th that the reranker promotes to 2nd. Without
the wider net there is nothing for reranking to find.

**Reranking latency is measured separately.** ``LatencyBreakdown`` has its own
field for it, so the comparison against A3 answers "what did reranking cost"
rather than "A4 is slower".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from evidence_route.generation.client import GenerationProvider
from evidence_route.generation.prompts import GROUNDED_V1, PromptTemplate
from evidence_route.reranking.base import LexicalOverlapReranker, Reranker
from evidence_route.retrieval.bm25 import BM25Config, BM25Retriever
from evidence_route.retrieval.dense import DenseConfig, DenseRetriever
from evidence_route.retrieval.embeddings import Embedder, HashingEmbedder
from evidence_route.retrieval.hybrid import HybridConfig, HybridRetriever
from evidence_route.storage.records import LatencyBreakdown, RetrievedItem
from evidence_route.workflows.actions import Action
from evidence_route.workflows.base import GenerativeWorkflow, WorkflowContext

__all__ = ["RerankWorkflow"]


@dataclass
class RerankWorkflow(GenerativeWorkflow):
    """A4 — fusion, rerank, generate."""

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        reranker: Reranker | None = None,
        embedder: Embedder | None = None,
        template: PromptTemplate = GROUNDED_V1,
        bm25_config: BM25Config | None = None,
        dense_config: DenseConfig | None = None,
        hybrid_config: HybridConfig | None = None,
        candidate_k: int = 30,
        top_k: int = 10,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        deployment_ref: str = "chat",
        workflow_version: str = "0.1.0",
    ) -> None:
        super().__init__(
            provider=provider,
            template=template,
            action_id=Action.HYBRID_RERANK,
            top_k=top_k,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            deployment_ref=deployment_ref,
            workflow_version=workflow_version,
        )
        self.reranker = reranker or LexicalOverlapReranker()
        self.embedder = embedder or HashingEmbedder()
        self.bm25_config = bm25_config or BM25Config()
        self.dense_config = dense_config or DenseConfig()
        self.hybrid_config = hybrid_config or HybridConfig()
        self.candidate_k = candidate_k
        self._last_rerank_ms = 0.0

    def retrieve(self, context: WorkflowContext) -> list[RetrievedItem]:
        self._last_rerank_ms = 0.0
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
        candidates = hybrid.search(context.question.question_text, k=self.candidate_k)
        if not candidates:
            return []

        started = time.perf_counter()
        reranked = self.reranker.rerank(context.question.question_text, candidates, k=self.top_k)
        self._last_rerank_ms = (time.perf_counter() - started) * 1000
        return reranked

    def _latency(
        self, started: float, retrieval_ms: float, generation_ms: float
    ) -> LatencyBreakdown:
        """Split reranking out of retrieval time.

        Otherwise the A3-versus-A4 comparison shows only that A4 is slower,
        without saying which stage was responsible.
        """
        rerank_ms = self._last_rerank_ms
        return LatencyBreakdown(
            total_ms=(time.perf_counter() - started) * 1000,
            retrieval_ms=max(retrieval_ms - rerank_ms, 0.0),
            reranking_ms=rerank_ms,
            generation_ms=generation_ms,
        )

    def _model_configuration(self, response=None) -> dict:
        config = super()._model_configuration(response)
        config["embedding_model"] = self.embedder.model_version
        config["fusion"] = "reciprocal_rank_fusion"
        config["candidate_k"] = self.candidate_k
        config["reranker"] = self.reranker.name
        config["reranker_model"] = self.reranker.model_version
        # Mirrors embedding_is_semantic: a stub reranker cannot support a
        # conclusion about whether reranking helps.
        config["reranker_is_learned"] = self.reranker.is_learned
        return config
