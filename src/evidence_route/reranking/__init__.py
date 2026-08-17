"""Reranking (action A4).

Reranking is a candidate whose cost must be justified, not an assumed
improvement. The measurement that matters is whether it raises evidence quality
*enough* to pay for its added latency — which is why the reranking stage has its
own entry in ``LatencyBreakdown`` rather than being folded into retrieval time.

Both candidates for protocol decision 6 are implemented, so the choice can be
made on measured evidence rather than on install size:

    ``CrossEncoderReranker``   free per query, deterministic, offline, ~3-4 GB
    ``LLMReranker``            billed per query, no local install, listwise
    ``LexicalOverlapReranker`` deterministic stub for CI; is_learned = False
"""

from evidence_route.reranking.base import LexicalOverlapReranker, Reranker, reorder
from evidence_route.reranking.cross_encoder import CrossEncoderReranker
from evidence_route.reranking.llm import RERANK_PROMPT_V1, LLMReranker

__all__ = [
    "RERANK_PROMPT_V1",
    "CrossEncoderReranker",
    "LLMReranker",
    "LexicalOverlapReranker",
    "Reranker",
    "reorder",
]
