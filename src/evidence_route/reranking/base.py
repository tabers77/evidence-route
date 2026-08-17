"""The reranker interface, and a deterministic stub for offline use.

A reranker reorders an already-retrieved candidate set using a more expensive
scoring function than the retriever could afford over the whole corpus. That
trade — spend more per candidate, on fewer candidates — is the entire premise of
action A4, and whether it pays is exactly what the experiment measures.

``is_learned`` mirrors ``Embedder.is_semantic`` and exists for the same reason:
:class:`LexicalOverlapReranker` runs offline and in CI, but it scores by token
overlap, which is roughly what BM25 already did. Concluding "reranking does not
help" from a run using it would be an artifact of the stub, so the flag travels
with the result and the reporting layer can refuse it.

Rerankers preserve the pre-rerank position in ``component_ranks``. Without it,
"did reranking move anything, and was the movement an improvement" is
unanswerable after the fact — and that is the only question A4 exists to answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evidence_route.retrieval.tokenization import tokenize
from evidence_route.storage.records import RetrievedItem

__all__ = ["LexicalOverlapReranker", "Reranker", "reorder"]


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved candidates."""

    @property
    def name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def is_learned(self) -> bool:
        """False for stubs that only approximate what retrieval already did."""
        ...

    def rerank(
        self, query: str, items: list[RetrievedItem], k: int = 10
    ) -> list[RetrievedItem]: ...


def reorder(items: list[RetrievedItem], scores: list[float], k: int) -> list[RetrievedItem]:
    """Rebuild a ranked list from new scores, keeping the original position.

    Ties break on the pre-rerank rank, so a reranker that cannot separate two
    candidates leaves retrieval's ordering intact rather than shuffling them
    arbitrarily. Arbitrary shuffling would show up as reranker "effect" in the
    comparison while being pure noise.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if len(scores) != len(items):
        raise ValueError(
            f"Got {len(scores)} scores for {len(items)} candidates; a misaligned "
            f"batch would attach every score to the wrong chunk."
        )

    order = sorted(range(len(items)), key=lambda i: (-scores[i], items[i].rank, items[i].chunk_id))

    reranked: list[RetrievedItem] = []
    for new_rank, index in enumerate(order[:k], start=1):
        original = items[index]
        reranked.append(
            RetrievedItem(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                rank=new_rank,
                text=original.text,
                score=scores[index],
                # Retrieval's own ranking, preserved so movement is measurable.
                component_ranks={**original.component_ranks, "retrieval": original.rank},
            )
        )
    return reranked


@dataclass
class LexicalOverlapReranker:
    """Scores by query-token overlap. Deterministic, free, offline.

    Exercises the A4 code path without a model. Not a cross-encoder: it has no
    notion of whether a passage *answers* the question, only whether it repeats
    its words — so ``is_learned`` is False and results carry that caveat.
    """

    @property
    def name(self) -> str:
        return "lexical_overlap"

    @property
    def model_version(self) -> str:
        return "lexical_overlap-v1"

    @property
    def is_learned(self) -> bool:
        return False

    def rerank(self, query: str, items: list[RetrievedItem], k: int = 10) -> list[RetrievedItem]:
        if not items:
            return []
        query_terms = set(tokenize(query))
        scores = []
        for item in items:
            terms = set(tokenize(item.text or ""))
            overlap = len(query_terms & terms)
            scores.append(overlap / len(query_terms) if query_terms else 0.0)
        return reorder(items, scores, k)
