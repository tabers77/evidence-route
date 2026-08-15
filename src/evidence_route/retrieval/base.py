"""The retriever interface every retrieval action shares.

BM25, dense and hybrid retrieval all satisfy this protocol, which is what lets
the workflow layer swap them and compare them on identical questions
(spec section 21, week 4 exit criteria).

Retrievers return :class:`~evidence_route.storage.records.RetrievedItem` — the
same type the outcome record stores — so there is one vocabulary for "a
retrieved chunk" from retrieval through scoring to the stored result. A separate
internal hit type would only have to be converted, and every conversion is a
chance to lose the provenance the evaluation depends on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evidence_route.documents.models import Chunk
from evidence_route.storage.records import RetrievedItem

__all__ = ["Retriever", "chunk_to_item"]


@runtime_checkable
class Retriever(Protocol):
    """Ranks chunks against a query."""

    @property
    def name(self) -> str:
        """Stable identifier recorded in the outcome's model configuration."""
        ...

    def search(self, query: str, k: int = 10) -> list[RetrievedItem]:
        """Return the top ``k`` chunks, rank 1 first.

        Implementations must be deterministic, including how they break score
        ties. Two runs over the same index and query have to produce the same
        ordering, or retrieval results are not reproducible and neither are any
        of the comparisons built on them.
        """
        ...


def chunk_to_item(
    chunk: Chunk,
    *,
    rank: int,
    score: float | None = None,
    include_text: bool = True,
    component_ranks: dict[str, int] | None = None,
) -> RetrievedItem:
    """Convert a chunk into the canonical retrieved-item record."""
    return RetrievedItem(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        rank=rank,
        text=chunk.text if include_text else None,
        score=score,
        component_ranks=component_ranks or {},
    )
