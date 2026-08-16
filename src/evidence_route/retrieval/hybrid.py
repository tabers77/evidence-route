"""Hybrid retrieval by reciprocal rank fusion (action A3).

The fusion method is stated, not implied. Calling a system "hybrid" without
documenting how the rankings combine is not a method (spec section 6, A3).

    RRF(chunk) = sum over retrievers of  weight / (rrf_k + rank)

**Why ranks and not scores.** BM25 scores are unbounded and corpus-dependent —
a score of 8.4 means nothing on its own. Cosine similarities live in [-1, 1].
Adding or averaging them directly would let whichever retriever happens to
produce larger numbers dominate the fusion, and that scale difference has
nothing to do with retrieval quality. Ranks are comparable by construction, so
RRF sidesteps normalization entirely rather than inventing a mapping between two
incomparable scales.

**Why the constant.** ``rrf_k`` flattens the contribution of top ranks: with
k=60 the difference between rank 1 and rank 2 is small, so a retriever must be
*consistently* high to win. Setting it near zero would let a single retriever's
top hit dominate, which defeats the point of fusing.

A chunk found by both retrievers accumulates from both, which is the behaviour
that makes fusion worth its extra cost: agreement is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence_route.storage.records import RetrievedItem

__all__ = ["HybridConfig", "HybridRetriever", "reciprocal_rank_fusion"]


@dataclass(frozen=True)
class HybridConfig:
    """Fusion parameters. Part of the versioned experiment config."""

    #: Rank-flattening constant. 60 is the value from the original RRF paper and
    #: what configs/workflows/mvp_actions.yaml declares.
    rrf_k: int = 60
    #: How many candidates to request from each component before fusing. Wider
    #: than the final k, so a chunk ranked 20th by one retriever and 2nd by the
    #: other can still surface.
    candidate_k: int = 50
    #: Per-retriever weights, keyed by component name. Equal by default; an
    #: unequal weighting is a claim about which retriever to trust and should be
    #: justified on validation data rather than assumed.
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")


def reciprocal_rank_fusion(
    rankings: dict[str, list[RetrievedItem]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    k: int = 10,
) -> list[RetrievedItem]:
    """Fuse several ranked lists into one.

    The returned items carry ``component_ranks`` — where each source placed the
    chunk — so a fused result can be traced back to what produced it. Without
    that, a hybrid result is unattributable and the question "did fusion help,
    and via which retriever" becomes unanswerable.
    """
    if not rankings:
        return []
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    items: dict[str, RetrievedItem] = {}

    for source, ranked in rankings.items():
        weight = weights.get(source, 1.0)
        for item in ranked:
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (rrf_k + item.rank)
            ranks.setdefault(item.chunk_id, {})[source] = item.rank
            # Keep the first-seen item for its text; sources iterate in a fixed
            # order so this is deterministic.
            items.setdefault(item.chunk_id, item)

    # Ties break on the chunk id, so fusion is reproducible even when two chunks
    # accumulate identical scores — which happens whenever both are found at the
    # same rank by the same set of retrievers.
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))

    fused: list[RetrievedItem] = []
    for rank, chunk_id in enumerate(order[:k], start=1):
        source_item = items[chunk_id]
        fused.append(
            RetrievedItem(
                chunk_id=chunk_id,
                document_id=source_item.document_id,
                rank=rank,
                text=source_item.text,
                score=scores[chunk_id],
                component_ranks=ranks[chunk_id],
            )
        )
    return fused


@dataclass
class HybridRetriever:
    """Fuses several component retrievers by reciprocal rank fusion."""

    components: dict[str, object]
    config: HybridConfig = field(default_factory=HybridConfig)

    @property
    def name(self) -> str:
        return f"hybrid_rrf(k={self.config.rrf_k}, components={sorted(self.components)})"

    def search(self, query: str, k: int = 10) -> list[RetrievedItem]:
        """Retrieve from every component, then fuse."""
        rankings: dict[str, list[RetrievedItem]] = {}
        # Sorted so component order — and therefore tie-breaking — does not
        # depend on dict insertion order.
        for source in sorted(self.components):
            retriever = self.components[source]
            rankings[source] = retriever.search(query, k=self.config.candidate_k)  # type: ignore[attr-defined]

        return reciprocal_rank_fusion(
            rankings, rrf_k=self.config.rrf_k, weights=self.config.weights, k=k
        )

    def stats(self) -> dict[str, object]:
        return {
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": self.config.rrf_k,
            "candidate_k": self.config.candidate_k,
            "components": sorted(self.components),
            "weights": dict(self.config.weights),
        }
