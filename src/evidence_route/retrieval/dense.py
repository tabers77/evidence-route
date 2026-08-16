"""Dense embedding retrieval (action A2).

Vectors are L2-normalized at embedding time, so cosine similarity reduces to a
dot product and the whole search is one matrix multiply. For a corpus of a few
hundred to a few thousand chunks that is faster than any index structure would
be, and it has no approximation error — which matters here, because an
approximate index would add a second source of retrieval difference on top of
the lexical-versus-semantic one this action exists to measure.

FAISS slots in behind the same interface when the corpus outgrows this. That
change would be a performance decision, not a research one, and the tests would
carry over unchanged.

**The embedding model is part of the index's identity.** An index built with one
model and queried with another produces vectors from different spaces whose
similarities are meaningless — and, crucially, meaningless in a way that still
returns confident-looking rankings. The retriever records its model version and
refuses the mismatch rather than silently returning noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evidence_route.documents.models import Chunk
from evidence_route.retrieval.base import chunk_to_item
from evidence_route.retrieval.embeddings import Embedder, HashingEmbedder
from evidence_route.storage.records import RetrievedItem

__all__ = ["DenseConfig", "DenseRetriever"]


@dataclass(frozen=True)
class DenseConfig:
    """Dense retrieval parameters. Part of the versioned experiment config."""

    #: Similarity floor. Below this a chunk is not returned at all, rather than
    #: padding the result list with weak matches that inflate the apparent
    #: context the generator was given.
    min_similarity: float = 0.0
    index_type: str = "flat_inner_product"


@dataclass
class DenseRetriever:
    """In-memory dense retrieval over normalized embeddings."""

    embedder: Embedder = field(default_factory=HashingEmbedder)
    config: DenseConfig = field(default_factory=DenseConfig)

    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _matrix: np.ndarray | None = field(default=None, repr=False)
    _index_model_version: str | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"dense({self.embedder.model_version})"

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    @property
    def model_version(self) -> str:
        """The embedding model this index was built with."""
        return self._index_model_version or self.embedder.model_version

    def index(self, chunks: list[Chunk]) -> DenseRetriever:
        """Embed and store the corpus."""
        self._chunks = list(chunks)
        self._index_model_version = self.embedder.model_version

        if not self._chunks:
            self._matrix = None
            return self

        vectors = self.embedder.embed([chunk.text for chunk in self._chunks])
        if len(vectors) != len(self._chunks):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(self._chunks)} "
                f"chunks. A misaligned batch would attach every vector to the "
                f"wrong chunk."
            )
        self._matrix = np.asarray(vectors, dtype=np.float32)
        return self

    def _check_model_matches(self) -> None:
        current = self.embedder.model_version
        if self._index_model_version and current != self._index_model_version:
            raise ValueError(
                f"This index was built with embedding model "
                f"{self._index_model_version!r} but the embedder is now "
                f"{current!r}. Similarities between vectors from different models "
                f"are meaningless, and would still return a confident ranking. "
                f"Rebuild the index."
            )

    def score(self, query: str) -> np.ndarray:
        """Cosine similarity of every indexed chunk against the query."""
        self._check_model_matches()
        if self._matrix is None or not self._chunks:
            return np.zeros(0, dtype=np.float32)

        query_vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        # Both sides are unit-normalized, so the dot product is the cosine.
        return self._matrix @ query_vector

    def search(self, query: str, k: int = 10) -> list[RetrievedItem]:
        """Return the top ``k`` chunks, rank 1 first.

        Ties break on chunk ordinal then id, exactly as BM25 does. Using the same
        rule across retrievers matters: a difference in tie-breaking would show
        up in a paired comparison as a difference between retrieval methods.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if not self._chunks:
            return []

        scores = self.score(query)
        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-float(scores[i]), self._chunks[i].ordinal, self._chunks[i].chunk_id),
        )

        results: list[RetrievedItem] = []
        for rank, index in enumerate(order[:k], start=1):
            similarity = float(scores[index])
            if similarity <= self.config.min_similarity:
                break
            results.append(chunk_to_item(self._chunks[index], rank=rank, score=similarity))
        return results

    def stats(self) -> dict[str, object]:
        return {
            "n_chunks": self.n_chunks,
            "embedding_model": self.model_version,
            "is_semantic": self.embedder.is_semantic,
            "dimension": self.embedder.dimension,
            "index_type": self.config.index_type,
        }
