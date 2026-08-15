"""Okapi BM25 lexical retrieval (action A1).

Implemented directly rather than pulled from a library. The spec permits
"rank-bm25, Pyserini or an equivalent documented choice" (section 17), and a
documented implementation earns three things a wrapper would not:

1. **Explicit tie-breaking.** Financial corpora produce exact score ties
   constantly — repeated boilerplate, near-identical table rows. A library that
   leaves ties to sort order makes retrieval non-reproducible in precisely the
   cases this benchmark is full of. Ties here break on chunk ordinal, which is
   stable across runs and across machines.
2. **A non-negative IDF.** The textbook IDF goes negative for terms appearing in
   more than half the corpus, letting a common term *reduce* a document's score
   below zero. On a single-company filing set, terms like the company name are
   exactly that common.
3. **No dependency on the core retrieval path**, so the offline smoke experiment
   and CI run BM25 without installing extras.

Swapping in a library later is a one-class change behind the ``Retriever``
protocol, and the tests here would validate the replacement unchanged.

The scoring function, for the record:

    score(D, Q) = sum over q of
        IDF(q) * ( f(q,D) * (k1 + 1) ) / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )

where f(q,D) is the term frequency of q in chunk D, |D| the chunk length in
tokens, avgdl the mean chunk length, N the number of chunks and n(q) the number
of chunks containing q.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from evidence_route.documents.models import Chunk
from evidence_route.retrieval.base import chunk_to_item
from evidence_route.retrieval.tokenization import tokenize
from evidence_route.storage.records import RetrievedItem

__all__ = ["BM25Config", "BM25Retriever"]


@dataclass(frozen=True)
class BM25Config:
    """BM25 parameters. Part of the versioned experiment configuration."""

    #: Term-frequency saturation. Higher means term frequency keeps mattering;
    #: 1.2 is the standard default and what configs/workflows/mvp_actions.yaml
    #: declares.
    k1: float = 1.2
    #: Length normalization, 0 (off) to 1 (full). 0.75 is standard.
    b: float = 0.75
    expand_numbers: bool = True
    remove_stopwords: bool = True

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 must be non-negative.")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b must lie in [0, 1], got {self.b}.")


@dataclass
class BM25Retriever:
    """An in-memory BM25 index over a chunk corpus.

    In-memory is a deliberate MVP choice: the FinanceBench sample is small, and
    the simplest infrastructure that supports reproducible experiments is worth
    more than a managed service here (spec section 17).
    """

    config: BM25Config = field(default_factory=BM25Config)

    # Populated by index().
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _doc_tokens: list[list[str]] = field(default_factory=list, repr=False)
    _doc_freqs: list[Counter[str]] = field(default_factory=list, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, repr=False)
    _document_frequency: Counter[str] = field(default_factory=Counter, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avg_length: float = 0.0

    @property
    def name(self) -> str:
        return f"bm25(k1={self.config.k1},b={self.config.b})"

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    @property
    def vocabulary_size(self) -> int:
        return len(self._document_frequency)

    def index(self, chunks: list[Chunk]) -> BM25Retriever:
        """Build the index. Returns self so construction can be chained."""
        self._chunks = list(chunks)
        self._doc_tokens = [
            tokenize(
                chunk.text,
                expand_numbers=self.config.expand_numbers,
                remove_stopwords=self.config.remove_stopwords,
            )
            for chunk in self._chunks
        ]
        self._doc_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )

        self._document_frequency = Counter()
        for freqs in self._doc_freqs:
            self._document_frequency.update(freqs.keys())

        n_docs = len(self._chunks)
        # Precomputed because IDF depends only on the corpus, not the query.
        self._idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._document_frequency.items()
        }
        return self

    def score(self, query: str) -> list[float]:
        """BM25 score of every indexed chunk against ``query``."""
        query_terms = tokenize(
            query,
            expand_numbers=self.config.expand_numbers,
            remove_stopwords=self.config.remove_stopwords,
        )
        scores = [0.0] * len(self._chunks)
        if not query_terms or not self._chunks:
            return scores

        k1 = self.config.k1
        b = self.config.b
        avgdl = self._avg_length or 1.0

        # Duplicated query terms genuinely count more than once in BM25, so the
        # raw term list is used rather than a set.
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:  # term absent from the corpus contributes nothing
                continue
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                norm = 1.0 - b + b * (self._doc_lengths[i] / avgdl)
                scores[i] += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)

        return scores

    def search(self, query: str, k: int = 10) -> list[RetrievedItem]:
        """Return the top ``k`` chunks, rank 1 first.

        Ties break on chunk ordinal, ascending. Without an explicit rule, equal
        scores would order by whatever the sort happened to do, and repeated
        boilerplate in filings makes exact ties common — retrieval would stop
        being reproducible exactly where this corpus is densest.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if not self._chunks:
            return []

        scores = self.score(query)
        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-scores[i], self._chunks[i].ordinal, self._chunks[i].chunk_id),
        )

        results: list[RetrievedItem] = []
        for rank, index in enumerate(order[:k], start=1):
            # A zero score means no query term matched; returning such a chunk
            # would pad the result list with arbitrary text and inflate the
            # apparent context the generator was given.
            if scores[index] <= 0.0:
                break
            results.append(chunk_to_item(self._chunks[index], rank=rank, score=scores[index]))
        return results

    def stats(self) -> dict[str, float | int]:
        """Index statistics, recorded alongside experiment results."""
        return {
            "n_chunks": self.n_chunks,
            "vocabulary_size": self.vocabulary_size,
            "avg_chunk_length_tokens": round(self._avg_length, 2),
            "k1": self.config.k1,
            "b": self.config.b,
        }
