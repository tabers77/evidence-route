"""Lexical, dense and hybrid retrieval (actions A1-A3).

All three retrievers expose one interface so they can be swapped inside the
workflow layer and compared on identical questions. Retrieval is scored
separately from generation: a correct answer can conceal poor retrieval, and a
poor answer can hide successful evidence retrieval (spec section 11.1).

Hybrid fusion must be explicit. Describing a system as "hybrid" without stating
how rankings are combined is not a documented method — the MVP uses reciprocal
rank fusion, and the fusion constant is part of the experiment config.

Modules:
    ``tokenization``  number-aware tokenizer; the reason A1 can match figures
    ``base``          the shared retriever protocol
    ``bm25``          Okapi BM25 lexical retrieval (implemented)
    ``dense``         embedding retrieval over a local index (planned)
    ``hybrid``        reciprocal rank fusion over the two above (planned)
"""

from evidence_route.retrieval.base import Retriever, chunk_to_item
from evidence_route.retrieval.bm25 import BM25Config, BM25Retriever
from evidence_route.retrieval.tokenization import (
    STOPWORDS,
    has_digits,
    normalize_number,
    tokenize,
)

__all__ = [
    "STOPWORDS",
    "BM25Config",
    "BM25Retriever",
    "Retriever",
    "chunk_to_item",
    "has_digits",
    "normalize_number",
    "tokenize",
]
