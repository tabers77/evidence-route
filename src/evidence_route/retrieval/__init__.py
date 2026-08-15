"""Lexical, dense and hybrid retrieval (actions A1-A3).

All three retrievers expose one interface so they can be swapped inside the
workflow layer and compared on identical questions. Retrieval is scored
separately from generation: a correct answer can conceal poor retrieval, and a
poor answer can hide successful evidence retrieval (spec section 11.1).

Hybrid fusion must be explicit. Describing a system as "hybrid" without stating
how rankings are combined is not a documented method — the MVP uses reciprocal
rank fusion, and the fusion constant is part of the experiment config.

Planned modules:
    ``base``     the shared retriever protocol and ``RetrievedItem`` production
    ``bm25``     lexical retrieval
    ``dense``    embedding retrieval over a local index
    ``hybrid``   reciprocal rank fusion over the two above
    ``index``    index building, persistence and versioning
"""
