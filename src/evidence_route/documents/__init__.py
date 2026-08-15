"""Document parsing and chunking.

FinanceBench is a PDF corpus, and financial PDFs are hostile to naive parsers:
the answer to a numerical question usually lives in a table, and a parser that
flattens tables into prose destroys exactly the content the benchmark tests.
Parsing quality therefore bounds retrieval quality, and a retrieval failure
caused by a parsing failure must be attributed correctly in the failure taxonomy
(spec section 16, "table or numerical content lost during parsing").

Planned modules:
    ``parse``      PDF → structured text with table handling
    ``chunk``      chunking strategy with stable, content-derived chunk ids
    ``validate``   parsing checks, including table-specific assertions
"""
