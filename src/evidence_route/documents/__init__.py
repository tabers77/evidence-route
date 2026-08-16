"""Document parsing and chunking.

FinanceBench is a PDF corpus, and financial PDFs are hostile to naive parsers:
the answer to a numerical question usually lives in a table, and a parser that
flattens tables into prose destroys exactly the content the benchmark tests.
Parsing quality therefore bounds retrieval quality, and a retrieval failure
caused by a parsing failure must be attributed correctly in the failure taxonomy
(spec section 16, "table or numerical content lost during parsing").

Modules:
    ``models``    ParsedDocument, ParsedPage, Table, Chunk
    ``parsing``   text and (optional) pdfplumber backends, plus parse validation
    ``chunking``  stable content-derived chunk ids with page provenance
    ``evidence``  resolving page-level evidence references onto chunks
"""

from evidence_route.documents.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_id_for,
    make_chunk_id,
)
from evidence_route.documents.corpus import Corpus, CorpusProvenance, require_reportable
from evidence_route.documents.evidence import (
    EvidenceResolution,
    parse_evidence_reference,
    resolve_evidence,
)
from evidence_route.documents.evidence_corpus import (
    build_evidence_page_corpus,
    parsed_documents_from_evidence,
)
from evidence_route.documents.models import (
    Chunk,
    ParsedDocument,
    ParsedPage,
    Table,
    estimate_tokens,
)
from evidence_route.documents.parsing import (
    PAGE_DELIMITER,
    ParsingReport,
    parse_pdf_document,
    parse_text_document,
    validate_parsed_document,
)

__all__ = [
    "PAGE_DELIMITER",
    "Chunk",
    "ChunkingConfig",
    "Corpus",
    "CorpusProvenance",
    "EvidenceResolution",
    "ParsedDocument",
    "ParsedPage",
    "ParsingReport",
    "Table",
    "build_evidence_page_corpus",
    "chunk_document",
    "chunk_id_for",
    "estimate_tokens",
    "make_chunk_id",
    "parse_evidence_reference",
    "parse_pdf_document",
    "parse_text_document",
    "parsed_documents_from_evidence",
    "require_reportable",
    "resolve_evidence",
    "validate_parsed_document",
]
