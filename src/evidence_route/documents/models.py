"""Document and chunk models.

Page provenance is carried all the way from the parser into every chunk, and
that is the load-bearing design decision in this package.

FinanceBench states its evidence as *document plus page number*. Retrieval,
however, returns chunks. If a chunk does not know which pages it came from,
there is no way to ask "did retrieval find the evidence?" — and retrieval recall
becomes unmeasurable, which would collapse the separation between retrieval
quality and answer quality that spec section 11.1 depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Chunk",
    "ParsedDocument",
    "ParsedPage",
    "Table",
    "estimate_tokens",
]


def estimate_tokens(text: str) -> int:
    """Approximate token count without pulling in a tokenizer.

    Roughly four characters per token, the usual rule of thumb for English
    prose. It is an *approximation*, and it is deliberately labelled as one:
    financial tables tokenize far more densely than prose because digits and
    separators split aggressively, so this understates table-heavy content.

    That is acceptable for chunk sizing — which only needs to be consistent, not
    exact — but it must never be used for cost estimation. Cost comes from the
    provider's reported usage, which the outcome record stores directly.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Table:
    """A table extracted from a page.

    Kept structured rather than flattened into prose. Most FinanceBench
    numerical answers live inside tables, and a parser that renders a balance
    sheet as a run-on sentence destroys precisely the content the benchmark
    tests (spec section 16, "table or numerical content lost during parsing").
    """

    rows: tuple[tuple[str, ...], ...]
    page_number: int
    caption: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_text(self) -> str:
        """Render as pipe-delimited text for indexing.

        Row and column structure is preserved as delimiters so that a lexical
        retriever can still match "Total revenues | 66,608" as adjacent terms.
        """
        return "\n".join(" | ".join(cell.strip() for cell in row) for row in self.rows)


@dataclass(frozen=True)
class ParsedPage:
    """One page of a parsed document."""

    page_number: int
    text: str
    tables: tuple[Table, ...] = ()

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or bool(self.tables)


@dataclass(frozen=True)
class ParsedDocument:
    """A source document after parsing, before chunking."""

    document_id: str
    pages: tuple[ParsedPage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_tables(self) -> int:
        return sum(len(page.tables) for page in self.pages)

    def page(self, page_number: int) -> ParsedPage | None:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def empty_pages(self) -> list[int]:
        """Pages that produced no content.

        Reported rather than ignored: a run of empty pages usually means the
        parser failed on a scanned or image-based section, and that is a
        retrieval failure waiting to be misattributed to the retriever.
        """
        return [p.page_number for p in self.pages if not p.has_content]


@dataclass(frozen=True)
class Chunk:
    """An indexable unit of text with its page provenance."""

    chunk_id: str
    document_id: str
    text: str
    #: Every page this chunk draws from. A chunk may span a page boundary, so
    #: this is a tuple rather than a single number — collapsing it to one page
    #: would silently drop evidence matches at boundaries.
    page_numbers: tuple[int, ...]
    #: Position within the document, 0-based. Used to fetch neighbouring chunks
    #: for the agentic workflow's read_neighbouring_chunks tool.
    ordinal: int
    token_estimate: int
    contains_table: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def first_page(self) -> int:
        return self.page_numbers[0]

    def covers_page(self, page_number: int) -> bool:
        return page_number in self.page_numbers
