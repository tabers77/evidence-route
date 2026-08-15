"""Document parsers.

Parsing quality bounds retrieval quality. If a financial table is flattened into
prose during parsing, no retriever can recover the figure, and the resulting
failure looks like a retrieval failure in the taxonomy when it is really a
parsing one. That misattribution is worth guarding against explicitly, which is
why :func:`validate_parsed_document` exists and why table counts are reported.

Two backends:

``parse_text_document``
    Plain text with an explicit page delimiter. No dependencies, fully
    deterministic, and what the offline smoke path and the test suite use.

``parse_pdf_document``
    pdfplumber-backed, behind the optional ``documents`` extra. Imported lazily
    so the core package installs and tests without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_route.documents.models import ParsedDocument, ParsedPage, Table

__all__ = [
    "PAGE_DELIMITER",
    "ParsingReport",
    "parse_pdf_document",
    "parse_text_document",
    "validate_parsed_document",
]

#: Page separator used by the plain-text backend and the fixture corpus.
PAGE_DELIMITER = "\n<<<PAGE>>>\n"


def parse_text_document(
    path: Path,
    *,
    document_id: str | None = None,
    first_page_number: int = 1,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """Parse a plain-text document whose pages are delimited by ``PAGE_DELIMITER``.

    Page numbers start at ``first_page_number`` (1 by default) so they line up
    with how FinanceBench cites evidence — a 0-based offset here would shift
    every evidence match by one page, which is exactly the sort of silent
    off-by-one that would depress retrieval recall for reasons unrelated to
    retrieval.
    """
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    raw = path.read_text(encoding="utf-8")
    pages = tuple(
        ParsedPage(page_number=first_page_number + i, text=block.strip())
        for i, block in enumerate(raw.split(PAGE_DELIMITER))
    )
    return ParsedDocument(
        document_id=document_id or path.stem,
        pages=pages,
        metadata={"source_path": str(path), "backend": "text", **(metadata or {})},
    )


def parse_pdf_document(
    path: Path,
    *,
    document_id: str | None = None,
    extract_tables: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """Parse a PDF with pdfplumber, keeping tables structured.

    Requires the ``documents`` extra. The import is deliberately lazy and the
    error deliberately actionable, because the core install does not carry PDF
    dependencies and hitting an ImportError deep in a batch run is a poor way to
    discover that.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "PDF parsing requires the 'documents' extra. "
            'Install it with: pip install -e ".[documents]"'
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    pages: list[ParsedPage] = []
    with pdfplumber.open(path) as pdf:  # pragma: no cover - needs a real PDF
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables: list[Table] = []
            if extract_tables:
                for raw_table in page.extract_tables() or []:
                    rows = tuple(tuple((cell or "").strip() for cell in row) for row in raw_table)
                    if rows:
                        tables.append(Table(rows=rows, page_number=index))
            pages.append(ParsedPage(page_number=index, text=text.strip(), tables=tuple(tables)))

    return ParsedDocument(
        document_id=document_id or path.stem,
        pages=tuple(pages),
        metadata={"source_path": str(path), "backend": "pdfplumber", **(metadata or {})},
    )


@dataclass
class ParsingReport:
    """Health check on a parsed document."""

    document_id: str
    n_pages: int
    n_tables: int
    empty_pages: list[int]
    total_characters: int
    warnings: list[str]

    @property
    def is_healthy(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        lines = [
            f"{self.document_id}: {self.n_pages} pages, {self.n_tables} tables, "
            f"{self.total_characters} chars"
        ]
        lines.extend(f"  WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_parsed_document(
    document: ParsedDocument,
    *,
    max_empty_page_fraction: float = 0.1,
    expect_tables: bool = True,
) -> ParsingReport:
    """Check a parsed document for the failure modes that masquerade as others.

    A scanned page yields no text; a table-extraction failure yields prose with
    the numbers missing. Both look like retrieval failures downstream unless
    they are caught here (spec section 16).
    """
    empty = document.empty_pages()
    total_chars = sum(len(p.text) for p in document.pages)
    warnings: list[str] = []

    if document.n_pages == 0:
        warnings.append("document parsed to zero pages")

    if document.n_pages and len(empty) / document.n_pages > max_empty_page_fraction:
        warnings.append(
            f"{len(empty)}/{document.n_pages} pages are empty "
            f"(> {max_empty_page_fraction:.0%}); the source may be scanned or "
            f"image-based, which would make its content unretrievable"
        )

    if expect_tables and document.n_tables == 0 and document.n_pages > 0:
        warnings.append(
            "no tables extracted; most FinanceBench numerical answers live in "
            "tables, so this may indicate a table-extraction failure rather "
            "than a document without tables"
        )

    return ParsingReport(
        document_id=document.document_id,
        n_pages=document.n_pages,
        n_tables=document.n_tables,
        empty_pages=empty,
        total_characters=total_chars,
        warnings=warnings,
    )
