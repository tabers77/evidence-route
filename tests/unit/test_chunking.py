"""Tests for chunking, chunk identity and page provenance."""

from __future__ import annotations

from itertools import pairwise

import pytest

from evidence_route.documents.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_id_for,
    make_chunk_id,
)
from evidence_route.documents.models import (
    ParsedDocument,
    ParsedPage,
    Table,
    estimate_tokens,
)


def _page(number: int, paragraphs: int = 3, words: int = 40) -> ParsedPage:
    text = "\n\n".join(
        " ".join(f"page{number}para{p}word{w}" for w in range(words)) for p in range(paragraphs)
    )
    return ParsedPage(page_number=number, text=text)


def _document(n_pages: int = 4, **kwargs) -> ParsedDocument:
    return ParsedDocument(
        document_id="ACME_2023_10K",
        pages=tuple(_page(i, **kwargs) for i in range(1, n_pages + 1)),
    )


# ---------------------------------------------------------------------------
# Identifier stability
# ---------------------------------------------------------------------------
def test_chunking_is_reproducible():
    """Same document, same settings, same identifiers.

    Otherwise every stored evidence label and retrieval result silently stops
    pointing at the same text.
    """
    doc = _document()
    a = chunk_document(doc)
    b = chunk_document(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_chunk_ids_are_derived_from_content():
    doc = _document()
    for chunk in chunk_document(doc):
        assert chunk_id_for(chunk) == chunk.chunk_id


def test_identical_text_at_different_positions_gets_distinct_ids():
    """Filings repeat boilerplate verbatim.

    Hashing content alone would collide on repeated headers and merge two
    distinct locations into one identifier.
    """
    boilerplate = "See accompanying notes to consolidated financial statements."
    doc = ParsedDocument(
        document_id="D",
        pages=(
            ParsedPage(page_number=1, text=boilerplate),
            ParsedPage(page_number=2, text=boilerplate),
        ),
    )
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=8, chunk_overlap_tokens=0))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "identical boilerplate produced colliding ids"


def test_same_text_in_different_documents_gets_distinct_ids():
    assert make_chunk_id("DOC_A", 0, "shared") != make_chunk_id("DOC_B", 0, "shared")


def test_id_ignores_whitespace_differences():
    """Re-parsing may alter incidental whitespace; that must not change the id."""
    assert make_chunk_id("D", 0, "a  b\n c") == make_chunk_id("D", 0, "a b c")


def test_chunk_id_carries_readable_provenance():
    chunk_id = make_chunk_id("ACME_2023_10K", 7, "text")
    assert chunk_id.startswith("ACME_2023_10K::c0007::")


# ---------------------------------------------------------------------------
# Page provenance
# ---------------------------------------------------------------------------
def test_every_chunk_records_its_pages():
    chunks = chunk_document(_document())
    assert chunks
    for chunk in chunks:
        assert chunk.page_numbers
        assert all(isinstance(p, int) for p in chunk.page_numbers)


def test_chunk_spanning_a_boundary_records_both_pages():
    """Collapsing provenance to one page would drop evidence at boundaries."""
    doc = _document(n_pages=4, paragraphs=1, words=10)
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=0))
    # Small pages all fit in one chunk, which must therefore claim every page.
    assert len(chunks) == 1
    assert chunks[0].page_numbers == (1, 2, 3, 4)


def test_page_numbers_are_sorted_and_unique():
    for chunk in chunk_document(_document()):
        assert list(chunk.page_numbers) == sorted(set(chunk.page_numbers))


def test_all_pages_are_represented_somewhere():
    doc = _document(n_pages=6)
    covered = {p for c in chunk_document(doc) for p in c.page_numbers}
    assert covered == {1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Sizing and overlap
# ---------------------------------------------------------------------------
def test_chunks_respect_the_token_budget():
    config = ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=10)
    for chunk in chunk_document(_document(n_pages=5), config):
        # The budget is applied to packing; a single oversized segment is hard
        # split beforehand, so nothing should greatly exceed it.
        assert chunk.token_estimate <= config.chunk_size_tokens * 2


def _many_small_paragraphs(n_paragraphs: int = 30) -> ParsedDocument:
    """A document of many short paragraphs, so several fit per chunk."""
    pages = tuple(
        ParsedPage(
            page_number=i + 1,
            text="\n\n".join(f"p{i}q{j} alpha beta gamma delta" for j in range(5)),
        )
        for i in range(n_paragraphs // 5)
    )
    return ParsedDocument(document_id="SMALL", pages=pages)


def test_overlap_produces_shared_content():
    config = ChunkingConfig(chunk_size_tokens=40, chunk_overlap_tokens=12)
    chunks = chunk_document(_many_small_paragraphs(), config)
    assert len(chunks) > 1
    overlapping = sum(1 for a, b in pairwise(chunks) if set(a.text.split()) & set(b.text.split()))
    assert overlapping > 0


def test_single_segment_chunks_cannot_overlap():
    """A deliberate limit, not a bug.

    When one segment already fills the whole budget, carrying it into the next
    chunk would mean the next chunk starts full and immediately flushes — the
    packer would never advance. Overlap is skipped instead of looping forever.
    """
    config = ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=10)
    # Each paragraph on its own exceeds the budget, so every chunk is one segment.
    doc = _document(n_pages=3, paragraphs=2, words=40)
    chunks = chunk_document(doc, config)
    assert len(chunks) > 1
    for a, b in pairwise(chunks):
        assert a.text != b.text


def test_zero_overlap_is_allowed():
    config = ChunkingConfig(chunk_size_tokens=60, chunk_overlap_tokens=0)
    chunks = chunk_document(_document(n_pages=3), config)
    assert len(chunks) > 1


def test_oversized_paragraph_is_split_rather_than_emitted_whole():
    huge = " ".join(f"word{i}" for i in range(5000))
    doc = ParsedDocument(document_id="D", pages=(ParsedPage(1, huge),))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=0))
    assert len(chunks) > 1


def test_ordinals_are_contiguous_from_zero():
    chunks = chunk_document(_document(n_pages=5))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_empty_document_yields_no_chunks():
    doc = ParsedDocument(document_id="D", pages=(ParsedPage(1, "   "),))
    assert chunk_document(doc) == []


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def test_table_content_is_chunked_and_flagged():
    table = Table(rows=(("Line item", "2022"), ("Total revenues", "66,608")), page_number=2)
    doc = ParsedDocument(
        document_id="D",
        pages=(
            ParsedPage(1, "Some prose."),
            ParsedPage(2, "More prose.", tables=(table,)),
        ),
    )
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.contains_table]
    assert table_chunks
    assert "Total revenues | 66,608" in "\n".join(c.text for c in table_chunks)


def test_table_rendering_preserves_row_structure():
    table = Table(rows=(("a", "b"), ("c", "d")), page_number=1)
    assert table.to_text() == "a | b\nc | d"
    assert table.n_rows == 2
    assert table.n_cols == 2


def test_table_keeps_its_page_number():
    table = Table(rows=(("Revenue", "100"),), page_number=42)
    doc = ParsedDocument(document_id="D", pages=(ParsedPage(42, "prose", tables=(table,)),))
    chunks = chunk_document(doc)
    assert all(42 in c.page_numbers for c in chunks)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_overlap_must_be_smaller_than_chunk_size():
    """Equal overlap and size cannot advance — it would loop forever."""
    with pytest.raises(ValueError, match="must be smaller"):
        ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        ChunkingConfig(chunk_size_tokens=0)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        ChunkingConfig(chunk_overlap_tokens=-1)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
def test_token_estimate_scales_with_length():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_token_estimate_is_never_zero():
    """A zero estimate would let empty segments pack forever."""
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a") >= 1
