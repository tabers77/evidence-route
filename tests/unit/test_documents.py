"""Tests for parsing, parse validation and evidence resolution."""

from __future__ import annotations

import pytest

from evidence_route.documents.chunking import ChunkingConfig, chunk_document
from evidence_route.documents.evidence import (
    parse_evidence_reference,
    resolve_evidence,
)
from evidence_route.documents.models import Chunk, ParsedDocument, ParsedPage, Table
from evidence_route.documents.parsing import (
    PAGE_DELIMITER,
    parse_text_document,
    validate_parsed_document,
)


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------
def test_parses_delimited_pages(tmp_path):
    path = tmp_path / "ACME_2023_10K.txt"
    path.write_text(
        PAGE_DELIMITER.join(["First page.", "Second page.", "Third page."]),
        encoding="utf-8",
    )
    doc = parse_text_document(path)

    assert doc.document_id == "ACME_2023_10K"
    assert doc.n_pages == 3
    assert doc.pages[0].text == "First page."


def test_page_numbers_start_at_one(tmp_path):
    """FinanceBench cites 1-based pages.

    A 0-based offset here would shift every evidence match by a page and
    depress retrieval recall for reasons unrelated to retrieval.
    """
    path = tmp_path / "d.txt"
    path.write_text(PAGE_DELIMITER.join(["a", "b"]), encoding="utf-8")
    doc = parse_text_document(path)
    assert [p.page_number for p in doc.pages] == [1, 2]


def test_document_id_can_be_overridden(tmp_path):
    path = tmp_path / "messy-filename.txt"
    path.write_text("content", encoding="utf-8")
    assert parse_text_document(path, document_id="CLEAN_ID").document_id == "CLEAN_ID"


def test_missing_document_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_text_document(tmp_path / "absent.txt")


def test_page_lookup(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text(PAGE_DELIMITER.join(["a", "b"]), encoding="utf-8")
    doc = parse_text_document(path)
    assert doc.page(2) is not None
    assert doc.page(2).text == "b"
    assert doc.page(99) is None


# ---------------------------------------------------------------------------
# Parse validation
# ---------------------------------------------------------------------------
def _doc_with_tables() -> ParsedDocument:
    table = Table(rows=(("Revenue", "100"),), page_number=1)
    return ParsedDocument(
        document_id="D",
        pages=(
            ParsedPage(1, "Prose here.", tables=(table,)),
            ParsedPage(2, "More prose."),
        ),
    )


def test_healthy_document_has_no_warnings():
    report = validate_parsed_document(_doc_with_tables())
    assert report.is_healthy
    assert report.n_tables == 1
    assert report.n_pages == 2


def test_mostly_empty_document_is_flagged():
    """A run of empty pages usually means a scanned or image-based source,
    which would otherwise be misattributed as a retrieval failure."""
    doc = ParsedDocument(
        document_id="D",
        pages=tuple(ParsedPage(i, "") for i in range(1, 6)),
    )
    report = validate_parsed_document(doc, expect_tables=False)
    assert not report.is_healthy
    assert any("empty" in w for w in report.warnings)
    assert report.empty_pages == [1, 2, 3, 4, 5]


def test_absent_tables_are_flagged_when_expected():
    doc = ParsedDocument(document_id="D", pages=(ParsedPage(1, "prose only"),))
    report = validate_parsed_document(doc, expect_tables=True)
    assert any("no tables extracted" in w for w in report.warnings)


def test_table_expectation_can_be_disabled():
    doc = ParsedDocument(document_id="D", pages=(ParsedPage(1, "prose only"),))
    assert validate_parsed_document(doc, expect_tables=False).is_healthy


def test_zero_page_document_is_flagged():
    report = validate_parsed_document(ParsedDocument(document_id="D", pages=()))
    assert any("zero pages" in w for w in report.warnings)


def test_report_summary_mentions_warnings():
    doc = ParsedDocument(document_id="D", pages=(ParsedPage(1, "prose"),))
    summary = validate_parsed_document(doc).summary()
    assert "WARNING" in summary


# ---------------------------------------------------------------------------
# Evidence reference parsing
# ---------------------------------------------------------------------------
def test_parses_a_reference():
    assert parse_evidence_reference("ACME_10K::p60") == ("ACME_10K", 60)


def test_parses_unknown_page():
    assert parse_evidence_reference("ACME_10K::p?") == ("ACME_10K", None)


def test_document_names_containing_colons_still_parse():
    assert parse_evidence_reference("A::B_10K::p3") == ("A::B_10K", 3)


@pytest.mark.parametrize("bad", ["no-separator", "DOC::page5", "DOC::p", "DOC::pX"])
def test_malformed_reference_raises(bad: str):
    """Guessing would produce an empty gold set and make retrieval look worse."""
    with pytest.raises(ValueError, match="Malformed evidence reference"):
        parse_evidence_reference(bad)


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------
def _corpus() -> list[Chunk]:
    doc = ParsedDocument(
        document_id="ACME_10K",
        pages=tuple(
            ParsedPage(i, "\n\n".join(f"page {i} paragraph {j} " + "x " * 60 for j in range(3)))
            for i in range(1, 6)
        ),
    )
    return chunk_document(doc, ChunkingConfig(chunk_size_tokens=120, chunk_overlap_tokens=0))


def test_resolves_a_page_reference_to_covering_chunks():
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p3"], chunks)

    assert resolution.is_fully_resolved
    matched = resolution.chunks_by_reference["ACME_10K::p3"]
    assert matched
    for chunk_id in matched:
        chunk = next(c for c in chunks if c.chunk_id == chunk_id)
        assert chunk.covers_page(3)


def test_unknown_document_is_reported_not_silently_empty():
    resolution = resolve_evidence(["OTHER_10K::p1"], _corpus())
    assert resolution.unknown_documents == ["OTHER_10K::p1"]
    assert not resolution.is_fully_resolved
    assert resolution.gold_chunk_ids == set()


def test_page_beyond_the_document_is_unresolved():
    resolution = resolve_evidence(["ACME_10K::p999"], _corpus())
    assert resolution.unresolved == ["ACME_10K::p999"]


def test_unknown_page_falls_back_to_the_whole_document_and_is_recorded():
    """Recorded separately so these can be excluded from precision-sensitive
    analyses rather than quietly widening the gold set."""
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p?"], chunks)
    assert resolution.unknown_pages == ["ACME_10K::p?"]
    assert len(resolution.gold_chunk_ids) == len(chunks)


def test_malformed_reference_is_collected_not_raised():
    resolution = resolve_evidence(["garbage"], _corpus())
    assert resolution.unresolved == ["garbage"]


def test_multiple_references_union_into_the_gold_set():
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p1", "ACME_10K::p4"], chunks)
    assert len(resolution.chunks_by_reference) == 2
    assert resolution.gold_chunk_ids


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------
def test_recall_is_one_when_every_gold_chunk_is_retrieved():
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p2"], chunks)
    gold = sorted(resolution.gold_chunk_ids)
    assert resolution.recall_at_k(gold, k=10) == 1.0


def test_recall_is_zero_when_nothing_relevant_is_retrieved():
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p2"], chunks)
    irrelevant = [c.chunk_id for c in chunks if c.chunk_id not in resolution.gold_chunk_ids]
    assert resolution.recall_at_k(irrelevant, k=10) == 0.0


def test_recall_respects_the_cutoff():
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p1", "ACME_10K::p5"], chunks)
    gold = sorted(resolution.gold_chunk_ids)
    assert len(gold) >= 2
    assert resolution.recall_at_k(gold, k=1) < resolution.recall_at_k(gold, k=len(gold))


def test_unresolvable_evidence_scores_zero_not_one():
    """An unmeasurable question must depress the metric visibly.

    Returning 1.0 for an empty gold set would silently inflate retrieval recall
    with exactly the questions whose evidence could not be located.
    """
    resolution = resolve_evidence(["OTHER::p1"], _corpus())
    assert resolution.recall_at_k(["anything"], k=10) == 0.0
    assert resolution.hit_at_k(["anything"], k=10) is False


def test_hit_at_k_is_true_for_a_single_match():
    chunks = _corpus()
    resolution = resolve_evidence(["ACME_10K::p3"], chunks)
    one_gold = [sorted(resolution.gold_chunk_ids)[0]]
    assert resolution.hit_at_k(one_gold, k=1) is True
