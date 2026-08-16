"""Tests for corpus provenance and the evidence-page development corpus."""

from __future__ import annotations

import json

import pytest

from evidence_route.documents.chunking import ChunkingConfig
from evidence_route.documents.corpus import Corpus, CorpusProvenance, require_reportable
from evidence_route.documents.evidence_corpus import (
    build_evidence_page_corpus,
    parsed_documents_from_evidence,
)
from evidence_route.documents.models import Chunk


def _chunk(ordinal: int, document_id: str = "D") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::c{ordinal:04d}",
        document_id=document_id,
        text="text",
        page_numbers=(1,),
        ordinal=ordinal,
        token_estimate=1,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_only_full_documents_are_reportable():
    assert CorpusProvenance.FULL_DOCUMENTS.is_reportable
    assert not CorpusProvenance.EVIDENCE_PAGES_ONLY.is_reportable
    assert not CorpusProvenance.FIXTURE.is_reportable


def test_every_provenance_explains_itself():
    """The rationale is what a reader sees when a run is refused."""
    for provenance in CorpusProvenance:
        assert len(provenance.rationale) > 40


def test_evidence_pages_rationale_names_the_specific_bias():
    text = CorpusProvenance.EVIDENCE_PAGES_ONLY.rationale.lower()
    assert "bm25" in text
    assert "distractor" in text


def test_require_reportable_rejects_the_dev_corpus():
    """Forgetting the caveat must fail loudly, not produce a plausible table."""
    corpus = Corpus("dev", [_chunk(0)], CorpusProvenance.EVIDENCE_PAGES_ONLY)
    with pytest.raises(ValueError, match="cannot support a reported result"):
        require_reportable(corpus)


def test_require_reportable_explains_why():
    corpus = Corpus("dev", [_chunk(0)], CorpusProvenance.EVIDENCE_PAGES_ONLY)
    with pytest.raises(ValueError, match="BM25"):
        require_reportable(corpus)


def test_require_reportable_accepts_full_documents():
    corpus = Corpus("real", [_chunk(0)], CorpusProvenance.FULL_DOCUMENTS)
    require_reportable(corpus)  # must not raise


def test_fixture_corpora_are_also_refused():
    """Fixtures exercise code paths; they measure nothing."""
    with pytest.raises(ValueError):
        require_reportable(Corpus("fx", [_chunk(0)], CorpusProvenance.FIXTURE))


# ---------------------------------------------------------------------------
# Corpus behaviour
# ---------------------------------------------------------------------------
def test_corpus_reports_its_documents():
    corpus = Corpus(
        "c",
        [_chunk(0, "A"), _chunk(1, "A"), _chunk(0, "B")],
        CorpusProvenance.FULL_DOCUMENTS,
    )
    assert len(corpus) == 3
    assert corpus.document_ids == {"A", "B"}


def test_chunks_can_be_scoped_to_specific_documents():
    """FinanceBench names its filing, so retrieval is scoped to it."""
    corpus = Corpus("c", [_chunk(0, "A"), _chunk(0, "B")], CorpusProvenance.FULL_DOCUMENTS)
    scoped = corpus.chunks_for({"A"})
    assert [c.document_id for c in scoped] == ["A"]


def test_stats_expose_reportability():
    corpus = Corpus("c", [_chunk(0)], CorpusProvenance.EVIDENCE_PAGES_ONLY)
    stats = corpus.stats()
    assert stats["reportable"] is False
    assert stats["provenance"] == "evidence_pages_only"


# ---------------------------------------------------------------------------
# Building from FinanceBench evidence
# ---------------------------------------------------------------------------
def _write_source(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _row(doc="ACME_2022_10K", page=5, text="Total revenues 66,608", qid="q1"):
    return {
        "financebench_id": qid,
        "company": "ACME",
        "doc_name": doc,
        "question": "revenue?",
        "answer": "66,608",
        "evidence": [
            {
                "doc_name": doc,
                "evidence_page_num": page,
                "evidence_text": text[:20],
                "evidence_text_full_page": text,
            }
        ],
    }


def test_reconstructs_documents_from_embedded_pages(tmp_path):
    source = _write_source(
        tmp_path / "fb.jsonl",
        [_row(page=5, qid="q1"), _row(page=9, qid="q2", text="Operating income 1,234")],
    )
    documents = parsed_documents_from_evidence(source)

    assert len(documents) == 1
    assert documents[0].document_id == "ACME_2022_10K"
    assert [p.page_number for p in documents[0].pages] == [5, 9]


def test_real_page_numbers_are_preserved(tmp_path):
    """Existing DOC::p<page> evidence references must stay valid."""
    source = _write_source(tmp_path / "fb.jsonl", [_row(page=60)])
    documents = parsed_documents_from_evidence(source)
    assert documents[0].pages[0].page_number == 60


def test_repeated_citations_of_one_page_are_deduplicated(tmp_path):
    """Indexing a page twice would let it occupy two retrieval slots."""
    source = _write_source(
        tmp_path / "fb.jsonl",
        [_row(page=5, qid="q1"), _row(page=5, qid="q2"), _row(page=5, qid="q3")],
    )
    documents = parsed_documents_from_evidence(source)
    assert len(documents[0].pages) == 1


def test_multiple_documents_are_separated(tmp_path):
    source = _write_source(
        tmp_path / "fb.jsonl",
        [_row(doc="A_2022_10K"), _row(doc="B_2021_10K", qid="q2")],
    )
    documents = parsed_documents_from_evidence(source)
    assert {d.document_id for d in documents} == {"A_2022_10K", "B_2021_10K"}


def test_entries_without_page_text_are_skipped(tmp_path):
    row = _row()
    row["evidence"][0]["evidence_text_full_page"] = None
    row["evidence"][0]["evidence_text"] = None
    source = _write_source(tmp_path / "fb.jsonl", [row])
    assert parsed_documents_from_evidence(source) == []


def test_malformed_lines_do_not_abort_the_build(tmp_path):
    path = tmp_path / "fb.jsonl"
    path.write_text(json.dumps(_row()) + "\n{not json\n", encoding="utf-8")
    assert len(parsed_documents_from_evidence(path)) == 1


def test_built_corpus_is_stamped_as_development_only(tmp_path):
    """The whole point: the stamp travels with the data."""
    source = _write_source(tmp_path / "fb.jsonl", [_row()])
    corpus = build_evidence_page_corpus(source)

    assert corpus.provenance is CorpusProvenance.EVIDENCE_PAGES_ONLY
    assert not corpus.is_reportable
    with pytest.raises(ValueError):
        require_reportable(corpus)


def test_built_corpus_contains_chunks_and_metadata(tmp_path):
    source = _write_source(tmp_path / "fb.jsonl", [_row(page=p, qid=f"q{p}") for p in range(1, 6)])
    corpus = build_evidence_page_corpus(
        source, chunking=ChunkingConfig(chunk_size_tokens=32, chunk_overlap_tokens=0)
    )
    assert len(corpus) > 0
    assert corpus.metadata["n_documents"] == 1
    assert corpus.metadata["n_pages"] == 5


def test_missing_source_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="data prepare"):
        build_evidence_page_corpus(tmp_path / "absent.jsonl")


def test_source_without_evidence_raises(tmp_path):
    source = _write_source(
        tmp_path / "fb.jsonl", [{"financebench_id": "q", "doc_name": "D", "evidence": []}]
    )
    with pytest.raises(ValueError, match="No evidence pages"):
        build_evidence_page_corpus(source)
