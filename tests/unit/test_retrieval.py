"""Tests for tokenization and BM25 retrieval."""

from __future__ import annotations

import math

import pytest

from evidence_route.documents.models import Chunk
from evidence_route.retrieval.bm25 import BM25Config, BM25Retriever
from evidence_route.retrieval.tokenization import normalize_number, tokenize


def _chunk(ordinal: int, text: str, document_id: str = "ACME_10K") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::c{ordinal:04d}",
        document_id=document_id,
        text=text,
        page_numbers=(ordinal + 1,),
        ordinal=ordinal,
        token_estimate=max(1, len(text) // 4),
    )


# ---------------------------------------------------------------------------
# Tokenization — number handling is the point
# ---------------------------------------------------------------------------
def test_thousands_separated_numbers_survive_intact():
    """Shredding "66,608" into "66" and "608" would destroy the exact-match
    signal BM25 exists to exploit on this corpus."""
    tokens = tokenize("Total revenues 66,608 million")
    assert "66,608" in tokens


def test_numbers_expand_to_a_normalized_variant():
    tokens = tokenize("Total revenues 66,608")
    assert "66,608" in tokens
    assert "66608" in tokens


def test_currency_and_percent_are_preserved_and_normalized():
    tokens = tokenize("spend of $1,577.00 and margin of 23.4%")
    assert "$1,577.00" in tokens
    assert "1577" in tokens
    assert "23.4%" in tokens
    assert "23.4" in tokens


def test_accounting_negatives_normalize_to_a_minus():
    assert normalize_number("(1,577)") == "-1577"


def test_normalization_drops_trailing_decimal_zeros():
    """A filing writes 1,577.00 where a question says 1577; both must match."""
    assert normalize_number("$1,577.00") == "1577"
    assert normalize_number("23.40%") == "23.4"


def test_plain_integers_are_not_duplicated():
    """A year adds no variant, so its term frequency is not silently doubled."""
    assert tokenize("in 2018").count("2018") == 1


def test_query_and_document_number_formats_match():
    doc = tokenize("Total revenues 66,608")
    query = tokenize("What was revenue of $66,608?")
    assert set(doc) & set(query)


def test_tokenization_is_deterministic():
    text = "Revenue of $1,577.00 rose 23.4% in FY2018."
    assert tokenize(text) == tokenize(text)


def test_case_is_folded():
    assert tokenize("Revenue REVENUE revenue") == ["revenue"] * 3


def test_possessives_are_kept_whole():
    assert "company's" in tokenize("the company's revenue")


def test_stopwords_are_removed_by_default():
    tokens = tokenize("what was the revenue of the company")
    assert "the" not in tokens
    assert "revenue" in tokens


def test_financial_line_item_words_are_not_stopped():
    """ "net", "total" and "other" are line-item names here, not noise."""
    tokens = tokenize("net total other income")
    for term in ("net", "total", "other", "income"):
        assert term in tokens


def test_stopword_removal_can_be_disabled():
    assert "the" in tokenize("the revenue", remove_stopwords=False)


def test_empty_text_yields_no_tokens():
    assert tokenize("") == []
    assert tokenize("!!! ??? ...") == []


# ---------------------------------------------------------------------------
# BM25 configuration
# ---------------------------------------------------------------------------
def test_b_must_be_a_fraction():
    with pytest.raises(ValueError, match=r"b must lie in \[0, 1\]"):
        BM25Config(b=1.5)


def test_k1_must_be_non_negative():
    with pytest.raises(ValueError, match="k1 must be non-negative"):
        BM25Config(k1=-1.0)


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------
def _corpus() -> list[Chunk]:
    return [
        _chunk(0, "Total revenues 66,608 for the fiscal year"),
        _chunk(1, "Research and development expense 2,852"),
        _chunk(2, "Operating income margin 23.4% improved"),
        _chunk(3, "Nothing relevant about weather or sport"),
    ]


def test_retrieves_the_matching_chunk_first():
    retriever = BM25Retriever().index(_corpus())
    hits = retriever.search("total revenues", k=3)
    assert hits
    assert hits[0].chunk_id == "ACME_10K::c0000"


def test_numeric_query_finds_the_figure():
    """The behaviour the number-aware tokenizer exists to deliver."""
    retriever = BM25Retriever().index(_corpus())
    hits = retriever.search("66,608", k=3)
    assert hits[0].chunk_id == "ACME_10K::c0000"


def test_differently_formatted_number_still_matches():
    retriever = BM25Retriever().index(_corpus())
    assert retriever.search("66608", k=3)[0].chunk_id == "ACME_10K::c0000"


def test_ranks_are_one_based_and_contiguous():
    retriever = BM25Retriever().index(_corpus())
    hits = retriever.search("revenues expense income", k=4)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_scores_are_descending():
    retriever = BM25Retriever().index(_corpus())
    hits = retriever.search("revenues expense income margin", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_zero_scoring_chunks_are_not_returned():
    """Padding results with unmatched text would inflate the apparent context
    the generator was given."""
    retriever = BM25Retriever().index(_corpus())
    hits = retriever.search("revenues", k=4)
    assert len(hits) < 4
    assert all(h.score and h.score > 0 for h in hits)


def test_query_with_no_corpus_terms_returns_nothing():
    retriever = BM25Retriever().index(_corpus())
    assert retriever.search("zzzz nonexistent", k=5) == []


def test_empty_index_returns_nothing():
    assert BM25Retriever().index([]).search("anything", k=5) == []


def test_k_must_be_positive():
    retriever = BM25Retriever().index(_corpus())
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search("revenue", k=0)


def test_k_caps_the_result_count():
    corpus = [_chunk(i, f"revenue figure {i}") for i in range(10)]
    assert len(BM25Retriever().index(corpus).search("revenue", k=3)) == 3


# ---------------------------------------------------------------------------
# Determinism and tie-breaking
# ---------------------------------------------------------------------------
def test_search_is_reproducible():
    retriever = BM25Retriever().index(_corpus())
    a = retriever.search("revenues income", k=4)
    b = retriever.search("revenues income", k=4)
    assert [h.chunk_id for h in a] == [h.chunk_id for h in b]


def test_exact_ties_break_on_ordinal():
    """Repeated boilerplate makes exact ties common in filings.

    Without an explicit rule the ordering would depend on sort incidentals, and
    retrieval would stop being reproducible precisely where this corpus is
    densest.
    """
    identical = [_chunk(i, "See accompanying notes to the statements") for i in range(5)]
    hits = BM25Retriever().index(identical).search("accompanying notes", k=5)
    assert len({h.score for h in hits}) == 1, "expected an exact score tie"
    assert [h.chunk_id for h in hits] == [c.chunk_id for c in identical]


def test_index_order_does_not_change_tie_ordering():
    identical = [_chunk(i, "See accompanying notes") for i in range(5)]
    forward = BM25Retriever().index(identical).search("accompanying notes", k=5)
    reverse = BM25Retriever().index(list(reversed(identical))).search("accompanying notes", k=5)
    assert [h.chunk_id for h in forward] == [h.chunk_id for h in reverse]


# ---------------------------------------------------------------------------
# The formula itself
# ---------------------------------------------------------------------------
def test_idf_is_never_negative():
    """The textbook IDF goes negative for terms in more than half the corpus.

    On a single-company filing set the company name is exactly that common, and
    a negative IDF would let it push a document's score below zero.
    """
    corpus = [_chunk(i, "acme acme acme") for i in range(10)]
    retriever = BM25Retriever().index(corpus)
    assert all(score >= 0 for score in retriever.score("acme"))


def test_score_matches_hand_computation():
    """Verify the scoring function, not just its behaviour."""
    corpus = [
        _chunk(0, "alpha beta"),
        _chunk(1, "beta gamma"),
        _chunk(2, "gamma delta"),
    ]
    config = BM25Config(k1=1.2, b=0.75, expand_numbers=False, remove_stopwords=False)
    retriever = BM25Retriever(config=config).index(corpus)

    n_docs, df = 3, 1  # "alpha" appears in exactly one document
    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    tf, doc_len, avgdl = 1, 2, 2.0
    norm = 1.0 - config.b + config.b * (doc_len / avgdl)
    expected = idf * (tf * (config.k1 + 1.0)) / (tf + config.k1 * norm)

    assert retriever.score("alpha")[0] == pytest.approx(expected)


def test_rarer_terms_score_higher():
    corpus = [
        _chunk(0, "common rare"),
        _chunk(1, "common filler"),
        _chunk(2, "common filler"),
        _chunk(3, "common filler"),
    ]
    retriever = BM25Retriever().index(corpus)
    assert retriever.score("rare")[0] > retriever.score("common")[0]


def test_length_normalization_penalises_padding():
    """b > 0 must make a long, padded chunk score below a short focused one."""
    corpus = [
        _chunk(0, "revenue"),
        _chunk(1, "revenue " + "padding " * 200),
    ]
    retriever = BM25Retriever().index(corpus)
    scores = retriever.score("revenue")
    assert scores[0] > scores[1]


def test_length_normalization_can_be_switched_off():
    corpus = [_chunk(0, "revenue"), _chunk(1, "revenue " + "padding " * 200)]
    retriever = BM25Retriever(config=BM25Config(b=0.0)).index(corpus)
    scores = retriever.score("revenue")
    assert scores[0] == pytest.approx(scores[1])


def test_repeated_query_terms_count_more_than_once():
    corpus = [_chunk(0, "revenue growth"), _chunk(1, "revenue")]
    retriever = BM25Retriever().index(corpus)
    single = retriever.score("revenue")[0]
    doubled = retriever.score("revenue revenue")[0]
    assert doubled > single


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_retriever_reports_its_configuration():
    retriever = BM25Retriever().index(_corpus())
    assert "bm25" in retriever.name
    assert "k1=1.2" in retriever.name


def test_stats_describe_the_index():
    retriever = BM25Retriever().index(_corpus())
    stats = retriever.stats()
    assert stats["n_chunks"] == 4
    assert stats["vocabulary_size"] > 0
    assert stats["b"] == 0.75


def test_hits_carry_document_provenance():
    hits = BM25Retriever().index(_corpus()).search("revenues", k=1)
    assert hits[0].document_id == "ACME_10K"
    assert hits[0].text is not None
