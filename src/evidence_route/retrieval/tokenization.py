"""Tokenization for lexical retrieval.

Number handling is the whole point of this module, not an afterthought.

A conventional tokenizer splits on punctuation, which turns ``66,608`` into
``66`` and ``608`` and ``$1,577.00`` into three meaningless fragments. On a
general corpus that costs little. On FinanceBench it is close to fatal: most
questions ask for a specific figure, the answer appears in the document as a
formatted number, and shredding it destroys the exact-match signal that BM25
exists to exploit. A1 would then look like a weak baseline for reasons that have
nothing to do with BM25.

So numbers are kept whole *and* expanded into normalized variants, letting a
query for ``66608``, ``66,608`` or ``$66,608`` all match the same document term.
"""

from __future__ import annotations

import re

__all__ = ["STOPWORDS", "has_digits", "normalize_number", "tokenize"]

#: Numbers first, so the number pattern claims "66,608" before the word pattern
#: can split it. Covers currency prefixes, thousands separators, decimals,
#: percentages and negatives written in accounting parentheses.
_TOKEN_RE = re.compile(
    r"""
    (?P<number>
        \(?                 # accounting negative: (1,234)
        [$€£]?              # currency symbol
        \d[\d,]*            # integer part with thousands separators
        (?:\.\d+)?          # decimal part
        \)?
        %?                  # percentage
    )
    |
    (?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)   # words, keeping "company's" together
    """,
    re.VERBOSE,
)

_DIGITS_RE = re.compile(r"\d")

#: A short, explicit stop list. Deliberately small: BM25's IDF term already
#: discounts common words, and aggressive stopping removes tokens that carry
#: real meaning in financial questions ("net", "total", "other" are all
#: line-item names). Only true function words are listed.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)


def normalize_number(token: str) -> str:
    """Reduce a formatted number to comparable digits.

    ``$1,577.00`` and ``(1,577)`` both normalize to ``1577.00`` / ``1577``, so a
    question phrasing a figure differently from the filing still matches.
    Trailing zeros after a decimal point are dropped, because ``1577.00`` and
    ``1577`` are the same quantity and only one of the two will typically appear.
    """
    negative = token.startswith("(") and token.endswith(")")
    cleaned = token.strip("()").lstrip("$€£").rstrip("%").replace(",", "")
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return f"-{cleaned}" if negative and cleaned else cleaned


def tokenize(text: str, *, expand_numbers: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Split text into lexical retrieval terms.

    Deterministic by construction: the same input always yields the same token
    list in the same order. Retrieval reproducibility depends on it, since a
    tokenizer that varied would change BM25 scores and therefore rankings.

    When ``expand_numbers`` is set, a formatted number contributes both its
    surface form and its normalized form, so the document and the question do
    not have to agree on formatting.
    """
    tokens: list[str] = []

    for match in _TOKEN_RE.finditer(text):
        number = match.group("number")
        if number:
            surface = number.lower()
            tokens.append(surface)
            if expand_numbers:
                normalized = normalize_number(surface)
                # Only add a variant if it actually differs, to avoid inflating
                # term frequency for plain integers like "2018".
                if normalized and normalized != surface:
                    tokens.append(normalized)
            continue

        word = match.group("word")
        if word:
            lowered = word.lower()
            if remove_stopwords and lowered in STOPWORDS:
                continue
            tokens.append(lowered)

    return tokens


def has_digits(token: str) -> bool:
    """Whether a token contains any digit. Used by numeric-aware features."""
    return bool(_DIGITS_RE.search(token))
