"""Corpora and their provenance.

A corpus is not just a bag of chunks — it carries *how it was assembled*, and
that determines what may be claimed from results computed over it.

The reason this is a type rather than a convention: EvidenceRoute has a corpus
that is genuinely useful for development and genuinely invalid for reporting.
The evidence-page corpus is built from the gold pages FinanceBench embeds in its
metadata, so it contains real filing text and needs no download — but pooling
gold pages across questions means most distractors come from *other companies'*
documents, where the company name is a strong lexical discriminator. Real
retrieval means finding one page among ~150 pages of the *same* filing.

That does not merely make retrieval easier; it makes it easier **for BM25
specifically**, which is exactly the comparison the routing conclusions rest on.
A result computed there would bias H2 and H3 in a direction no confidence
interval would reveal.

So the constraint is enforced by :func:`require_reportable`, which the reporting
path calls, rather than left to whoever remembers the caveat six weeks from now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from evidence_route.documents.models import Chunk

__all__ = ["Corpus", "CorpusProvenance", "require_reportable"]


class CorpusProvenance(str, Enum):
    """How a corpus was assembled, and therefore what it can support."""

    #: Complete source documents. The only provenance valid for reported results.
    FULL_DOCUMENTS = "full_documents"

    #: Gold evidence pages only, pooled across questions. Development use only:
    #: the distractor distribution is unrepresentative in a way that flatters
    #: lexical retrieval.
    EVIDENCE_PAGES_ONLY = "evidence_pages_only"

    #: Hand-built fixtures for tests and the offline smoke experiment.
    FIXTURE = "fixture"

    @property
    def is_reportable(self) -> bool:
        """Whether a public claim may cite results computed over this corpus."""
        return self is CorpusProvenance.FULL_DOCUMENTS

    @property
    def rationale(self) -> str:
        """Why this provenance is or is not reportable."""
        return _RATIONALE[self]


_RATIONALE: dict[CorpusProvenance, str] = {
    CorpusProvenance.FULL_DOCUMENTS: (
        "Complete source documents; the retrieval task matches deployment."
    ),
    CorpusProvenance.EVIDENCE_PAGES_ONLY: (
        "Contains only gold evidence pages pooled across questions. Most "
        "distractors come from other companies' documents, so the company name "
        "becomes a strong lexical discriminator and retrieval is easier than "
        "reality — disproportionately so for BM25. Valid for building and "
        "debugging the pipeline; invalid for any reported comparison."
    ),
    CorpusProvenance.FIXTURE: ("Hand-built test data. Exercises code paths, measures nothing."),
}


@dataclass
class Corpus:
    """Indexable chunks plus the provenance that governs their use."""

    name: str
    chunks: list[Chunk]
    provenance: CorpusProvenance
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_reportable(self) -> bool:
        return self.provenance.is_reportable

    @property
    def document_ids(self) -> set[str]:
        return {c.document_id for c in self.chunks}

    def chunks_for(self, document_ids: set[str]) -> list[Chunk]:
        """Restrict to chunks from the given documents.

        Used to scope retrieval to the documents a question is asked against,
        which is how FinanceBench is posed — the question names its filing.
        """
        return [c for c in self.chunks if c.document_id in document_ids]

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance.value,
            "reportable": self.is_reportable,
            "n_chunks": len(self.chunks),
            "n_documents": len(self.document_ids),
            "total_tokens": sum(c.token_estimate for c in self.chunks),
        }


def require_reportable(corpus: Corpus) -> None:
    """Raise unless results over ``corpus`` may be published.

    Called from the reporting path. The point is that forgetting the caveat
    produces a loud failure rather than a plausible-looking table.
    """
    if not corpus.is_reportable:
        raise ValueError(
            f"Corpus {corpus.name!r} has provenance "
            f"{corpus.provenance.value!r} and cannot support a reported result.\n"
            f"{corpus.provenance.rationale}\n"
            f"Build a full-document corpus, or mark this run as exploratory."
        )
