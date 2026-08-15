"""Resolving evidence references to chunks.

This module is the join between two vocabularies that would otherwise never
meet. FinanceBench states evidence as ``DOCUMENT::p60``; retrieval returns
chunks. Retrieval recall — the metric that keeps retrieval quality separable
from answer quality (spec section 11.1) — is only computable once those are
reconciled.

A reference resolves to **every chunk covering that page**. Some of those chunks
will contain the cited figure and some will merely overlap the page, so this is
a generous gold set: recall measured against it is an upper bound on how
precisely retrieval located the evidence. That limitation is stated here and
reported rather than hidden, because the honest alternative — span-level
alignment against the quoted evidence text — is a separate piece of work, and
pretending page-level matching is span-level would overstate retrieval quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from evidence_route.documents.models import Chunk

__all__ = [
    "EvidenceResolution",
    "parse_evidence_reference",
    "resolve_evidence",
]

#: Matches "DOCUMENT_NAME::p60" and the unknown-page form "DOCUMENT_NAME::p?".
_REFERENCE_RE = re.compile(r"^(?P<doc>.+)::p(?P<page>\d+|\?)$")


def parse_evidence_reference(reference: str) -> tuple[str, int | None]:
    """Split an evidence reference into document id and page number.

    Returns ``(document_id, None)`` when the page is unknown. Raises on a
    malformed reference rather than guessing — a silently misparsed reference
    would produce an empty gold set and make retrieval look worse than it is.
    """
    match = _REFERENCE_RE.match(reference.strip())
    if not match:
        raise ValueError(
            f"Malformed evidence reference {reference!r}; expected 'DOCUMENT::p<page>'."
        )
    page = match.group("page")
    return match.group("doc"), (None if page == "?" else int(page))


@dataclass
class EvidenceResolution:
    """Which chunks constitute the gold set for one question's evidence."""

    #: evidence reference -> chunk ids covering it
    chunks_by_reference: dict[str, list[str]] = field(default_factory=dict)
    #: References that matched no chunk at all.
    unresolved: list[str] = field(default_factory=list)
    #: References naming a document absent from the corpus.
    unknown_documents: list[str] = field(default_factory=list)
    #: References whose page number was unknown in the source data.
    unknown_pages: list[str] = field(default_factory=list)

    @property
    def gold_chunk_ids(self) -> set[str]:
        """The union of all chunks counted as evidence."""
        return {cid for ids in self.chunks_by_reference.values() for cid in ids}

    @property
    def is_fully_resolved(self) -> bool:
        return not self.unresolved and not self.unknown_documents

    def recall_at_k(self, retrieved_chunk_ids: list[str], k: int) -> float:
        """Fraction of gold chunks appearing in the top ``k`` retrieved.

        Returns 0.0 when there is no gold set — an unmeasurable question scores
        zero rather than a misleading 1.0, so questions whose evidence could not
        be resolved depress the metric visibly instead of silently inflating it.
        """
        gold = self.gold_chunk_ids
        if not gold:
            return 0.0
        top = set(retrieved_chunk_ids[:k])
        return len(gold & top) / len(gold)

    def hit_at_k(self, retrieved_chunk_ids: list[str], k: int) -> bool:
        """Whether *any* gold chunk appears in the top ``k``."""
        gold = self.gold_chunk_ids
        if not gold:
            return False
        return bool(gold & set(retrieved_chunk_ids[:k]))


def resolve_evidence(references: list[str], chunks: list[Chunk]) -> EvidenceResolution:
    """Map page-level evidence references onto chunk identifiers.

    ``chunks`` is the corpus to resolve against — typically every chunk of the
    documents a question refers to.
    """
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    resolution = EvidenceResolution()

    for reference in references:
        try:
            document_id, page = parse_evidence_reference(reference)
        except ValueError:
            resolution.unresolved.append(reference)
            continue

        candidates = by_document.get(document_id)
        if candidates is None:
            resolution.unknown_documents.append(reference)
            continue

        if page is None:
            # Page unknown in the source: the whole document is the gold set.
            # Recorded separately so these questions can be excluded from
            # precision-sensitive analyses rather than quietly widening them.
            resolution.unknown_pages.append(reference)
            matched = [c.chunk_id for c in candidates]
        else:
            matched = [c.chunk_id for c in candidates if c.covers_page(page)]

        if matched:
            resolution.chunks_by_reference[reference] = matched
        else:
            resolution.unresolved.append(reference)

    return resolution
