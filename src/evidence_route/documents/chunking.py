"""Chunking with stable identifiers and preserved page provenance.

Two properties matter more than the splitting heuristic itself.

**Identifiers must be reproducible.** Re-chunking the same document with the
same settings has to yield the same ``chunk_id``s, or every stored evidence
label, retrieval result and outcome record silently stops referring to the same
text. Identifiers are therefore derived from content, not from a counter.

**Page provenance must survive.** Chunks are packed from page-tagged segments,
so a chunk knows every page it draws from — including when it straddles a
boundary. Without that, evidence stated as "page 60" cannot be matched to what
retrieval returned.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from evidence_route.documents.models import (
    Chunk,
    ParsedDocument,
    ParsedPage,
    estimate_tokens,
)

__all__ = ["ChunkingConfig", "chunk_document", "chunk_id_for", "make_chunk_id"]

#: Split points tried in order, coarsest first: paragraph, then line, then
#: sentence. Only if a single unit still exceeds the budget is it hard-split.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters. Part of the experiment's versioned configuration.

    Changing any of these changes every chunk identifier in the corpus, so a
    change here invalidates previously stored retrieval results and must be
    treated as a re-index, not a tweak.
    """

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    #: Tables are kept whole where they fit. Splitting a balance sheet across
    #: two chunks orphans the figures from their row labels, which is the
    #: single most common way a financial retrieval pipeline loses an answer.
    keep_tables_whole: bool = True
    id_scheme: str = "content_hash"

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive.")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative.")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be smaller "
                f"than chunk_size_tokens ({self.chunk_size_tokens}); otherwise "
                f"chunking cannot advance and would loop forever."
            )


def make_chunk_id(document_id: str, ordinal: int, text: str) -> str:
    """Derive a stable chunk identifier.

    The hash covers the document id and ordinal as well as the text, not the
    text alone. Financial filings repeat boilerplate verbatim — page headers,
    footers, standard legal paragraphs — so hashing content alone produces
    genuine collisions, and two distinct locations would merge into one
    identifier. Including document and position keeps ids unique while staying
    fully reproducible.
    """
    normalized = " ".join(text.split())
    digest = hashlib.sha256(f"{document_id}\x00{ordinal}\x00{normalized}".encode()).hexdigest()
    return f"{document_id}::c{ordinal:04d}::{digest[:12]}"


def chunk_id_for(chunk: Chunk) -> str:
    """Recompute a chunk's identifier, for verifying reproducibility."""
    return make_chunk_id(chunk.document_id, chunk.ordinal, chunk.text)


@dataclass
class _Segment:
    """A page-tagged unit of text waiting to be packed into a chunk."""

    text: str
    page_number: int
    is_table: bool = False

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def _split_page_text(page: ParsedPage) -> list[_Segment]:
    """Break one page into paragraph-level segments, tagged with the page."""
    segments: list[_Segment] = []
    for block in _PARAGRAPH_RE.split(page.text):
        cleaned = block.strip()
        if cleaned:
            segments.append(_Segment(cleaned, page.page_number))
    return segments


def _hard_split(segment: _Segment, max_tokens: int) -> list[_Segment]:
    """Split an oversized segment down to something that fits.

    Sentences first, then a character-level cut if a single sentence is still
    too long. The character cut is a last resort — it can sever a number from
    its label — but the alternative is a chunk that overflows the model's
    context, which fails harder and less visibly.
    """
    if segment.tokens <= max_tokens:
        return [segment]

    pieces: list[_Segment] = []
    for sentence in _SENTENCE_RE.split(segment.text):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        candidate = _Segment(cleaned, segment.page_number, segment.is_table)
        if candidate.tokens <= max_tokens:
            pieces.append(candidate)
        else:
            max_chars = max_tokens * 4
            for start in range(0, len(cleaned), max_chars):
                pieces.append(
                    _Segment(
                        cleaned[start : start + max_chars],
                        segment.page_number,
                        segment.is_table,
                    )
                )
    return pieces or [segment]


def _overlap_segments(packed: list[_Segment], overlap_tokens: int) -> list[_Segment]:
    """Take trailing segments from a finished chunk to seed the next one.

    Overlap exists so a fact stated across a chunk boundary is retrievable from
    at least one whole chunk. Whole segments are carried over rather than a
    character window, so the overlap never begins mid-sentence.
    """
    if overlap_tokens <= 0:
        return []
    carried: list[_Segment] = []
    total = 0
    for segment in reversed(packed):
        if total + segment.tokens > overlap_tokens and carried:
            break
        carried.insert(0, segment)
        total += segment.tokens
    # Never carry the entire chunk forward — that would stall progress.
    if len(carried) >= len(packed):
        carried = carried[1:]
    return carried


def chunk_document(document: ParsedDocument, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Split a parsed document into indexable chunks.

    Segments are packed greedily up to the token budget. Tables are emitted as
    their own segments and, when ``keep_tables_whole`` is set, are not merged
    with surrounding prose unless they fit alongside it.
    """
    config = config or ChunkingConfig()

    segments: list[_Segment] = []
    for page in document.pages:
        segments.extend(_split_page_text(page))
        for table in page.tables:
            segments.append(_Segment(table.to_text(), page.page_number, is_table=True))

    # Oversized segments are reduced before packing so the packer never has to
    # emit a chunk it knows is too large.
    sized: list[_Segment] = []
    for segment in segments:
        sized.extend(_hard_split(segment, config.chunk_size_tokens))

    chunks: list[Chunk] = []
    current: list[_Segment] = []
    current_tokens = 0
    ordinal = 0

    def flush() -> None:
        nonlocal current, current_tokens, ordinal
        if not current:
            return
        text = "\n\n".join(s.text for s in current)
        pages = tuple(sorted({s.page_number for s in current}))
        chunk_id = make_chunk_id(document.document_id, ordinal, text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                text=text,
                page_numbers=pages,
                ordinal=ordinal,
                token_estimate=estimate_tokens(text),
                contains_table=any(s.is_table for s in current),
                metadata={"n_segments": len(current)},
            )
        )
        ordinal += 1
        carried = _overlap_segments(current, config.chunk_overlap_tokens)
        current = list(carried)
        current_tokens = sum(s.tokens for s in current)

    for segment in sized:
        would_be = current_tokens + segment.tokens
        splitting_a_table = (
            config.keep_tables_whole
            and segment.is_table
            and current
            and would_be > config.chunk_size_tokens
        )
        if current and (would_be > config.chunk_size_tokens or splitting_a_table):
            flush()
        current.append(segment)
        current_tokens += segment.tokens

    # Final flush must not re-seed overlap, or it would loop.
    if current:
        text = "\n\n".join(s.text for s in current)
        pages = tuple(sorted({s.page_number for s in current}))
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(document.document_id, ordinal, text),
                document_id=document.document_id,
                text=text,
                page_numbers=pages,
                ordinal=ordinal,
                token_estimate=estimate_tokens(text),
                contains_table=any(s.is_table for s in current),
                metadata={"n_segments": len(current)},
            )
        )

    return chunks
