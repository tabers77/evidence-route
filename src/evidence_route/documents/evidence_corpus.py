"""Building the evidence-page development corpus.

FinanceBench embeds ``evidence_text_full_page`` alongside every evidence
reference — the complete text of the page the answer came from. Across the 150
open-source questions that is 189 pages of real filing text, already present in
the metadata file and needing no download.

That makes it the fastest way to get a working end-to-end pipeline on genuine
financial prose and tables: retrieval, generation, citation checking and the
deterministic scorers can all be built and debugged before a single PDF or HTML
filing has been fetched.

It is **not** a benchmark. See :class:`~evidence_route.documents.corpus.CorpusProvenance`
for why, and note that everything built here is stamped
``EVIDENCE_PAGES_ONLY`` so the reporting path refuses it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evidence_route.documents.chunking import ChunkingConfig, chunk_document
from evidence_route.documents.corpus import Corpus, CorpusProvenance
from evidence_route.documents.models import ParsedDocument, ParsedPage

__all__ = ["build_evidence_page_corpus", "parsed_documents_from_evidence"]


def parsed_documents_from_evidence(source: Path) -> list[ParsedDocument]:
    """Reconstruct partial documents from FinanceBench's embedded page text.

    Pages are keyed by ``(doc_name, evidence_page_num)`` and de-duplicated: the
    same page is frequently cited by several questions, and indexing it twice
    would let one page occupy two retrieval slots and distort recall.

    The resulting "documents" contain only their cited pages, so page numbers
    are real but non-contiguous — page 60 may be followed by page 104. That is
    faithful to what the data actually supports, and preserving the true page
    numbers keeps the existing ``DOC::p<page>`` evidence references valid.
    """
    pages_by_doc: dict[str, dict[int, str]] = defaultdict(dict)

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            for entry in row.get("evidence") or []:
                if not isinstance(entry, dict):
                    continue
                doc_name = str(entry.get("doc_name") or row.get("doc_name") or "").strip()
                page = entry.get("evidence_page_num")
                text = entry.get("evidence_text_full_page") or entry.get("evidence_text")
                if not doc_name or page is None or not text:
                    continue
                # First occurrence wins; later citations of the same page carry
                # identical text, and keeping one avoids duplicate chunks.
                pages_by_doc[doc_name].setdefault(int(page), str(text))

    documents: list[ParsedDocument] = []
    for doc_name in sorted(pages_by_doc):
        pages = tuple(
            ParsedPage(page_number=number, text=pages_by_doc[doc_name][number])
            for number in sorted(pages_by_doc[doc_name])
        )
        documents.append(
            ParsedDocument(
                document_id=doc_name,
                pages=pages,
                metadata={
                    "corpus_kind": "evidence_pages_only",
                    "n_pages_available": len(pages),
                    "note": (
                        "Cited evidence pages only; page numbers are real but "
                        "non-contiguous. Not the complete filing."
                    ),
                },
            )
        )
    return documents


def build_evidence_page_corpus(
    source: Path,
    *,
    chunking: ChunkingConfig | None = None,
    name: str = "financebench-evidence-pages",
) -> Corpus:
    """Build the development corpus from a FinanceBench metadata file.

    The returned corpus is stamped ``EVIDENCE_PAGES_ONLY``, so
    :func:`~evidence_route.documents.corpus.require_reportable` will reject it.
    That is deliberate and not a limitation to work around.
    """
    if not source.exists():
        raise FileNotFoundError(
            f"FinanceBench metadata not found at {source}. "
            f"Run `evidence-route data prepare --config configs/datasets/financebench.yaml` first."
        )

    documents = parsed_documents_from_evidence(source)
    if not documents:
        raise ValueError(
            f"No evidence pages could be reconstructed from {source}. "
            f"Expected records with evidence[].evidence_text_full_page."
        )

    config = chunking or ChunkingConfig()
    chunks = [chunk for document in documents for chunk in chunk_document(document, config)]

    return Corpus(
        name=name,
        chunks=chunks,
        provenance=CorpusProvenance.EVIDENCE_PAGES_ONLY,
        metadata={
            "source": str(source),
            "n_documents": len(documents),
            "n_pages": sum(d.n_pages for d in documents),
            "chunk_size_tokens": config.chunk_size_tokens,
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
        },
    )
