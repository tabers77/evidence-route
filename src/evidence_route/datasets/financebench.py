"""FinanceBench loader (spec section 7.1).

Reads the public ``financebench_open_source.jsonl`` sample into
:class:`RawQuestion` objects.

Two decisions worth stating:

**Question type is left unset.** FinanceBench labels questions as
``domain-relevant`` / ``novel-generated`` / ``metrics-generated``, which
describes how each question was *produced*, not what answering it requires.
EvidenceRoute's taxonomy — extraction, comparison, arithmetic, synthesis — is
about the latter, and it drives routing. Mapping one onto the other would
fabricate a signal the router then learns from, so the raw label is preserved in
metadata and ``question_type`` stays ``None`` until a classifier fills it in.

**Evidence identifiers are derived, not stored.** FinanceBench gives evidence as
document name plus page number. Chunk identifiers do not exist until parsing and
chunking run, so this loader emits stable ``doc::p<page>`` references that the
document stage later resolves to chunks. Retrieval recall is measured against
those, independently of whether the final answer was right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidence_route.datasets.base import RawQuestion

__all__ = ["evidence_reference", "load_financebench", "parse_financebench_row"]

DATASET_NAME = "financebench"


def evidence_reference(doc_name: str, page: int | str | None) -> str:
    """Build a stable evidence reference from a document and page.

    Deterministic so that re-running the loader produces identical references,
    and so evidence labels stay valid across re-parses.
    """
    if page is None or page == "":
        return f"{doc_name}::p?"
    return f"{doc_name}::p{page}"


def parse_financebench_row(row: dict[str, Any]) -> RawQuestion:
    """Convert one FinanceBench JSONL record into a :class:`RawQuestion`.

    Raises ``KeyError`` or ``ValueError`` on a row that cannot be interpreted;
    the caller records that as a parse failure rather than skipping it quietly.
    """
    question_id = str(row.get("financebench_id") or "").strip()
    if not question_id:
        raise ValueError("row has no financebench_id")

    company = str(row.get("company") or "").strip()
    doc_name = str(row.get("doc_name") or "").strip()

    evidence_entries = row.get("evidence") or []
    references: list[str] = []
    for entry in evidence_entries:
        if not isinstance(entry, dict):
            continue
        entry_doc = str(entry.get("doc_name") or doc_name).strip()
        if not entry_doc:
            continue
        references.append(evidence_reference(entry_doc, entry.get("evidence_page_num")))

    # Grouping by company, not by document: a single company usually has several
    # filings, and questions about different filings from the same company still
    # share terminology, formatting and business context. Grouping at the
    # document level would leave that shared signal spanning the split.
    group_key = company or doc_name or question_id

    return RawQuestion(
        question_id=question_id,
        dataset=DATASET_NAME,
        question_text=str(row.get("question") or "").strip(),
        group_key=group_key,
        document_ids=(doc_name,) if doc_name else (),
        reference_answer=(str(row["answer"]).strip() if row.get("answer") else None),
        reference_evidence=tuple(dict.fromkeys(references)),  # dedupe, keep order
        question_type=None,  # see module docstring
        metadata={
            "company": company,
            "doc_period": row.get("doc_period"),
            "doc_type": row.get("doc_type"),
            "doc_link": row.get("doc_link"),
            # FinanceBench's own label, kept verbatim so it is never confused
            # with EvidenceRoute's routing taxonomy.
            "financebench_question_type": row.get("question_type"),
            "justification": row.get("justification"),
            "evidence_pages": [
                e.get("evidence_page_num") for e in evidence_entries if isinstance(e, dict)
            ],
        },
    )


def load_financebench(path: Path) -> tuple[list[RawQuestion], list[str]]:
    """Load the FinanceBench JSONL sample.

    Returns the parsed questions and a list of parse-failure descriptions. The
    failures are returned rather than raised so one malformed row does not
    abandon the whole dataset — but they are returned rather than swallowed, so
    they still have to be looked at.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FinanceBench data not found at {path}. Run `evidence-route data prepare` "
            f"or download it from https://github.com/patronus-ai/financebench"
        )

    questions: list[RawQuestion] = []
    failures: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                failures.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            try:
                questions.append(parse_financebench_row(row))
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"line {line_number}: {exc}")

    return questions, failures
