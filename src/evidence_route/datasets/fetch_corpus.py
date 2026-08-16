"""Fetching the FinanceBench document corpus from EDGAR.

Per-document failure is expected and survivable. Across 84 documents from 32
companies some will not resolve — a company renamed, a fiscal year that does not
line up, an earnings release that was never an SEC filing. Aborting the whole
run on the first miss would make the corpus impossible to assemble
incrementally.

What is *not* acceptable is a silent gap. Every failure is recorded with a
reason, the coverage report states exactly which documents are missing, and the
manifest carries that so a later result can say honestly which subset it covers
(spec section 21, week 11: negative findings are retained).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from evidence_route.datasets.edgar import (
    EdgarClient,
    FilingFetchResult,
    load_company_cik_map,
    parse_document_name,
)

__all__ = ["CorpusFetchReport", "fetch_financebench_corpus", "plan_corpus_fetch"]


@dataclass
class DocumentRequest:
    """One document the corpus needs."""

    document_name: str
    company: str
    fiscal_year: int
    quarter: str | None
    form: str | None
    doc_link: str | None


@dataclass
class CorpusFetchReport:
    """What the fetch obtained, and what it did not."""

    results: list[FilingFetchResult] = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def succeeded(self) -> list[FilingFetchResult]:
        return [r for r in self.results if r.succeeded]

    @property
    def failed(self) -> list[FilingFetchResult]:
        return [r for r in self.results if not r.succeeded]

    @property
    def coverage(self) -> float:
        return len(self.succeeded) / len(self.results) if self.results else 0.0

    def failure_reasons(self) -> dict[str, int]:
        return dict(Counter((r.reason or "unknown").split(":")[0] for r in self.failed))

    def summary(self) -> str:
        lines = [
            f"corpus: {len(self.succeeded)}/{len(self.results)} documents "
            f"({self.coverage:.0%} coverage)"
        ]
        for reason, count in sorted(self.failure_reasons().items()):
            lines.append(f"  {reason}: {count}")
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        """Persist the coverage report next to the corpus.

        Committed alongside results so any claim about FinanceBench can state
        which documents it actually had.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_requested": len(self.results),
            "n_obtained": len(self.succeeded),
            "coverage": round(self.coverage, 4),
            "failure_reasons": self.failure_reasons(),
            "obtained": sorted(r.document_name for r in self.succeeded),
            "missing": [
                {"document": r.document_name, "reason": r.reason}
                for r in sorted(self.failed, key=lambda x: x.document_name)
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def plan_corpus_fetch(source: Path) -> list[DocumentRequest]:
    """Read the distinct documents FinanceBench references."""
    seen: dict[str, DocumentRequest] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            doc_name = str(row.get("doc_name") or "").strip()
            if not doc_name or doc_name in seen:
                continue
            try:
                _, year, quarter, form = parse_document_name(doc_name)
            except ValueError:
                # Recorded as a request with no form so it appears in the
                # coverage report rather than vanishing from the plan.
                year, quarter, form = 0, None, None

            seen[doc_name] = DocumentRequest(
                document_name=doc_name,
                company=str(row.get("company") or "").strip(),
                fiscal_year=year,
                quarter=quarter,
                form=form,
                doc_link=row.get("doc_link"),
            )

    return [seen[name] for name in sorted(seen)]


def fetch_financebench_corpus(
    source: Path,
    output_dir: Path,
    cik_map_path: Path,
    *,
    user_agent: str,
    limit: int | None = None,
    skip_existing: bool = True,
) -> CorpusFetchReport:
    """Fetch FinanceBench's SEC filings from EDGAR.

    ``limit`` fetches only the first N documents, which is how to sanity-check
    the path before committing to all 84.
    """
    requests = plan_corpus_fetch(source)
    if limit is not None:
        requests = requests[:limit]

    cik_map = load_company_cik_map(cik_map_path)
    client = EdgarClient(user_agent=user_agent)
    report = CorpusFetchReport(output_dir=output_dir)

    for request in requests:
        result = _fetch_one(client, request, cik_map, output_dir, skip_existing)
        report.results.append(result)

    return report


def _fetch_one(
    client: EdgarClient,
    request: DocumentRequest,
    cik_map: dict[str, int],
    output_dir: Path,
    skip_existing: bool,
) -> FilingFetchResult:
    """Fetch one document, converting any failure into a recorded reason."""
    if request.form is None:
        return FilingFetchResult(
            request.document_name,
            succeeded=False,
            reason=(
                "not_an_sec_filing: earnings releases are not filed with the SEC "
                f"and must come from {request.doc_link or 'their original link'}"
            ),
        )

    cik = cik_map.get(request.company)
    if cik is None:
        return FilingFetchResult(
            request.document_name,
            succeeded=False,
            reason=f"unknown_company: {request.company!r} absent from the reviewed CIK map",
        )

    # EDGAR serves HTML, so the extension reflects what is actually stored.
    destination = output_dir / f"{request.document_name}.htm"
    if skip_existing and destination.exists() and destination.stat().st_size > 0:
        return FilingFetchResult(
            request.document_name,
            succeeded=True,
            path=destination,
            bytes_written=destination.stat().st_size,
            reason="already_present",
        )

    try:
        filing = client.find_filing(cik, request.form, request.fiscal_year, request.quarter)
    except OSError as exc:
        return FilingFetchResult(
            request.document_name, succeeded=False, reason=f"lookup_failed: {exc}"
        )

    if filing is None:
        return FilingFetchResult(
            request.document_name,
            succeeded=False,
            reason=(
                f"filing_not_found: no {request.form} with a {request.fiscal_year} "
                f"report date for CIK {cik}"
            ),
        )

    if not filing.primary_document:
        return FilingFetchResult(
            request.document_name,
            succeeded=False,
            filing=filing,
            reason=f"no_primary_document: {filing.accession_number}",
        )

    try:
        written = client.download_filing(filing, destination)
    except OSError as exc:
        return FilingFetchResult(
            request.document_name, succeeded=False, filing=filing, reason=f"download_failed: {exc}"
        )

    return FilingFetchResult(
        request.document_name,
        succeeded=True,
        path=destination,
        filing=filing,
        bytes_written=written,
    )
