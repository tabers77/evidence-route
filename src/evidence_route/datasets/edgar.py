"""Fetching SEC filings from EDGAR.

Why EDGAR rather than the links FinanceBench ships: those point at 23 different
corporate investor-relations hosts and CDNs, only 9 of 84 carry an accession
number, and corporate IR pages rot and block automated clients. Putting that on
the reproduction path would undermine the claim that a reviewer can rebuild the
benchmark (spec section 32).

EDGAR is the opposite: permanent URLs, a documented submissions API, explicit
permission for automated access given a declaring User-Agent, and a published
rate limit. 78 of the 84 FinanceBench documents are SEC filings (64 10-K, 8
10-Q, 6 8-K) and are all obtainable this way; the remaining 6 are earnings
releases, which are not SEC filings and must come from their original links or
be documented as excluded.

**One consequence to be aware of.** EDGAR serves HTML/iXBRL, not the PDFs
FinanceBench annotated. That is a net gain for parsing — tables are real
``<table>`` elements rather than glyph positions to be reconstructed — but it
means *page numbers do not exist*. Evidence anchored as ``DOC::p60`` cannot be
resolved against an HTML filing. Text-anchored alignment against the
``evidence_text`` span is required instead, and that is deliberately not
implemented here: this module acquires documents, nothing more.

SEC access rules, honoured below:
  - a User-Agent identifying the requester, including contact information
  - no more than 10 requests per second
Both are documented at https://www.sec.gov/os/webmaster-faq#developers
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EDGAR_RATE_LIMIT_PER_SECOND",
    "EdgarClient",
    "EdgarFiling",
    "FilingFetchResult",
    "load_company_cik_map",
    "parse_document_name",
]

#: The SEC's published ceiling. Requests are spaced to stay under it.
EDGAR_RATE_LIMIT_PER_SECOND = 10.0

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
_TIMEOUT_SECONDS = 60

#: FinanceBench document-type labels mapped onto EDGAR form types.
FORM_TYPES: dict[str, str] = {
    "10k": "10-K",
    "10q": "10-Q",
    "8k": "8-K",
}


@dataclass(frozen=True)
class EdgarFiling:
    """One filing located in EDGAR's submissions index."""

    cik: int
    accession_number: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return _ARCHIVE_URL.format(
            cik=self.cik,
            accession_nodash=self.accession_nodash,
            document=self.primary_document,
        )

    @property
    def report_year(self) -> int | None:
        try:
            return int(self.report_date[:4])
        except (ValueError, TypeError):
            return None


@dataclass
class FilingFetchResult:
    """Outcome of trying to obtain one document."""

    document_name: str
    succeeded: bool
    path: Path | None = None
    filing: EdgarFiling | None = None
    bytes_written: int = 0
    reason: str | None = None


def load_company_cik_map(path: Path) -> dict[str, int]:
    """Load the reviewed company-name → CIK mapping.

    A committed, hand-verified file rather than fuzzy matching at runtime.
    Automatic matching against EDGAR's ticker file resolved "AMD" to *Amdocs*,
    which would have silently fetched a different company's filings and
    corrupted the benchmark in a way nothing downstream could detect. Name
    matching is a suggestion tool; the mapping in git is the source of truth.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {company: int(entry["cik"]) for company, entry in payload.items()}


def parse_document_name(doc_name: str) -> tuple[str, int, str | None, str | None]:
    """Split a FinanceBench ``doc_name`` into its parts.

    ``3M_2018_10K``                     -> ("3M", 2018, None, "10-K")
    ``3M_2023Q2_10Q``                   -> ("3M", 2023, "Q2", "10-Q")
    ``AMCOR_2022_8K_dated-2022-07-01``  -> ("AMCOR", 2022, None, "8-K")

    The form token is located by scanning rather than taken from the end. 8-K
    names carry a trailing ``dated-<date>`` qualifier, so reading the last part
    as the form silently classified six genuine SEC filings as unfetchable.

    Returns the form as ``None`` when no SEC form type appears, which is how
    earnings releases are identified and routed elsewhere.
    """
    parts = doc_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unrecognised document name {doc_name!r}; expected COMPANY_PERIOD_FORM.")

    # Scan from the right: the form follows the period, and a trailing
    # qualifier may follow the form.
    form_index = next(
        (i for i in range(len(parts) - 1, 0, -1) if parts[i].lower() in FORM_TYPES),
        None,
    )
    if form_index is None or form_index < 1:
        # No recognisable form. Fall back to the trailing layout so the period
        # is still reported, and let the caller treat form=None as non-SEC.
        form_index = len(parts) - 1

    form_token = parts[form_index].lower()
    period_token = parts[form_index - 1]
    company_token = "_".join(parts[: form_index - 1])

    quarter: str | None = None
    if "Q" in period_token.upper():
        year_text, _, quarter_text = period_token.upper().partition("Q")
        quarter = f"Q{quarter_text}" if quarter_text else None
    else:
        year_text = period_token

    try:
        year = int(year_text)
    except ValueError as exc:
        raise ValueError(f"Could not read a year from {doc_name!r}.") from exc

    return company_token, year, quarter, FORM_TYPES.get(form_token)


@dataclass
class EdgarClient:
    """A rate-limited EDGAR client.

    ``user_agent`` must identify the requester and include contact information;
    the SEC requires it and returns 403 without it. It is read from the
    environment rather than hardcoded, because it contains a real email address.
    """

    user_agent: str
    rate_limit_per_second: float = EDGAR_RATE_LIMIT_PER_SECOND
    _last_request_at: float = field(default=0.0, repr=False)
    _submissions_cache: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.user_agent or "@" not in self.user_agent:
            raise ValueError(
                "EDGAR requires a User-Agent identifying you and including a "
                "contact email, e.g. 'EvidenceRoute research you@example.com'. "
                "Set EVIDENCE_ROUTE_SEC_USER_AGENT in .env.\n"
                "See https://www.sec.gov/os/webmaster-faq#developers"
            )
        if self.rate_limit_per_second > EDGAR_RATE_LIMIT_PER_SECOND:
            raise ValueError(
                f"rate_limit_per_second must not exceed the SEC's published "
                f"limit of {EDGAR_RATE_LIMIT_PER_SECOND}/s."
            )

    # -- transport ----------------------------------------------------------
    def _throttle(self) -> None:
        """Space requests to stay inside the published rate limit."""
        minimum_gap = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < minimum_gap:
            time.sleep(minimum_gap - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, url: str) -> bytes:
        self._throttle()
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise OSError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise OSError(f"Could not reach {url}: {exc.reason}") from exc

    # -- API ----------------------------------------------------------------
    def submissions(self, cik: int) -> dict[str, Any]:
        """Fetch (and cache) a company's submissions index.

        Cached per client because one company typically supplies several
        documents, and refetching a 160 KB index per filing wastes both the rate
        budget and the SEC's bandwidth.
        """
        if cik not in self._submissions_cache:
            payload = self._get(_SUBMISSIONS_URL.format(cik=cik))
            self._submissions_cache[cik] = json.loads(payload)
        return self._submissions_cache[cik]

    def list_filings(self, cik: int, form: str) -> list[EdgarFiling]:
        """Every filing of one form type, newest first.

        Only ``filings.recent`` is read. It holds the most recent 1000 filings,
        which covers FinanceBench's 2015-2023 range comfortably; older pages
        exist under ``filings.files`` and would need paging for an earlier
        corpus.
        """
        data = self.submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])

        filings: list[EdgarFiling] = []
        for i, filing_form in enumerate(forms):
            if filing_form != form:
                continue
            filings.append(
                EdgarFiling(
                    cik=cik,
                    accession_number=recent["accessionNumber"][i],
                    form=filing_form,
                    filing_date=recent["filingDate"][i],
                    report_date=recent.get("reportDate", [""] * len(forms))[i],
                    primary_document=recent.get("primaryDocument", [""] * len(forms))[i],
                )
            )
        return filings

    def find_filing(
        self, cik: int, form: str, fiscal_year: int, quarter: str | None = None
    ) -> EdgarFiling | None:
        """Locate the filing covering a fiscal year (and quarter, for 10-Qs).

        Matched on ``reportDate`` rather than ``filingDate``: a 10-K for fiscal
        2018 is filed in early 2019, so filing date would systematically select
        the wrong year. Companies with non-calendar fiscal years are why the
        match also accepts the following calendar year — Nike's FY2021 ends in
        May 2021, but Walmart's FY2021 ends in January 2021.
        """
        candidates = [f for f in self.list_filings(cik, form) if f.report_year == fiscal_year]

        if quarter and candidates:
            month_for_quarter = {
                "Q1": (1, 2, 3, 4),
                "Q2": (4, 5, 6, 7),
                "Q3": (7, 8, 9, 10),
                "Q4": (10, 11, 12, 1),
            }
            months = month_for_quarter.get(quarter.upper())
            if months:
                narrowed = [
                    f
                    for f in candidates
                    if f.report_date[5:7].isdigit() and int(f.report_date[5:7]) in months
                ]
                if narrowed:
                    candidates = narrowed

        if not candidates:
            return None
        # Newest report date first, so an amended or later-covering filing wins.
        return sorted(candidates, key=lambda f: f.report_date, reverse=True)[0]

    def download_filing(self, filing: EdgarFiling, destination: Path) -> int:
        """Download a filing's primary document. Returns bytes written.

        Written atomically, for the same reason dataset downloads are: a
        truncated file left at the destination would be silently treated as a
        complete document on the next run.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._get(filing.document_url)
        if not payload:
            raise OSError(f"Empty document at {filing.document_url}")

        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return len(payload)
