"""Tests for EDGAR filing acquisition.

The network is mocked throughout — these check the logic that decides *which*
filing to fetch, which is where a silent error would corrupt the corpus.
"""

from __future__ import annotations

import json

import pytest

from evidence_route.datasets.edgar import (
    EDGAR_RATE_LIMIT_PER_SECOND,
    EdgarClient,
    EdgarFiling,
    FilingFetchResult,
    load_company_cik_map,
    parse_document_name,
)
from evidence_route.datasets.fetch_corpus import (
    CorpusFetchReport,
    fetch_financebench_corpus,
    plan_corpus_fetch,
)

UA = "EvidenceRoute test test@example.com"


# ---------------------------------------------------------------------------
# Document name parsing
# ---------------------------------------------------------------------------
def test_parses_an_annual_filing():
    assert parse_document_name("3M_2018_10K") == ("3M", 2018, None, "10-K")


def test_parses_a_quarterly_filing():
    assert parse_document_name("3M_2023Q2_10Q") == ("3M", 2023, "Q2", "10-Q")


def test_parses_a_multiword_company():
    company, year, _, form = parse_document_name("JOHNSON_JOHNSON_2022_10K")
    assert company == "JOHNSON_JOHNSON"
    assert (year, form) == (2022, "10-K")


def test_parses_an_8k_with_a_trailing_date_qualifier():
    """8-K names carry a `dated-<date>` suffix after the form.

    Reading the last token as the form silently classified six genuine SEC
    filings as unfetchable.
    """
    assert parse_document_name("AMCOR_2022_8K_dated-2022-07-01") == (
        "AMCOR",
        2022,
        None,
        "8-K",
    )
    assert parse_document_name("FOOTLOCKER_2022_8K_dated_2022-08-19")[3] == "8-K"


def test_non_sec_form_yields_no_form_type():
    """Earnings releases are not filed with the SEC and must be routed elsewhere."""
    assert parse_document_name("PEPSICO_2022_EARNINGS")[3] is None
    assert parse_document_name("ULTABEAUTY_2023Q4_EARNINGS")[3] is None


def test_unparseable_name_raises():
    with pytest.raises(ValueError, match="Unrecognised document name"):
        parse_document_name("nonsense")


def test_non_numeric_year_raises():
    with pytest.raises(ValueError, match="Could not read a year"):
        parse_document_name("ACME_NOTAYEAR_10K")


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------
def test_user_agent_must_carry_contact_information():
    """EDGAR returns 403 without one; failing early beats failing mid-fetch."""
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient(user_agent="evidence-route")


def test_empty_user_agent_is_rejected():
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient(user_agent="")


def test_rate_limit_cannot_exceed_the_published_ceiling():
    with pytest.raises(ValueError, match="published"):
        EdgarClient(user_agent=UA, rate_limit_per_second=EDGAR_RATE_LIMIT_PER_SECOND + 1)


# ---------------------------------------------------------------------------
# Filing selection — where a silent error would corrupt the corpus
# ---------------------------------------------------------------------------
def _submissions(forms: list[tuple[str, str, str, str]]) -> dict:
    """Build a submissions payload from (form, accession, reportDate, doc)."""
    return {
        "cik": "0000000123",
        "filings": {
            "recent": {
                "form": [f[0] for f in forms],
                "accessionNumber": [f[1] for f in forms],
                "reportDate": [f[2] for f in forms],
                "filingDate": [f[2] for f in forms],
                "primaryDocument": [f[3] for f in forms],
            }
        },
    }


@pytest.fixture
def client(monkeypatch):
    c = EdgarClient(user_agent=UA)
    payload = _submissions(
        [
            ("10-K", "0000-23-000001", "2022-12-31", "acme-20221231.htm"),
            ("10-K", "0000-22-000001", "2021-12-31", "acme-20211231.htm"),
            ("10-Q", "0000-22-000050", "2022-06-30", "acme-20220630.htm"),
            ("10-Q", "0000-22-000030", "2022-03-31", "acme-20220331.htm"),
            ("8-K", "0000-22-000099", "2022-05-01", "acme-8k.htm"),
        ]
    )
    monkeypatch.setattr(EdgarClient, "_get", lambda self, url: json.dumps(payload).encode())
    # Remove the sleep so tests do not pay the rate limit.
    monkeypatch.setattr(EdgarClient, "_throttle", lambda self: None)
    return c


def test_finds_the_filing_for_a_fiscal_year(client):
    filing = client.find_filing(123, "10-K", 2022)
    assert filing is not None
    assert filing.accession_number == "0000-23-000001"


def test_matches_on_report_date_not_filing_date(client):
    """A 10-K for fiscal 2021 is filed in 2022.

    Matching on filing date would systematically select the wrong year.
    """
    filing = client.find_filing(123, "10-K", 2021)
    assert filing is not None
    assert filing.report_date.startswith("2021")


def test_quarter_narrows_among_several_10qs(client):
    q2 = client.find_filing(123, "10-Q", 2022, quarter="Q2")
    q1 = client.find_filing(123, "10-Q", 2022, quarter="Q1")
    assert q2 is not None and q2.report_date == "2022-06-30"
    assert q1 is not None and q1.report_date == "2022-03-31"


def test_missing_year_returns_none_rather_than_a_wrong_filing(client):
    assert client.find_filing(123, "10-K", 1999) is None


def test_older_pages_are_searched_when_recent_misses(monkeypatch):
    """`filings.recent` caps around 1000 filings.

    For a high-volume filer that is a short window — JPMorgan's covers under two
    years — so a corpus reaching back to 2015 must page through the older files
    or it silently loses a third of its documents.
    """
    main = _submissions([("10-K", "new", "2025-12-31", "new.htm")])
    main["filings"]["files"] = [
        {"name": "CIK-old-001.json", "filingFrom": "2014-01-01", "filingTo": "2018-12-31"}
    ]
    older = _submissions([("10-K", "old", "2016-12-31", "old.htm")])["filings"]["recent"]

    fetched: list[str] = []

    def _get(self, url, attempts=3):
        fetched.append(url)
        payload = older if "old-001" in url else main
        return json.dumps(payload if "old-001" not in url else older).encode()

    monkeypatch.setattr(EdgarClient, "_get", _get)
    monkeypatch.setattr(EdgarClient, "_throttle", lambda self: None)

    client = EdgarClient(user_agent=UA)
    filing = client.find_filing(123, "10-K", 2016)

    assert filing is not None
    assert filing.accession_number == "old"
    assert any("old-001" in url for url in fetched)


def test_older_pages_outside_the_target_range_are_skipped(monkeypatch):
    """JPMorgan has 69 older pages. Loading them all to find one 10-K would
    waste most of the rate budget for a company needing three documents."""
    main = _submissions([("10-K", "new", "2025-12-31", "new.htm")])
    main["filings"]["files"] = [
        {"name": "far-past.json", "filingFrom": "1995-01-01", "filingTo": "1999-12-31"},
        {"name": "in-range.json", "filingFrom": "2015-01-01", "filingTo": "2019-12-31"},
    ]
    fetched: list[str] = []

    def _get(self, url, attempts=3):
        fetched.append(url)
        if "in-range" in url:
            return json.dumps(
                _submissions([("10-K", "hit", "2016-12-31", "d.htm")])["filings"]["recent"]
            ).encode()
        return json.dumps(main).encode()

    monkeypatch.setattr(EdgarClient, "_get", _get)
    monkeypatch.setattr(EdgarClient, "_throttle", lambda self: None)

    EdgarClient(user_agent=UA).find_filing(123, "10-K", 2016)

    assert any("in-range" in url for url in fetched)
    assert not any("far-past" in url for url in fetched)


def test_recent_hit_does_not_load_older_pages(monkeypatch):
    """The common case must stay at one request."""
    main = _submissions([("10-K", "new", "2025-12-31", "new.htm")])
    main["filings"]["files"] = [
        {"name": "old.json", "filingFrom": "2015-01-01", "filingTo": "2019-12-31"}
    ]
    fetched: list[str] = []

    def _get(self, url, attempts=3):
        fetched.append(url)
        return json.dumps(main).encode()

    monkeypatch.setattr(EdgarClient, "_get", _get)
    monkeypatch.setattr(EdgarClient, "_throttle", lambda self: None)

    EdgarClient(user_agent=UA).find_filing(123, "10-K", 2025)
    assert not any("old.json" in url for url in fetched)


def test_undated_older_page_is_not_ruled_out(monkeypatch):
    assert EdgarClient._page_could_cover({"name": "x"}, 2016) is True
    assert EdgarClient._page_could_cover(
        {"filingFrom": "2015-01-01", "filingTo": "2019-12-31"}, 2016
    )
    assert not EdgarClient._page_could_cover(
        {"filingFrom": "1995-01-01", "filingTo": "1999-12-31"}, 2016
    )


def test_form_type_is_respected(client):
    assert client.find_filing(123, "8-K", 2022) is not None
    assert len(client.list_filings(123, "10-K")) == 2


def test_submissions_are_cached(monkeypatch):
    """One company supplies several documents; refetching wastes rate budget."""
    calls: list[str] = []
    c = EdgarClient(user_agent=UA)

    def _get(self, url):
        calls.append(url)
        return json.dumps(_submissions([("10-K", "a", "2022-12-31", "d.htm")])).encode()

    monkeypatch.setattr(EdgarClient, "_get", _get)
    monkeypatch.setattr(EdgarClient, "_throttle", lambda self: None)

    c.submissions(123)
    c.submissions(123)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
def test_document_url_strips_dashes_from_the_accession():
    filing = EdgarFiling(
        cik=66740,
        accession_number="0000066740-23-000014",
        form="10-K",
        filing_date="2023-02-08",
        report_date="2022-12-31",
        primary_document="mmm-20221231.htm",
    )
    assert filing.document_url == (
        "https://www.sec.gov/Archives/edgar/data/66740/000006674023000014/mmm-20221231.htm"
    )
    assert filing.report_year == 2022


# ---------------------------------------------------------------------------
# CIK map
# ---------------------------------------------------------------------------
def test_loads_the_reviewed_cik_map(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"3M": {"cik": 66740, "edgar_name": "3M CO"}}), encoding="utf-8")
    assert load_company_cik_map(path) == {"3M": 66740}


def test_repo_cik_map_is_present_and_correct(repo_root):
    """A committed, reviewed mapping — not fuzzy matching at runtime.

    Automatic name matching resolved "AMD" to Amdocs, which would have fetched a
    different company's filings with nothing downstream able to detect it.
    """
    mapping = load_company_cik_map(repo_root / "data" / "reference" / "company_cik.json")
    assert len(mapping) == 32
    assert mapping["AMD"] == 2488, "AMD must be Advanced Micro Devices, not Amdocs"
    assert mapping["3M"] == 66740
    assert mapping["Activision Blizzard"] == 718877
    assert mapping["Foot Locker"] == 850209


# ---------------------------------------------------------------------------
# Corpus planning and failure handling
# ---------------------------------------------------------------------------
def test_plan_lists_distinct_documents(tmp_path):
    rows = [
        {"doc_name": "ACME_2022_10K", "company": "ACME", "doc_link": "http://x"},
        {"doc_name": "ACME_2022_10K", "company": "ACME", "doc_link": "http://x"},
        {"doc_name": "ACME_2021_10K", "company": "ACME", "doc_link": "http://y"},
    ]
    path = tmp_path / "fb.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    plan = plan_corpus_fetch(path)
    assert [p.document_name for p in plan] == ["ACME_2021_10K", "ACME_2022_10K"]


def test_earnings_releases_are_reported_not_attempted(tmp_path, monkeypatch):
    """Not SEC filings — must appear in coverage rather than silently vanish."""
    path = tmp_path / "fb.jsonl"
    path.write_text(
        json.dumps({"doc_name": "ACME_2022_EARNINGS", "company": "ACME", "doc_link": "http://x"}),
        encoding="utf-8",
    )
    cik_path = tmp_path / "map.json"
    cik_path.write_text(json.dumps({"ACME": {"cik": 123}}), encoding="utf-8")

    report = fetch_financebench_corpus(path, tmp_path / "out", cik_path, user_agent=UA)
    assert len(report.failed) == 1
    assert "not_an_sec_filing" in report.failed[0].reason


def test_unknown_company_is_reported(tmp_path):
    path = tmp_path / "fb.jsonl"
    path.write_text(
        json.dumps({"doc_name": "GHOST_2022_10K", "company": "Ghost", "doc_link": None}),
        encoding="utf-8",
    )
    cik_path = tmp_path / "map.json"
    cik_path.write_text(json.dumps({}), encoding="utf-8")

    report = fetch_financebench_corpus(path, tmp_path / "out", cik_path, user_agent=UA)
    assert "unknown_company" in report.failed[0].reason


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------
def _report() -> CorpusFetchReport:
    return CorpusFetchReport(
        results=[
            FilingFetchResult("A_2022_10K", succeeded=True),
            FilingFetchResult("B_2022_10K", succeeded=False, reason="filing_not_found: x"),
            FilingFetchResult("C_2022_EARNINGS", succeeded=False, reason="not_an_sec_filing: y"),
        ]
    )


def test_coverage_is_computed():
    report = _report()
    assert report.coverage == pytest.approx(1 / 3)
    assert len(report.succeeded) == 1


def test_failure_reasons_are_grouped():
    assert _report().failure_reasons() == {"filing_not_found": 1, "not_an_sec_filing": 1}


def test_coverage_report_names_every_missing_document(tmp_path):
    """A silent gap is the thing this must prevent."""
    path = _report().write(tmp_path / "coverage.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["n_obtained"] == 1
    assert payload["n_requested"] == 3
    missing = {m["document"] for m in payload["missing"]}
    assert missing == {"B_2022_10K", "C_2022_EARNINGS"}
    assert all(m["reason"] for m in payload["missing"])


def test_summary_mentions_coverage():
    assert "33%" in _report().summary()
