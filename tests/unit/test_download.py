"""Tests for dataset downloading.

No network access: the transport is patched, so these verify the behaviour that
protects the data — atomicity, idempotence, checksum enforcement and the offline
guarantee — rather than that HuggingFace is reachable.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error

import pytest

from evidence_route.datasets.download import (
    DATASET_SOURCES,
    download_file,
    ensure_dataset,
)

PAYLOAD = b'{"financebench_id": "x", "question": "q?"}\n'
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Patch urlopen, recording how many times it was called."""
    calls: list[str] = []

    def _open(request, timeout=None):
        calls.append(getattr(request, "full_url", str(request)))
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr("urllib.request.urlopen", _open)
    return calls


# ---------------------------------------------------------------------------
# Basic download
# ---------------------------------------------------------------------------
def test_downloads_and_reports(tmp_path, fake_urlopen):
    dest = tmp_path / "nested" / "data.jsonl"
    result = download_file("https://example.invalid/data.jsonl", dest)

    assert result.downloaded
    assert dest.read_bytes() == PAYLOAD
    assert result.bytes_written == len(PAYLOAD)
    assert result.checksum == f"sha256:{PAYLOAD_SHA}"
    assert len(fake_urlopen) == 1


def test_creates_missing_parent_directories(tmp_path, fake_urlopen):
    dest = tmp_path / "a" / "b" / "c" / "data.jsonl"
    download_file("https://example.invalid/d", dest)
    assert dest.exists()


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------
def test_existing_file_is_not_refetched(tmp_path, fake_urlopen):
    """Re-running `data prepare` should be free."""
    dest = tmp_path / "data.jsonl"
    dest.write_bytes(PAYLOAD)

    result = download_file("https://example.invalid/d", dest)
    assert not result.downloaded
    assert fake_urlopen == []


def test_force_refetches(tmp_path, fake_urlopen):
    dest = tmp_path / "data.jsonl"
    dest.write_bytes(b"stale")
    result = download_file("https://example.invalid/d", dest, force=True)
    assert result.downloaded
    assert dest.read_bytes() == PAYLOAD


def test_existing_file_failing_its_checksum_is_replaced(tmp_path, fake_urlopen):
    """A stale or corrupt local copy must not be preserved."""
    dest = tmp_path / "data.jsonl"
    dest.write_bytes(b"corrupt")
    result = download_file("https://example.invalid/d", dest, expected_sha256=PAYLOAD_SHA)
    assert result.downloaded
    assert dest.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Checksum enforcement
# ---------------------------------------------------------------------------
def test_matching_checksum_is_accepted(tmp_path, fake_urlopen):
    dest = tmp_path / "data.jsonl"
    assert download_file("https://example.invalid/d", dest, expected_sha256=PAYLOAD_SHA).downloaded


def test_checksum_mismatch_raises_and_leaves_nothing_behind(tmp_path, fake_urlopen):
    """A changed upstream file must stop the run.

    Results produced against a different version of a dataset are not
    comparable, so this fails loudly rather than adopting the new file.
    """
    dest = tmp_path / "data.jsonl"
    with pytest.raises(OSError, match="Checksum mismatch"):
        download_file("https://example.invalid/d", dest, expected_sha256="0" * 64)

    assert not dest.exists()
    assert list(tmp_path.glob("*.partial")) == []


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------
def test_failed_download_leaves_no_partial_file(tmp_path, monkeypatch):
    """An interrupted download must not leave a truncated file at the
    destination — the next run would find it and prepare a partial dataset."""

    def _boom(request, timeout=None):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    dest = tmp_path / "data.jsonl"
    with pytest.raises(OSError, match="Could not reach"):
        download_file("https://example.invalid/d", dest)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_good_file_survives_a_failed_forced_refetch(tmp_path, monkeypatch):
    def _boom(request, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    dest = tmp_path / "data.jsonl"
    dest.write_bytes(PAYLOAD)
    with pytest.raises(OSError):
        download_file("https://example.invalid/d", dest, force=True)

    assert dest.read_bytes() == PAYLOAD


def test_empty_response_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: io.BytesIO(b""))
    dest = tmp_path / "data.jsonl"
    with pytest.raises(OSError, match="zero bytes"):
        download_file("https://example.invalid/d", dest)
    assert not dest.exists()


def test_http_error_is_reported_with_its_status(tmp_path, monkeypatch):
    def _http_error(request, timeout=None):
        raise urllib.error.HTTPError("https://example.invalid/d", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _http_error)
    with pytest.raises(OSError, match="HTTP 404"):
        download_file("https://example.invalid/d", tmp_path / "d.jsonl")


# ---------------------------------------------------------------------------
# ensure_dataset and the offline guarantee
# ---------------------------------------------------------------------------
def test_ensure_dataset_fetches_the_registered_source(tmp_path, fake_urlopen):
    result = ensure_dataset("financebench", tmp_path)
    assert result.downloaded
    assert result.path.name == "financebench_merged.jsonl"
    assert "huggingface.co" in fake_urlopen[0]


def test_offline_mode_refuses_to_download(tmp_path):
    """The offline guarantee has to hold on the cache-miss path.

    That is the only path where breaking it would matter, and the only one
    where a stray network call would go unnoticed.
    """
    with pytest.raises(RuntimeError, match="offline mode is enabled"):
        ensure_dataset("financebench", tmp_path, offline=True)


def test_offline_mode_accepts_an_already_present_file(tmp_path):
    (tmp_path / "financebench_merged.jsonl").write_bytes(PAYLOAD)
    result = ensure_dataset("financebench", tmp_path, offline=True)
    assert not result.downloaded
    assert result.checksum.startswith("sha256:")


def test_unregistered_dataset_reports_what_is_available(tmp_path):
    with pytest.raises(NotImplementedError, match="No download source registered"):
        ensure_dataset("not_a_dataset", tmp_path)


def test_financebench_source_is_the_huggingface_dataset():
    """The GitHub repo named in the spec returns 404; HuggingFace replaced it."""
    source = DATASET_SOURCES["financebench"]
    assert "huggingface.co" in source.url
    assert source.license == "CC-BY-NC-4.0"
    assert source.filename == "financebench_merged.jsonl"
