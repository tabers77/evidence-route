"""Dataset downloading (spec section 19, ``data prepare``).

Only *metadata* is downloaded here — questions, answers, evidence references and
relevance labels. Source documents are a separate problem with a separate
strategy; see :mod:`evidence_route.datasets` for the split between them.

Three properties this module is built around:

**Atomic writes.** Downloads stream to a temporary file and are renamed into
place only on success. An interrupted download must never leave a truncated file
sitting at the destination path, because the next run would find it, treat it as
present, and silently prepare a partial dataset.

**Idempotence.** An existing file is left alone unless its checksum fails or the
caller forces a refresh. Re-running ``data prepare`` should be free.

**Offline honesty.** Downloading is refused outright in offline mode rather than
quietly attempted, so the offline guarantee the smoke experiment depends on
cannot be broken by a code path that reaches the network on a cache miss.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from evidence_route.evaluation.reproducibility import file_checksum

__all__ = [
    "DATASET_SOURCES",
    "DatasetSource",
    "DownloadResult",
    "download_file",
    "ensure_dataset",
]

_USER_AGENT = "evidence-route/0.1 (research benchmark; +https://github.com/tabers77/evidence-route)"
_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class DatasetSource:
    """Where a dataset's metadata comes from."""

    name: str
    url: str
    filename: str
    license: str
    #: Recorded when known so a changed upstream file is detected rather than
    #: silently adopted. Left None until a release is pinned — see the note in
    #: DATASET_SOURCES.
    sha256: str | None = None
    notes: str = ""


#: Verified 2026-08-15.
#:
#: The GitHub repository named in the specification
#: (https://github.com/patronus-ai/financebench) returns 404; FinanceBench is
#: distributed on HuggingFace instead. That the canonical source of a public
#: benchmark can disappear is exactly why manifests record checksums.
#:
#: Checksums are deliberately not pinned yet. Pinning one now would freeze
#: whatever the file happens to be today; the manifest records the checksum of
#: what was actually downloaded, and a hash is pinned here once the protocol is
#: frozen (spec section 25).
DATASET_SOURCES: dict[str, DatasetSource] = {
    "financebench": DatasetSource(
        name="financebench",
        url=(
            "https://huggingface.co/datasets/PatronusAI/financebench/"
            "resolve/main/financebench_merged.jsonl"
        ),
        filename="financebench_merged.jsonl",
        license="CC-BY-NC-4.0",
        notes=(
            "Question metadata only — 150 OPEN_SOURCE records across 32 "
            "companies and 84 documents. Contains no PDFs; source filings must "
            "be fetched separately from each record's doc_link."
        ),
    ),
}


@dataclass
class DownloadResult:
    """What a download attempt did."""

    path: Path
    downloaded: bool
    bytes_written: int
    checksum: str
    source_url: str | None = None

    @property
    def action(self) -> str:
        return "downloaded" if self.downloaded else "already present"


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
) -> DownloadResult:
    """Download ``url`` to ``destination``, atomically.

    Returns without re-fetching if the file already exists and either no
    checksum is expected or the existing file matches it.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        existing = file_checksum(destination)
        if expected_sha256 is None or existing == f"sha256:{expected_sha256}":
            return DownloadResult(
                path=destination,
                downloaded=False,
                bytes_written=destination.stat().st_size,
                checksum=existing,
            )
        # A mismatch means the local copy is stale or corrupt. Re-fetch rather
        # than proceed with data that is not what the manifest will claim.

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    # Written beside the destination so the final rename stays on one
    # filesystem; a cross-device rename is not atomic.
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".partial"
    )
    tmp_path = Path(tmp_name)
    written = 0
    try:
        with open(tmp_fd, "wb") as out:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                shutil.copyfileobj(response, out)
        written = tmp_path.stat().st_size
        if written == 0:
            raise OSError(f"Downloaded zero bytes from {url}")

        checksum = file_checksum(tmp_path)
        if expected_sha256 and checksum != f"sha256:{expected_sha256}":
            raise OSError(
                f"Checksum mismatch for {url}\n"
                f"  expected sha256:{expected_sha256}\n"
                f"  got      {checksum}\n"
                f"The upstream file has changed. Investigate before continuing — "
                f"results produced against a different version of a dataset are "
                f"not comparable."
            )

        tmp_path.replace(destination)
    except urllib.error.HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        raise OSError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise OSError(f"Could not reach {url}: {exc.reason}") from exc
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return DownloadResult(
        path=destination,
        downloaded=True,
        bytes_written=written,
        checksum=checksum,
        source_url=url,
    )


def ensure_dataset(
    dataset: str,
    raw_dir: Path,
    *,
    force: bool = False,
    offline: bool = False,
) -> DownloadResult:
    """Ensure a dataset's metadata file is present locally.

    Raises in offline mode when the file is absent, rather than reaching for the
    network — the offline guarantee has to hold on the cache-miss path, which is
    the only path where breaking it would matter.
    """
    source = DATASET_SOURCES.get(dataset)
    if source is None:
        known = ", ".join(sorted(DATASET_SOURCES)) or "none"
        raise NotImplementedError(
            f"No download source registered for {dataset!r}. Registered: {known}. "
            f"Fetch it manually into {raw_dir} or add a DatasetSource entry."
        )

    destination = raw_dir / source.filename

    if offline:
        if destination.exists():
            return DownloadResult(
                path=destination,
                downloaded=False,
                bytes_written=destination.stat().st_size,
                checksum=file_checksum(destination),
            )
        raise RuntimeError(
            f"{source.filename} is missing and offline mode is enabled, so it "
            f"cannot be downloaded. Either unset EVIDENCE_ROUTE_OFFLINE or place "
            f"the file at {destination} manually.\nSource: {source.url}"
        )

    return download_file(
        source.url,
        destination,
        expected_sha256=source.sha256,
        force=force,
    )
