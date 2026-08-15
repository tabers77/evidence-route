"""Download, validate, checksum and split a benchmark dataset.

    python scripts/download_data.py --config configs/datasets/financebench.yaml

Writes a manifest recording the source URL, license, download timestamp and
per-file checksums. Source documents land under `data/raw/` and are never
committed — see `data/README.md`.
"""

from __future__ import annotations

from _common import run_cli

if __name__ == "__main__":
    run_cli("data", "prepare")
