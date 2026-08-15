"""Build reproducible tables, figures and the static HTML report.

    python scripts/build_report.py --experiment-id final-v1

Every emitted number carries the experiment id and the commit SHA that produced
it. A run from a dirty worktree is marked as illustrative rather than reproduced
— see `RunManifest.is_publishable`.
"""

from __future__ import annotations

from _common import run_cli

if __name__ == "__main__":
    run_cli("report", "build")
