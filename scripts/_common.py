"""Shared helper for the script entry points.

Scripts stay thin on purpose: they delegate to the Typer CLI, which delegates to
the package. Logic that lives only in a script is logic that cannot be unit
tested (spec section 18, repository-structure principles).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running the scripts directly from a checkout without installing.
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_cli(*args: str) -> None:
    """Invoke the `evidence-route` CLI with the given arguments."""
    from evidence_route.cli import app

    app(list(args) + sys.argv[1:], standalone_mode=True)
