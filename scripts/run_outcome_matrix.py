"""Execute every candidate action on every question — the outcome matrix.

    python scripts/run_outcome_matrix.py --experiment configs/experiments/mvp.yaml --dry-run
    python scripts/run_outcome_matrix.py --experiment configs/experiments/mvp.yaml

THIS IS THE EXPENSIVE STEP. Cost scales as questions x actions x replicates.
Run `--dry-run` first: it estimates token use and spend on a subset before
anything is billed, and the run aborts rather than crossing the configured
budget ceiling.

The resulting matrix is what makes the oracle upper bound computable and gives
the offline policy evaluation study a ground truth to check estimates against.
Failed question-action pairs are recorded with an explicit error state, never
dropped — silently discarding them would bias the matrix toward actions that
fail loudly on hard questions.
"""

from __future__ import annotations

from _common import run_cli

if __name__ == "__main__":
    run_cli("outcomes", "run")
