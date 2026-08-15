"""Replay behavior policies over the outcome matrix to produce logged feedback.

    python scripts/simulate_bandit_logs.py --config configs/experiments/ope_simulation.yaml

Reveals only the reward of the action each behavior policy selected, reproducing
the partial feedback of production while retaining the full matrix so estimates
can later be checked against known values.

This is a controlled logged-feedback simulation constructed from full-information
benchmark outcomes. It is not production user feedback and not a deployed online
RL system — the distinction matters in every write-up.

Cheap: no model calls, only replay. Run it as often as needed.
"""

from __future__ import annotations

from _common import run_cli

if __name__ == "__main__":
    run_cli("bandit", "simulate")
